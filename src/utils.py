import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math as m
from PIL import Image
import copy, os, argparse, sys

# maxshift maxscale meanshift meanscale medianshift medianscale
path = sys.argv[2]

def parse_sysargs():
    """
    Parse arguments of the form --KEY VALUE from sys.argv,
    lowercase the keys, and inject them as variables.
    Example: --NUM_CLASSES 3 → num_classes = 3
    """
    args = sys.argv[1:]  # skip script name
    parsed = {}

    for i in range(0, len(args), 2):  # step through pairs
        key = args[i].lstrip("-")   # strip '--'
        val = args[i+1]

        # Try to convert to int or float automatically
        if val.isdigit():
            val = int(val)
        else:
            try:
                val = float(val)
            except ValueError:
                pass  # leave as string

        parsed[key] = val

    return parsed



# make sure loss between class is the same
# (not what we want? Might need larger gradients to even out initialization differences)
def loss_with_per_class_gap(pred, target, lambda_fair=0.1):
    """
    pred: (N, C) raw logits
    target: (N, C) one-hot labels
    """
    num_classes = target.size(1)

    print("pred target",pred, target)
    
    # Standard MSE
    mse_loss = F.mse_loss(pred, target)

    # Per-class average MSE
    per_class_losses = []
    labels = target.argmax(dim=1)
    print("labels", labels)
    for k in range(num_classes):
        mask = (labels == k)  # select samples of class k
        print("mask at k", mask, k)
        if mask.any():
            # FIXME VERIFY THIS IS SELECTING CLASSES CORRECTLY
            loss_k = F.mse_loss(pred[mask], target[mask])
            per_class_losses.append(loss_k)
        else:
            # No samples of this class in the batch
            per_class_losses.append(torch.tensor(0.0, device=pred.device))

    per_class_losses = torch.stack(per_class_losses)

    print("perclasslosses", per_class_losses)
    
    # Fairness penalty = gap between worst and best class
    fairness_loss = per_class_losses.max() - per_class_losses.min()

    # Combine
    total_loss = ((1-lambda_fair) * mse_loss) + (lambda_fair * fairness_loss)
    return total_loss, mse_loss, fairness_loss


# FIXME VERIFY THSI ONE TOO
# The gap in the mean softmax indicates confidence. Want even confidence in classification (good metric)
def loss_with_soft_accuracy_gap(pred, target, lambda_fair=0.1):
    """
    pred: (N, C) raw logits
    target: (N, C) one-hot labels
    """
    num_classes = target.size(1)
    
    # Standard MSE
    mse_loss = F.mse_loss(pred, target)

    # Convert to probabilities
    probs = F.softmax(pred, dim=1)
    print("probs", probs)
    labels = target.argmax(dim=1)

    # Per-class mean confidence
    class_confidences = []
    for k in range(num_classes):
        mask = (labels == k)
        if mask.any():
            conf_k = probs[mask, k].mean()
            class_confidences.append(conf_k)
        else:
            class_confidences.append(torch.tensor(0.0, device=pred.device))

    class_confidences = torch.stack(class_confidences)

    # Fairness penalty = confidence gap
    fairness_loss = class_confidences.max() - class_confidences.min()

    # Combine
    total_loss = mse_loss + lambda_fair * fairness_loss
    return total_loss, mse_loss, fairness_loss


def evo_weights(loops, pop_size, weights, data, model):
    pop = generate_population(weights, pop_size)
    for _ in range(loops):
        eval_pop(data, pop)



def generate_population(weights, pop_size):
    population = []
    population.append(weights)
    for i in range(pop_size):
        population.append(mutate(weights))
    return population

def mutate(population):
    # randomly mutate an entry with probability 1/6 (inject random noise)
    for i in population:
        for j in i:
            x = torch.randint(0, 6)
            if not x:
                j += torch.randn(1)[0]


def eval_pop(model, data, population):
    ranked = []
    # for i in population, plug in weights and eval accuracy using same functions
    # my guess is it's just going to optimize toward 0 unless I add another penalty term
    # so observe iti

    # torunament selectiom, divide into maybe 10
    # choose k (the tournament size) individuals from the population at random
    # choose the best individual from the tournament with probability p
    # choose the second best individual with probability p*(1-p)
    # choose the third best individual with probability p*((1-p)^2)
    # and so on
    for i in population:
        loss = model(data, )
        pass

def reproduction(population):
    
    pass



def compute_accuracies(num_classes, train_samples, test_samples, num_seeds, num_training_steps, train_dict, test_dict):
    """
    Computes average classification accuracies for each class
    """
    for i in range(num_classes):
        train_div = train_samples[i]*num_seeds
        test_div = test_samples[i]*num_seeds
        for j in range(num_training_steps):
            train_dict[i][j] /= train_div
            test_dict[i][j] /= test_div


def generate_samples(means, covs, num_classes, samples_per_class):
    """
    Generates input samples from 2D Gaussians.
    """
    X = []
    for class_id in range(num_classes):
        dist = torch.distributions.MultivariateNormal(means[class_id], covs[class_id])
        X += [dist.sample() for _ in range(samples_per_class)]
    return X


def create_labels(num_classes, samples_per_class, classes):
    """
    Creates corresponding class labels for generated input data.
    """
    Y = []
    for i in range(num_classes):
        Y += [classes[i] for _ in range(samples_per_class)]
    return Y


def initialize_accuracy_tracking(num_classes, num_training_steps, num_seeds):
    """
    Initializes dicts for tracking average and per seed classification accuracy.
    """
    train_dict = {i: [0] * num_training_steps for i in range(num_classes)}
    test_dict = {i: [0] * num_training_steps for i in range(num_classes)}
    per_seed = {
        'train': [[[] for _ in range(num_training_steps)] for _ in range(num_seeds)],
        'test':  [[[] for _ in range(num_training_steps)] for _ in range(num_seeds)]
    }
    return train_dict, test_dict, per_seed


def count_samples(data, key):
    """
    Counts how many samples belong to each class, used to calculate classification accuracy.
    """
    # counting loop. tmp index for each class
    tmp = [0] * len(key)

    for i in data:
        tmp[torch.argmax(i)] += 1

    return tmp


def scale_samples(x, y, scalars, decay_param):
    """
    Scales input samples toward target norms using provided scalars.

    Args:
        x (list[Tensor]): Input features.
        y (list[Tensor]): Corresponding one-hot labels.
        scalars (Tensor): Scalar per class. Shape: (num_classes,)
        decay_param (float): Strength of transformation [0, 1].
    """
    for i in range(len(x)):
        class_idx = torch.argmax(y[i]).item()
        scale = scalars[class_idx]
        x[i] = ((x[i] * scale) * decay_param) + (x[i] * (1 - decay_param))


def shift_samples(x, y, shift, decay_param):
    """
    Shifts input samples toward a class-specific target location.

    Args:
        x (list[Tensor]): Input features.
        y (list[Tensor]): Corresponding one-hot labels.
        shift (Tensor): Shift vectors for each class. Shape: (num_classes, 2)
        decay_param (float): Strength of shift [0, 1].
    """
    for i in range(len(x)):
        class_idx = torch.argmax(y[i]).item()
        x[i] = ((x[i] + shift[class_idx]) * decay_param) + (x[i] * (1 - decay_param))


# Function to track the total number of correct classifications at each step
def track_accuracy(predictions, step, dict, data, n_classes, per_seed, samples, seed, key):
    tmp_arr = [0] * n_classes
    tmp = 0
    # Check how many correct classifications for each class
    for i in range(len(predictions)):
        idx = torch.argmax(predictions[i])
        if (idx == torch.argmax(data[i])):
            tmp_arr[idx] += 1

    # Add number of correctly clasified test data to correct step spot in accuracy_dict
    for i in range(n_classes):
        dict[i][step] += tmp_arr[i]
        per_seed[key][seed][step].append(tmp_arr[i] / samples[i])
        tmp += (tmp_arr[i] / samples[i])

    return tmp / n_classes


def make_evenly_spaced_targets(num_points, radius=1.0):
    """
    Generate N evenly spaced points on a circle centered at origin.
    
    Args:
        num_points (int): Number of target points.
        radius (float): Radius of the circle.

    Returns:
        Tensor of shape (num_points, 2) with 2D coordinates.
    """
    # odd number of classes starts at top for symmetry
    if (num_points % 2):
        start_angle = m.pi / 2
    else:
        start_angle = (m.pi / 2) + (m.pi / num_points)

    angles = torch.linspace(0, 2 * m.pi, steps=num_points + 1)[:-1]  + start_angle # exclude endpoint
    x = radius * torch.cos(angles)
    y = radius * torch.sin(angles)
    means = torch.stack([x, y], dim=1)

    means = sorted(means, key=lambda p: (-p[1].item(), p[0].item()))

    return torch.stack(means)


def transform_to_even_space(means, mode='shift', ref_mode='mean'):
    """
    Transforms a set of 2D class means to be evenly spaced around the origin.
    
    Args:
        means (Tensor): Tensor of shape (N, 2), N is number of classes.
        mode (str): 'shift' or 'scale'. Shift vectors to ETF or scale for same norms.
        ref_mode (str): 'mean', 'max', or 'median' norm to use for shift/scale radius.

    Returns:
        Tensor: 
          - If mode='shift': target positions (N, 2)
          - If mode='scale': scaling factors (N,)
    """
    num_points = means.shape[0]
    
    # Compute norms for each mean
    norms = torch.linalg.norm(means, dim=1)
    
    # Choose reference radius
    if ref_mode == 'mean':
        radius = norms.mean().item()
    elif ref_mode == 'max':
        radius = norms.max().item()
    elif ref_mode == 'median':
        radius = norms.median().item()
    else:
        raise ValueError("ref_mode must be 'mean', 'max', or 'median'")
    
    # Generate evenly spaced target points
    targets = make_evenly_spaced_targets(num_points, radius)
    print(f'targets: {targets}')

    if mode == 'shift':
        # Return target positions to shift means toward
        return targets - means

    elif mode == 'scale':
        # Scale each mean vector to match target norm (1e-9 for stability)
        scalars = radius / (norms + 1e-9)
        return scalars

    else:
        raise ValueError("mode must be 'shift' or 'scale'")



def is_regular_polygon(points, tol=1e-9):
    """
    Check if N 2D points form a regular polygon (equal side lengths).
    
    Args:
        points (Tensor): Tensor of shape (N, 2)
        tol (float): Tolerance for distance comparison
        
    Returns:
        bool: True if all pairwise distances between adjacent points are equal.
    """
    num_points = points.shape[0]
    if num_points < 3:
        raise ValueError('Need at least 3 points to form an ETF!')  # Need at least 3 points to form a polygon

    # Compute squared distances between each adjacent pair (circularly)
    distances = []
    for i in range(num_points):
        p1 = points[i]
        p2 = points[(i + 1) % num_points]  # wrap around
        dist_sq = torch.sum((p1 - p2) ** 2)
        distances.append(dist_sq)
    
    distances = torch.stack(distances)

    # Check that all distances are close to the first one (within tolerance)
    return torch.all(torch.isclose(distances, distances[0], atol=tol))
