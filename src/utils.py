import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math as m
from models import LinearClassifier
from PIL import Image
import copy, os, argparse, sys
from typing import Iterable, List, Tuple


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
    total_loss = mse_loss + (lambda_fair * fairness_loss)
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
            print("confk", conf_k)
            class_confidences.append(conf_k)
        else:
            class_confidences.append(torch.tensor(0.0, device=pred.device))

    class_confidences = torch.stack(class_confidences)
    print("classconf", class_confidences)

    # Fairness penalty = confidence gap
    fairness_loss = class_confidences.max() - class_confidences.min()
    print("fairloss", fairness_loss)

    # Combine
    total_loss = mse_loss + (lambda_fair * fairness_loss)
    return total_loss, mse_loss, fairness_loss


def evo_weights(num_iter, pop_size, tournament_size, weights, means):
    num_groups = pop_size / tournament_size
    if (isinstance(num_groups, float)):
        print("pop size must be divisible by tournament size")
        exit(-1)
    for iter in range(num_iter):
        winners = []
        pop = manage_population(weights, pop_size, num_groups, iter)
        #pop now sorted
        for j in range(num_groups):
            population_sorted, entropy_sorted = eval_pop(means, pop)
            winners.append([population_sorted[0]])
        pop = copy.deepcopy(winners)
        # reproduce()
    pop, entropy_sorted = eval_pop(means, pop)
    return pop[0].clone()


def manage_population(weights, pop_size, num_groups, iter=0):
    if not iter:
        population = [[] for _ in range(num_groups)]
        for i in range(pop_size):
            torch.manual_seed(i)
            model = LinearClassifier(2, 3)
            population[i % num_groups].append(mutate(model.linear.weight.data, i))
            # population.append(mutate(weights))
        return population
    else:
        # massive problems this is so confusing i should just design a repeatable loop that is clear
        for i in range(pop_size-num_groups):
            x = torch.randint(1)
            torch.manual_seed(i)
            model = LinearClassifier(2, 3)
            if x > 0.1:
                weights[i % num_groups].append(model.linear.weight.data)
            else:
                weights[i % num_groups].append(mutate(weights[i%num_groups][0]))

        return weights


def mutate(weights):
    # randomly mutate an entry with probability 1/6 (inject random noise)
    for i in weights:
        for j in i:
            x = torch.randint(0, 6)
            if not x:
                j += torch.randn(1)[0]
    return weights


# minimize entropy. Take the means, put them into weights matrices.
# Will get 3 outputs and want max entries of each outputs to be the same.
# That is how they will be ranked? Because softmax is how confident?
# Can be confidently wrong? Is that fine? Could also compare to true?
# So if max-min good and all 3 are correct corresponding classes then you are good, otherwise no.

#OLD
# def eval_pop(means, population):
#     r1, r2 = [], []
#     labels = torch.tensor([1,0,0],[0,1,0],[0,0,1])
#     for i in population:
#         for j in means:
#             tmp = i @ j
#             r1.append(max(F.softmax(tmp)))
#             r1[-1] += labels[tmp.index(r1[-1])]
#         m = r1.mean()
#         for k in r1:
#             k -= m
#             k = abs(k)
#     r2.append(sum(r1)) 
#     # wrong because sorts r2 and not corresponding population entry
#     r2 = sorted(r2)




# FIXME does this work as intended?

# def eval_pop(means, population):
#     labels = [torch.tensor([1,0,0]),torch.tensor([0,1,0]),torch.tensor([0,0,1])]

#     print("labels", labels)

#     # FIXME sort the population
#     entropy_scores = []

#     for candidate in population:  
#         confidences = []
#         for mean_vec in means:
#             scores = candidate @ mean_vec             # raw logits
#             probs = F.softmax(scores, dim=0)          # convert to probabilities
#             max_prob, pred_class = torch.max(probs, dim=0)
            
#             # Adjust with label (if needed)
#             confidences.append(max_prob.item() + labels[pred_class.item()])
        
#         # Centering and absolute deviations
#         mean_conf = sum(confidences) / len(confidences)
#         deviations = [abs(c - mean_conf) for c in confidences]
        
#         entropy_scores.append(sum(deviations))

#     # Sort population by entropy
#     scored_population = sorted(zip(entropy_scores, population), key=lambda x: x[0])
#     entropy_scores_sorted, population_sorted = zip(*scored_population)
#     print("sorted", population_sorted)


#     return population_sorted

#FIXME didnt work. Verify get same result with old one. Show entropy of best weight matrix to verify as well taht it is very low

def eval_pop(means: Iterable[torch.Tensor],
             population: Iterable[torch.Tensor],
             return_best_only: bool = False
            ) -> Tuple[List[torch.Tensor], List[float]]:
    """
    Evaluate population and sort by entropy (low -> high).
    - means: iterable of mean vectors, each shape (feature_dim,)
    - population: iterable of candidate weight matrices, each shape (num_classes, feature_dim)
    Returns (population_sorted_list, entropy_scores_sorted_list).
    If return_best_only True, returns ([best_candidate], [best_score])
    """

    entropies = []

    # Parameters
    eps = 1e-12

    for candidate in population:
        # candidate @ mean -> logits for classes
        # compute entropy H = -sum(p * log p) for each mean, then average across means
        Hs = []
        for mean_vec in means:
            logits = candidate @ mean_vec            # shape: (num_classes,)
            probs = F.softmax(logits, dim=0)        # shape: (num_classes,)
            # Shannon entropy (scalar tensor)
            H = - (probs * (probs + eps).log()).sum()
            Hs.append(float(H.item()))               # convert to Python float

        # choose metric: mean entropy across mean vectors
        entropy_score = float(sum(Hs) / len(Hs))
        entropies.append(entropy_score)

    # pair and sort (low entropy first)
    paired = sorted(zip(entropies, population), key=lambda pair: pair[0])
    entropies_sorted, population_sorted = zip(*paired)

    population_sorted = list(population_sorted)
    entropies_sorted = list(entropies_sorted)

    if return_best_only:
        return [population_sorted[0]], [entropies_sorted[0]]

    return population_sorted, entropies_sorted


def reproduce(population):
    # for i in population, plug in weights and eval accuracy using same functions
    # my guess is it's just going to optimize toward 0 unless I add another penalty term
    # so observe iti

    # torunament selectiom, divide into maybe 10
    # choose k (the tournament size) individuals from the population at random
    # choose the best individual from the tournament with probability p=.75
    # choose the second best individual with probability p*(1-p)
    # choose the third best individual with probability p*((1-p)^2)
    # and so on
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
