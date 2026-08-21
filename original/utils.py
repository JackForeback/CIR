"""Shared helpers for the original scripts: argument parsing, data generation,
fairness losses, evolutionary weight search, and accuracy bookkeeping.

This module belongs to the *original*, pre-refactor codebase. Its style is
deliberately preserved -- plain functions over module-level state, driven by
``--key value`` pairs on the command line. See ``README.md`` for why, and
``../refactor/`` for the modern equivalent.
"""

import torch
import torch.nn.functional as F
import math as m
from models import LinearClassifier
import copy, os, sys
from typing import Iterable, List, Tuple


def parse_sysargs():
    """
    Parse arguments of the form --KEY VALUE from sys.argv,
    lowercase the keys, and return them as a dict.
    Example: --NUM_CLASSES 3 -> {'num_classes': 3}

    Values that look like integers or floats are converted; everything else is
    left as a string.
    """
    args = sys.argv[1:]  # skip script name
    parsed = {}

    for i in range(0, len(args) - 1, 2):  # step through pairs
        key = args[i].lstrip("-").lower()  # strip '--'
        val = args[i+1]

        # Try to convert to int or float automatically
        try:
            val = int(val)
        except ValueError:
            try:
                val = float(val)
            except ValueError:
                pass  # leave as string

        parsed[key] = val

    return parsed


def get_output_path(default="output"):
    """
    Return the --path argument, or `default` if it was not supplied.

    The pre-refactor code read `sys.argv[2]` at import time, which made this
    module impossible to import unless the caller happened to pass --path first.
    """
    return parse_sysargs().get("path", default)


# make sure loss between class is the same
# (not what we want? Might need larger gradients to even out initialization differences)
def loss_with_per_class_gap(pred, target, lambda_fair=0.1):
    """
    pred: (N, C) raw logits
    target: (N, C) one-hot labels
    """
    num_classes = target.size(1)

    # Standard MSE
    mse_loss = F.mse_loss(pred, target)

    # Per-class average MSE
    per_class_losses = []
    labels = target.argmax(dim=1)
    for k in range(num_classes):
        mask = (labels == k)  # select samples of class k
        if mask.any():
            loss_k = F.mse_loss(pred[mask], target[mask])
            per_class_losses.append(loss_k)
        else:
            # No samples of this class in the batch
            per_class_losses.append(torch.tensor(0.0, device=pred.device))

    per_class_losses = torch.stack(per_class_losses)

    # Fairness penalty = gap between worst and best class
    fairness_loss = per_class_losses.max() - per_class_losses.min()

    # Combine
    total_loss = mse_loss + (lambda_fair * fairness_loss)
    return total_loss, mse_loss, fairness_loss


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
    total_loss = mse_loss + (lambda_fair * fairness_loss)
    return total_loss, mse_loss, fairness_loss


def evo_weights(num_iter, pop_size, tournament_size, weights, means):
    """
    Search for an initial weight matrix that already separates the class means.

    Tournament selection over random weight matrices, scored by the mean softmax
    entropy of `W @ mean` (lower is better -- see `eval_pop`). The population is
    split into `pop_size // tournament_size` groups, the best member of each
    group survives, and the survivors are cloned and mutated to refill the
    population for the next generation.

    Args:
        num_iter (int): Number of generations.
        pop_size (int): Candidates per generation.
        tournament_size (int): Candidates per tournament group. Must divide
            pop_size evenly.
        weights (Tensor): Reference weight matrix; supplies the shape.
        means (Tensor): Class means, shape (num_classes, input_dim).

    Returns:
        Tensor: The best weight matrix found, same shape as `weights`.
    """
    if pop_size <= 0 or tournament_size <= 0 or pop_size % tournament_size:
        raise ValueError("pop_size must be a positive multiple of tournament_size")

    num_groups = pop_size // tournament_size
    num_classes, input_dim = weights.shape

    population = init_population(num_classes, input_dim, pop_size)

    for _ in range(num_iter):
        # One winner per tournament group.
        winners = []
        for g in range(num_groups):
            group = population[g * tournament_size:(g + 1) * tournament_size]
            group_sorted, _ = eval_pop(means, group)
            winners.append(group_sorted[0])

        # Refill the population from mutated copies of the winners.
        population = list(winners)
        while len(population) < pop_size:
            population.append(mutate(winners[len(population) % num_groups].clone()))

    population_sorted, _ = eval_pop(means, population)
    return population_sorted[0].clone()


def init_population(num_classes, input_dim, pop_size):
    """
    Build the first generation of candidate weight matrices.

    Each candidate is drawn from the same uniform range nn.Linear uses, so the
    search starts where ordinary initialization would.

    Args:
        num_classes (int): Rows of each weight matrix.
        input_dim (int): Columns of each weight matrix.
        pop_size (int): Number of candidates.

    Returns:
        list[Tensor]: The candidate weight matrices.
    """
    bound = 1.0 / (input_dim ** 0.5)
    return [
        (torch.rand(num_classes, input_dim) * 2 - 1) * bound
        for _ in range(pop_size)
    ]


def mutate(weights, rate=1/6, scale=1.0):
    """
    Inject Gaussian noise into a random subset of entries.

    Args:
        weights (Tensor): Weight matrix to mutate, modified in place.
        rate (float): Probability that any individual entry is perturbed.
        scale (float): Standard deviation of the perturbation.

    Returns:
        Tensor: The mutated weight matrix.
    """
    mask = torch.rand(weights.shape) < rate
    weights += mask * (torch.randn(weights.shape) * scale)
    return weights


def eval_pop(means, population):
    """
    Score and sort a population of candidate weight matrices.

    Each candidate is scored by the mean Shannon entropy of softmax(W @ mean)
    across all class means. Low entropy means the candidate already commits
    confidently to one class per cluster, so no class starts at a disadvantage.

    Args:
        means (Iterable[Tensor]): Class mean vectors, each shape (input_dim,).
        population (Iterable[Tensor]): Candidate weight matrices, each shape
            (num_classes, input_dim).

    Returns:
        tuple[list[Tensor], list[float]]: The population and its entropy scores,
        both sorted low entropy first.
    """
    eps = 1e-12
    entropies = []

    for candidate in population:
        # candidate @ mean -> one logit per class; average entropy over means
        per_mean = []
        for mean_vec in means:
            probs = F.softmax(candidate @ mean_vec, dim=0)
            per_mean.append(float(-(probs * (probs + eps).log()).sum()))
        entropies.append(sum(per_mean) / len(per_mean))

    # pair and sort (low entropy first)
    order = sorted(range(len(entropies)), key=lambda i: entropies[i])
    return [population[i] for i in order], [entropies[i] for i in order]


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

    # Rotate the cycle so the topmost point comes first. The points must stay in
    # adjacency order because is_regular_polygon() measures the distance between
    # consecutive entries; sorting them by position interleaves opposite sides of
    # the polygon and misreports a genuine ETF as irregular for 5+ classes.
    keys = [(-means[i, 1].item(), means[i, 0].item()) for i in range(num_points)]
    start = min(range(num_points), key=lambda i: keys[i])

    return torch.roll(means, shifts=-start, dims=0)


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



def is_regular_polygon(points, tol=1e-4):
    """
    Check if N 2D points form a regular polygon (equal side lengths).
    
    Args:
        points (Tensor): Tensor of shape (N, 2)
        tol (float): Absolute tolerance on the squared side lengths. Compared in
            float32, so a tolerance much below 1e-4 is finer than the arithmetic
        
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
