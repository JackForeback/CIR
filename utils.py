import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math as m
from PIL import Image
import copy
import os

# maxshift maxscale meanshift meanscale medianshift medianscale

# path to output folder
path="/users/jforebac/CIR/cause-tests/1far"


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
        if (torch.equal(i, key[0])):
            tmp[0] += 1
        elif (torch.equal(i, key[1])):
            tmp[1] += 1
        else:
            tmp[2] += 1

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
    start_angle = m.pi / 2
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
