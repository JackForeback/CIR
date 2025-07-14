import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math as m
from PIL import Image
import copy
import os

# path to output folder
path="/users/jforebac/CIR/cause-tests/0err"

# FIXME Add norm method instead of ETF
# maxnorm meannorm maxETF medianETF maxshift medianshift

def even_space(height):
    """
    Returns the vertical coordinate to evenly space three classes as triangle vertices.
    """
    # side length of triangle
    tmp = ((height**2)/2)

    # return even spaced height
    return (m.sqrt(3*tmp) - m.sqrt(tmp))


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


# Plot weights as lines, and weights plus bias to monitor parameters
def monitor_parameters(data, Y, classes, num_classes, weights, biases, step, seed, samples_per_class, w_grad, b_grad):
    # Sample plotting
    colors = ['green', 'blue', 'purple']
    labels = [f"Class {i}" for i in range(num_classes)]

    plt.figure(figsize=(10, 6))

    # Plot original class samples
    for class_id in range(num_classes):
        # Get all indices for this class
        class_indices = [i for i, label in enumerate(Y) if torch.equal(label, classes[class_id])]
        if len(class_indices) == 0:
            continue
        samples = data[class_indices]
        plt.scatter(
            samples[:, 0], samples[:, 1],
            color=colors[class_id],
            label=labels[class_id],
        )

    # Plotting
    x_vals = torch.linspace(data[:, 0].min() - 1, data[:, 0].max() + 1, 500)

    x_min, x_max = data[:, 0].min().item(), data[:, 0].max().item()
    y_min, y_max = data[:, 1].min().item(), data[:, 1].max().item()

    # Add a little margin
    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1

    tmp, new = [], []
    change = weights - 0.01 * w_grad

    # Compute and plot decision boundaries between each pair of classes
    for i in range(num_classes):
        for j in range(i + 1, num_classes):

            w_diff = weights[i] - weights[j]
            n_diff = change[i] - change[j]

            b_diff = biases[i] - biases[j]
            
            a, b = w_diff[0].item(), w_diff[1].item()
            if (n_diff[1].item() != 0):
                new.append(-(n_diff[0].item() / n_diff[1].item()))
            else:
                new.append(m.inf)
            c = b_diff.item()
            
            # plot the line
            if b != 0:
                y_vals = -(a / b) * x_vals - (c / b)
                tmp.append(-(a/b))
                plt.plot(x_vals, y_vals, label=f"Boundary {i} vs {j}")

            elif a == 0:
                x_intercept = 0
                tmp.append(m.inf)
                plt.axvline(x_intercept, label=f"Boundary {i} vs {j}")
            
            else:
                # Vertical line
                x_intercept = -c / a
                tmp.append(m.inf)
                plt.axvline(x_intercept, label=f"Boundary {i} vs {j}")

    # FIXME want to see how it affects lines. new abc for each of the 3 different lines
    data = [[w_grad[0, 0].item(), w_grad[0, 1].item(), b_grad[0].item()],
            [w_grad[1, 0].item(), w_grad[1, 1].item(), b_grad[1].item()],
            [w_grad[2, 0].item(), w_grad[2, 1].item(), b_grad[2].item()]]
    rows = ['R1G', 'R2G', 'R3G']
    columns = ['X', 'Y', 'Bias']

    plt.table(cellText=data,colLabels=columns,rowLabels=rows,loc='bottom',cellLoc='left')

    avg = sum(new)/3
    
    # FIXME Add comments and explain everything
    data = [[tmp[0], new[0], (tmp[0]-new[0])/avg],
            [tmp[1], new[1], (tmp[1]-new[1])/avg],
            [tmp[2], new[2], (tmp[2]-new[2])/avg]]
    columns = ['slope', 'new', 'dev']
    rows = ['DB1', 'DB2', 'DB3']

    plt.table(cellText=data,colLabels=columns,rowLabels=rows,loc='top',cellLoc='left')

    # Plot formatting
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis([x_min-x_margin, x_max+x_margin, y_min-y_margin, y_max+y_margin])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{path}/db/{seed}-{step}.png")
    plt.close()


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
    return torch.stack([x, y], dim=1)


def transform_to_even_space(means, mode='shift', ref_mode='mean'):
    """
    Transforms a set of 2D class means to be evenly spaced around the origin.
    
    Args:
        means (Tensor): Tensor of shape (N, 2), N is number of classes.
        mode (str): 'shift' or 'scale'. Whether to compute shift vectors or scaling factors.
        ref_mode (str): 'mean', 'max', or 'median' norm to use for radius in 'scale' mode.

    Returns:
        Tensor: 
          - If mode='shift': target positions (N, 2)
          - If mode='scale': scaling factors (N,)
          - If mode='norm': scaling factors (N,)
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

    if mode == 'shift':
        # Return target positions to shift means toward
        print(f'targets: {targets}')
        return targets - means

    elif mode == 'scale':
        # Compute scalar projection of target onto mean direction
        dot_products = torch.sum(targets * means, dim=1)        # T_i ⋅ A_i
        mean_norms_sq = torch.sum(means * means, dim=1) + 1e-9  # A_i ⋅ A_i (safe div)
        scalars = dot_products / mean_norms_sq
        return scalars

    elif mode == 'norm':
        # Compute scalar projection based off norms
        scalars = torch.full_like(norms, fill_value=radius) / norms
        return scalars

    else:
        raise ValueError("mode must be 'shift', 'scale', or 'norm'")



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
