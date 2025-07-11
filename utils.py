import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math as m
from PIL import Image
import copy
import os

# path to output folder
path="/users/jforebac/CIR/cause-tests/multfile"

# maxnorm100 meannorm100 maxETF100 medianETF100 maxshift100 medianshift100

def even_space(height):
    """
    Returns the vertical coordinate to evenly space three classes as triangle vertices.
    """

    # side length of triangle
    tmp = ((height**2)/2)

    # return even spaced height
    return (m.sqrt(3*tmp) - m.sqrt(tmp))


def scalar_calculation(means, method):
    scalars = []
    # find norms of all clusters
    for i in means:
        scalars.append(torch.linalg.vector_norm(i))

    if method == 'max-norm':
        tmp = max(scalars)
        for i in range(len(scalars)):
            scalars[i] = tmp / scalars[i]
    elif method == 'mean-norm':
        tmp = sum(scalars) / len(scalars)
        for i in range(len(scalars)):
            scalars[i] = tmp / scalars[i]
    elif method == 'max-ETF':
        scalars = space_calc(torch.stack(means), scalars, 'max')
    elif method == 'median-ETF':
        scalars = space_calc(torch.stack(means), scalars, 'median')
    elif method == 'max-shift':
        scalars = space_calc(torch.stack(means), scalars, 'max', 'shift')
    elif method == 'median-shift':
        scalars = space_calc(torch.stack(means), scalars, 'median', 'shift')
    else:
        scalars = torch.tensor([1.0, 1.0, 1.0])

    if isinstance(scalars, list):
        scalars = torch.tensor(scalars)

    return scalars

# Return how many samples are in each class
def count_samples(data, key):
    tmp = [0, 0, 0]
    for i in data:
        if (torch.equal(i, key[0])):
            tmp[0] += 1
        elif (torch.equal(i, key[1])):
            tmp[1] += 1
        else:
            tmp[2] += 1

    return tmp


# FIXME later add decay_rate, step to track, and method. Parameter is percent correct
def scale_samples(x, y, scalars, decay_param):
    # decay_param = 1
    scale_dict = {(1.0, 0.0, 0.0): 0,
                  (0.0, 1.0, 0.0): 1,
                  (0.0, 0.0, 1.0): 2}

    # general loop
    for i in range(len(x)):
        x[i] = (x[i]*(scalars[scale_dict[tuple(y[i].tolist())]]) * decay_param) + (x[i] * (1 - decay_param))


def shift_samples(x, y, shift, decay_param):
    # decay_param = 1
    scale_dict = {(1.0, 0.0, 0.0): 0,
                  (0.0, 1.0, 0.0): 1,
                  (0.0, 0.0, 1.0): 2}

    # general loop
    for i in range(len(x)):
        x[i] = (x[i] + (shift[scale_dict[tuple(y[i].tolist())]]) * decay_param) + (x[i] * (1 - decay_param))
    

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


# FIXME how do I find the scalar? USE MEDIAN CLUSTER NORM TO CONSTRUCT ETF PROJECTION! OR MAX

def space_calc(means, scalars, method, shift=False):
    # calculate max norm
    if method == 'max':
        idx = max(scalars)
        idx = scalars.index(idx)
        height = scalars[idx].item()
        # now calculate scalars to scale each to match that point
    else:
        sc = copy.deepcopy(scalars)
        sc.sort()
        idx = sc[(len(sc) // 2)]
        idx = scalars.index(idx)
        height = scalars[idx]

    if idx:
        # This works I think. Now just calculate scalars
        coord = m.sqrt((height**2)/2)
        height = even_space(height)
    else:
        coord = rotation(height, theta=((2*m.pi)/3))
    
    target = [torch.tensor([0.0, height]),
         torch.tensor([-coord, -coord]),
         torch.tensor([coord, -coord])]


    target = torch.stack(target)

    if shift:
        return target

    # now calculate scalars to scale each to match that point
    # Compute dot products for each row
    print(f'target: {target} means: {means}')
    numerator = torch.sum(target * means, dim=1)     # T_i ⋅ A_i
    denominator = torch.sum(means * means, dim=1)   # A_i ⋅ A_i
    print(f'num: {numerator} denom: {denominator}')
    scalars = numerator / denominator


    # Safe division (avoid div-by-zero if needed)
    print(f'scalars(func): {scalars}')

    return scalars
    

def rotation(height, theta):
    rot_mat = torch.tensor([[m.cos(theta), -m.sin(theta)],[m.sin(theta), m.cos(theta)]])

    coord = rot_mat @ torch.tensor([0, height])

    return abs(coord[0]).item()


def is_equilateral(points, tol=1e-9):
    """
    Check if 3 PyTorch 2D vectors form an equilateral triangle.
    
    Args:
        points (list of torch.Tensor): List of 3 tensors, each of shape (2,)
        tol (float): Tolerance for equality check due to floating-point error
    
    Returns:
        bool: True if the points form an equilateral triangle, False otherwise
    """
    dist = []
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            dist.append(torch.sum((points[i] - points[j]) ** 2))
        
    print(f'dist: {dist}')

    return torch.allclose(dist[0], dist[1], atol=tol) and torch.allclose(dist[1], dist[2], atol=tol)