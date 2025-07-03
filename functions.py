import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math as m
from PIL import Image
import copy
import os

# run all 4 methods based on lr results

path="/users/jforebac/CIR/cause-tests/scaling/2far/meannorm100"

# maxnorm100 meannorm100 maxETF100 medianETF100 maxshift100 medianshift100
# run base 10 and base 20 to scaling folder. Then run all far tests with everything and see what's working and fix what isn't
# Also before you run (not base tests) check why scaling is not working/producing weird plots


# Returns height of third cluster so classes are equidistant (if necessary)
def even_space(height):

    # side length of triangle
    tmp = ((height**2)/2)

    # return even spaced height
    return (m.sqrt(3*tmp) - m.sqrt(tmp))


# return scalars to multiply by to make sure all classes are equidistant
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


# Plots Input Data to the model
def plot_samples(data, num_classes, samples_per_class):
    # Sample plotting
    colors = ['green', 'blue', 'purple']
    labels = [f"Class {i}" for i in range(num_classes)]

    plt.figure(figsize=(10, 6))

    # Plot original class samples
    for class_id in range(num_classes):
        start = class_id * samples_per_class
        end = start + samples_per_class
        samples = data[start:end]
        plt.scatter(samples[:, 0], samples[:, 1],
                    color=colors[class_id],
                    label=labels[class_id])

    # Plot formatting
    plt.title("Generated 2D Gaussian Samples")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{path}/sample_plot.png")
    plt.close()


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


# FIXME later add decay_rate, step to track, and method
def scale_samples(x, y, scalars, decay_param):
    # decay_param = 1
    scale_dict = {(1.0, 0.0, 0.0): 0,
                  (0.0, 1.0, 0.0): 1,
                  (0.0, 0.0, 1.0): 2}

    # general loop
    for i in range(len(x)):
        x[i] = (x[i]*(scalars[scale_dict[tuple(y[i].tolist())]]) * decay_param) + (x[i] * (1 - decay_param))

    # # start simple now. First 5 full norm, then half, then full base normal after for next 5?
    # if method == 'linear':
    #     decays = [1]
    #     tmp = 1 / num_steps
    #     for i in num_steps:
    #         decays.append(decays[i] - tmp)
        
    # elif method == 'exponential':
    #     decay_param = 1 * m.exp(-decay_rate*step)

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
    # Check how many correct classifications for each class
    for i in range(len(predictions)):
        idx = torch.argmax(predictions[i])
        if (idx == torch.argmax(data[i])):
            tmp_arr[idx] += 1

    # Add number of correctly clasified test data to correct step spot in accuracy_dict
    for i in range(n_classes):
        dict[i][step] += tmp_arr[i]
        per_seed[key][seed][step].append(tmp_arr[i] / samples[i])


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


# makes images of decision boundary into animation
def make_animation(seeds, training_steps):

    for i in range(seeds):
        # Path to images
        image_dir = f"{path}/db/" 
        image_filenames = [f"{i}-{j}.png" for j in range(training_steps)]  # 0.png to 50.png
        image_paths = [os.path.join(image_dir, fname) for fname in image_filenames]

        # Load all images
        frames = [Image.open(img_path) for img_path in image_paths]

        # Save as animated GIF
        frames[0].save(
            f"{path}/ani/BA-seed:{i}.gif",
            save_all=True,
            append_images=frames[1:],  # all other frames
            duration=200,              # time between frames in ms
            loop=0                     # 0 = loop forever
        )


# Plot train and test classification accuracy
def plot_accuracy(train, test):

    plt.figure(figsize=(10, 6))

    colors = ['green', 'blue', 'purple']

    # Training accuracy
    for class_id, accuracy_list in train.items():
        plt.plot(
            accuracy_list,
            label=f"Class {class_id} (train)",
            linestyle='--',
            color=colors[class_id]
        )

    # Test accuracy
    for class_id, accuracy_list in test.items():
        plt.plot(
            accuracy_list,
            label=f"Class {class_id} (test)",
            linestyle='-',
            color=colors[class_id]
        )

    plt.xlabel("Training Step")
    plt.ylabel("Accuracy")
    plt.title("Per-Class Test Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{path}/avg_accuracy_graph.png")
    plt.close()


# plots accuracy by group for each seed
def seed_plot(per_seed, num_seeds, num_training_steps):
    colors = ['green', 'blue', 'purple']  # for 3 classes

    for seed in range(num_seeds):
        train = per_seed['train'][seed]  # list of length num_training_steps, each with 3 class accuracies
        test = per_seed['test'][seed]

        # Create dicts for class-wise accuracy over time
        train_by_class = {i: [] for i in range(3)}
        test_by_class = {i: [] for i in range(3)}

        for step in range(num_training_steps):
            for class_id in range(3):
                train_by_class[class_id].append(train[step][class_id])
                test_by_class[class_id].append(test[step][class_id])

        # Plotting
        plt.figure(figsize=(10, 6))

        # Per-class accuracy lines
        for class_id in range(3):
            plt.plot(
                train_by_class[class_id],
                label=f"Class {class_id} (train)",
                linestyle='--',
                color=colors[class_id]
            )
            plt.plot(
                test_by_class[class_id],
                label=f"Class {class_id} (test)",
                linestyle='-',
                color=colors[class_id]
            )

        # Max - Min accuracy range line per step
        train_range = [max(t) - min(t) for t in train]
        test_range = [max(t) - min(t) for t in test]

        plt.plot(
            train_range,
            label="Train (max - min)",
            linestyle=':',
            color='black'
        )
        plt.plot(
            test_range,
            label="Test (max - min)",
            linestyle='-.',
            color='gray'
        )

        plt.xlabel("Training Step")
        plt.ylabel("Accuracy")
        plt.title(f"Per-Class Accuracy + Range (Seed {seed})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{path}/seed/per_class_accuracy_seed{seed}.png")
        plt.close()



# used to plot samples before training to ensure using desired distribution
def tst_plot_samples(data, num_classes):
    # Sample plotting
    colors = ['green', 'blue', 'purple']
    labels = [f"Class {i}" for i in range(num_classes)]

    plt.figure(figsize=(10, 6))

    # Plot original class samples
    for class_id in range(num_classes):
        samples = data[class_id]
        plt.scatter(samples[:, 0], samples[:, 1],
                    # alpha=0.5,
                    color=colors[class_id],
                    label=labels[class_id])

    # Plot formatting
    plt.title("Generated 2D Gaussian Samples with Test Points")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{path}/plots/sample_plot.png")
    plt.close()

# FIXME how do I find the scalar? USE MEDIAN CLUSTER NORM TO CONSTRUCT ETF PROJECTION! OR MAX

def space_calc(means, scalars, method, *args):
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

    if args:
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



# # FIXME scale x and y axis of ecah cluster depending on distance.
# # finds how much and what to apply to each gradient
# def find_distance(means, method):
#     diffs = {'dx': [], 'dy': []}
#     totals = {i: 0.0 for i in range(len(means))}
#     x_avg, y_avg = [], []

#     # Find x and y distances between clusters
#     for i in range(len(means)):
#         for j in range(i + 1, len(means)):
#             diff_vec = means[j] - means[i]
#             dx, dy = diff_vec[0].item(), diff_vec[1].item()
#             dist = torch.linalg.vector_norm(diff_vec).item()

#             diffs['dx'].append(dx)
#             diffs['dy'].append(dy)

#     if method == 'mean':
#         # Calculate average x and y distances
#         for i in range(len(means)):
#             for j in range(i + 1, len(means)):
#                 x_avg.append((abs(diffs['dx'][i]) + abs(diffs['dx'][j]))/2)
#                 y_avg.append((abs(diffs['dy'][i]) + abs(diffs['dy'][j]))/2)

#         # scale each according to max
#         x = max(x_avg)
#         y = max(y_avg)
#         for i in range(len(means)):
#             x_avg[i] = x / x_avg[i]
#             y_avg[i] = y / y_avg[i]
#     else:
#         x
    
#     return [y_avg, x_avg]

# # Doing the dew
# def scale_calculation(vectors):
#     # random bullshit lists for first draft
#     lengths, angles, s1, s2 = [], [], [], []

#     # Find length of vectors for area calculation
#     for i in vectors:
#         lengths.append(torch.linalg.vector_norm(i).item())

#     max_idx = lengths.index(max(lengths))

#     # Calculate angles between each vector
#     for i in range(len(vectors)):
#         for j in range(i + 1, len(vectors)):
#             u, v = vectors[i], vectors[j]
#             dot = torch.dot(u, v)
#             norm_product = torch.norm(u) * torch.norm(v)
#             cos_theta = dot / norm_product
#             angle = torch.acos(cos_theta) # in radians
#             angle = angle / (2*m.pi) # portion of total circle convergence area covers
#             angles.append(angle.item())

#     # manual calculation to average angles. FIXME This is bad, but whatever
#     # sum clusters 0, 1, 2 divide by average total angle
#     s2.append(angles[0]*2*m.pi*lengths[0] + angles[1]*2*m.pi*lengths[0])
#     s2.append(angles[0]*2*m.pi*lengths[0] + angles[2]*2*m.pi*lengths[2])
#     s2.append(angles[1]*2*m.pi*lengths[0] + angles[2]*2*m.pi*lengths[2])

#     # calculate portion of circle corresponding to each angle
#     # for i in range(len(lengths)):
#     #     s2[i] *= (2*m.pi*lengths[i])

#     # How much extra area does the max cluster have compared to the other clusters?
#     # From that have a scale factor to apply to other clusters
#     maximum = max(s2)
#     for i in range(len(s2)):
#         s2[i] = maximum / s2[i]

#     s1 = find_distance(vectors)

#     for i in range(len(s1)):
#         for j in range(len(s1[i])):
#             if j == max_idx:
#                 s1[i][j] = 1.0
#             elif s1[i][j] > 1.0000001:
#                 s1[i][j] += s2[j]

#     return torch.transpose(torch.tensor(s1), 0, 1)
