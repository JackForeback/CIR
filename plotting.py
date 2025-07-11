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

def plot_samples(data, num_classes, samples_per_class):
    """
    Plots generated data before training
    
    Args:
        data (torch.Tensor): Single tensor of 2D points sampled from our Gaussian Distriutions
        num_classes (int): Number of Gaussians we generate samples for
        samples_per_class (int): Number of samples generated from each class
    """
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

# FIXME separate and fix this
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
    train_average = [0.0] * num_training_steps
    test_average = [0.0] * num_training_steps


    for seed in range(num_seeds):
        train = per_seed['train'][seed]  # list of length num_training_steps, each with 3 class accuracies
        test = per_seed['test'][seed]

        # # Create dicts for class-wise accuracy over time
        # train_by_class = {i: [] for i in range(3)}
        # test_by_class = {i: [] for i in range(3)}

        # for step in range(num_training_steps):
        #     for class_id in range(3):
        #         train_by_class[class_id].append(train[step][class_id])
        #         test_by_class[class_id].append(test[step][class_id])

        # # Plotting
        # plt.figure(figsize=(10, 6))

        # # Per-class accuracy lines
        # for class_id in range(3):
        #     plt.plot(
        #         train_by_class[class_id],
        #         label=f"Class {class_id} (train)",
        #         linestyle='--',
        #         color=colors[class_id]
        #     )
        #     plt.plot(
        #         test_by_class[class_id],
        #         label=f"Class {class_id} (test)",
        #         linestyle='-',
        #         color=colors[class_id]
        #     )

        # Max - Min accuracy range line per step
        train_range = [max(t) - min(t) for t in train]
        test_range = [max(t) - min(t) for t in test]
        for i in range(num_training_steps):
            train_average[i] += train_range[i]
            test_average[i] += test_range[i]

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

    for i in range(num_training_steps):
            train_average[i] /= num_seeds
            test_average[i] /= num_seeds

    plt.plot(
            train_average,
            label="Train (max - min)",
            linestyle=':',
            color='black'
        )
    plt.plot(
        test_average,
        label="Test (max - min)",
        linestyle='-.',
        color='gray'
    )

    plt.xlabel("Training Step")
    plt.ylabel("Average Accuracy")
    plt.title(f"Max-Min Range")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{path}/seed/avg_diff.png")
    plt.close()

