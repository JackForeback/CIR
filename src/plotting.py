import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image 
import os, sys

# path to output folder
path=sys.argv[2]

def plot_samples(data, num_classes, samples_per_class):
    """
    Plots 2D Gaussian data before training, colored by class.

    Args:
        data (torch.Tensor): Stacked tensor of 2D points, with shape (num_classes * samples_per_class, 2).
        num_classes (int): Number of Gaussian distributions (classes).
        samples_per_class (int): Number of points per class.
    """
    # Use a dynamic colormap for arbitrary number of classes
    cmap = plt.get_cmap('tab10' if num_classes <= 10 else 'nipy_spectral', num_classes)
    labels = [f"Class {i}" for i in range(num_classes)]

    plt.figure(figsize=(10, 6))

    for class_id in range(num_classes):
        start = class_id * samples_per_class
        end = start + samples_per_class
        samples = data[start:end]
        plt.scatter(samples[:, 0], samples[:, 1],
                    color=cmap(class_id),
                    label=labels[class_id])

    # Plot formatting
    format_plot("Generated 2D Gaussian Samples", save_path=f"{path}/sample_plot.png")


def make_animation(seeds, training_steps):
    """
    Compiles decision boundary plots into a per-seed GIF animation.

    Args:
        seeds (int): Number of training seeds (model runs).
        training_steps (int): Number of steps per training run.
    """

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


def plot_avg_accuracy(train, test, num_classes):
    """
    Plots average per-class training and test accuracy at each training step.

    Args:
        train (dict[int, list[float]]): Per-class train accuracy (key: class ID).
        test (dict[int, list[float]]): Per-class test accuracy.
    """

    plt.figure(figsize=(10, 6))

    cmap = plt.get_cmap('tab10' if num_classes <= 10 else 'nipy_spectral', num_classes)

    # Training accuracy
    for class_id, accuracy_list in train.items():
        plt.plot(
            accuracy_list,
            label=f"Class {class_id} (train)",
            linestyle='--',
            color=cmap(class_id)
        )

    # Test accuracy
    for class_id, accuracy_list in test.items():
        plt.plot(
            accuracy_list,
            label=f"Class {class_id} (test)",
            linestyle='-',
            color=cmap(class_id)
        )

    format_plot("Per-Class Average Accuracy", "Training Step", "Accuracy", f"{path}/avg_accuracy_graph.png")


def seed_plot(per_seed, num_seeds, num_training_steps):
    """
    Plots max-min class accuracy for each training step per seed,
    also plots the average across all seeds.

    Args:
        per_seed (dict): Dictionary containing 'train' and 'test' keys, each mapping to
                         a list of lists: [num_seeds][num_steps][num_classes].
        num_seeds (int): Number of seeds used in training runs.
        num_training_steps (int): Number of training steps.
    """
    train_average = [0.0] * num_training_steps
    test_average = [0.0] * num_training_steps


    for seed in range(num_seeds):
        train = per_seed['train'][seed]  # list of length num_training_steps, each with 3 class accuracies
        test = per_seed['test'][seed]

        # Max - Min accuracy per step
        train_range = [max(t) - min(t) for t in train]
        test_range = [max(t) - min(t) for t in test]

        # adding for average Max-Min
        for i in range(num_training_steps):
            train_average[i] += train_range[i]
            test_average[i] += test_range[i]

        plt.plot(
            train_range,
            label="Train (max - min)",
            linestyle='-',
            color='black'
        )
        plt.plot(
            test_range,
            label="Test (max - min)",
            linestyle='-',
            color='gray'
        )

        format_plot(f"Max-Min Class Accuracy: Seed {seed})", "Training Step", "Accuracy", 
                    f"{path}/seed/per_class_accuracy_seed{seed}.png")

    # avgerages Max-Min values
    for i in range(num_training_steps):
            train_average[i] /= num_seeds
            test_average[i] /= num_seeds

    plt.plot(
            train_average,
            label="Train (max - min)",
            linestyle='-',
            color='black'
        )
    plt.plot(
        test_average,
        label="Test (max - min)",
        linestyle='-',
        color='gray'
    )

    format_plot(f"Average Max-Min at Each Step", "Training Step", "Average Accuracy", f"{path}/seed/avg_diff.png")


def plot_decision_boundaries(data, Y, classes, num_classes, weights, biases, step, seed):
    """
    Plots class samples and decision boundaries without gradient tables.

    Args:
        data (Tensor): Input samples of shape (N, 2).
        Y (list[Tensor]): One-hot labels.
        classes (list[Tensor]): One-hot class vectors.
        num_classes (int): Number of classes.
        weights (Tensor): Class weights (num_classes, 2).
        biases (Tensor): Class biases (num_classes,).
        step (int): Training step.
        seed (int): Seed for reproducibility.
    """
    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap('tab10' if num_classes <= 10 else 'nipy_spectral', num_classes)
    x_vals = torch.linspace(data[:, 0].min() - 1, data[:, 0].max() + 1, 500)

    # Plot samples
    for class_id in range(num_classes):
        indices = [i for i, label in enumerate(Y) if torch.equal(label, classes[class_id])]
        if indices:
            samples = data[indices]
            plt.scatter(samples[:, 0], samples[:, 1],
                        color=cmap(class_id),
                        label=f"Class {class_id}")

    # Plot decision boundaries
    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            w_diff = weights[i] - weights[j]
            b_diff = biases[i] - biases[j]
            a, b = w_diff[0].item(), w_diff[1].item()
            c = b_diff.item()

            if b != 0:
                y_vals = -(a / b) * x_vals - (c / b)
                plt.plot(x_vals, y_vals, label=f"Boundary {i} vs {j}")
            elif a != 0:
                x_intercept = -c / a
                plt.axvline(x=x_intercept, label=f"Boundary {i} vs {j}")
            else:
                continue  # Degenerate case, do nothing

    # Plot formatting
    x_min, x_max = data[:, 0].min().item(), data[:, 0].max().item()
    y_min, y_max = data[:, 1].min().item(), data[:, 1].max().item()
    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1

    plt.axis([x_min - x_margin, x_max + x_margin,
              y_min - y_margin, y_max + y_margin])
    format_plot(f"Decision Boundary Visualization", save_path=f"{path}/db/{seed}-{step}.png")


def format_plot(title="", xlabel="X", ylabel="Y", save_path=f'{path}'):
    """
    Applies consistent formatting to matplotlib plots and saves them.

    Args:
        title (str): Plot title.
        xlabel (str): Label for X-axis.
        ylabel (str): Label for Y-axis.
        save_path (str): Path to save the plot image. If None, plot is saved to default path
    """
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()