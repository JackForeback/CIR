import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
import os

jobname="even50"
path="/users/jforebac/CIR"
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


# Function to track the total number of correct classifications at each step
def track_accuracy(predictions, step, dict, data, n_classes):
    tmp_arr = [0] * n_classes
    # Check how many correct classifications for each class
    for i in range(len(predictions)):
        idx = torch.argmax(predictions[i])
        if (idx == torch.argmax(data[i])):
            tmp_arr[idx] += 1

    # Add number of correctly clasified test data to correct step spot in accuracy_dict
    for i in range(n_classes):
        dict[i][step] += tmp_arr[i]


# Plots Input Data to the model
def plot_samples(data, num_classes):
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
    plt.savefig(f"{path}/cause-tests/{jobname}/sample_plot.png")
    plt.close()


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
    plt.savefig(f"{path}/cause-tests/{jobname}/accuracy_graph.png")
    plt.close()
    

# Plot weights as lines, and weights plus bias to monitor parameters
def monitor_parameters(data, inputs, num_classes, weights, biases, step, seed):
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

    # Plotting
    x_vals = torch.linspace(inputs[:, 0].min() - 1, inputs[:, 0].max() + 1, 500)

    # Compute and plot decision boundaries between each pair of classes
    for i in range(num_classes):
        for j in range(i + 1, 3):

            w_diff = weights[i] - weights[j]
            b_diff = biases[i] - biases[j]
            
            a, b = w_diff[0].item(), w_diff[1].item()
            c = b_diff.item()
            
            if b != 0:
                y_vals = -(a / b) * x_vals - (c / b)
                plt.plot(x_vals, y_vals, label=f"Boundary {i} vs {j}")

            else:
                # Vertical line
                x_intercept = -c / a
                plt.axvline(x_intercept, label=f"Boundary {i} vs {j}")


    # Plot formatting
    plt.title("Decision Boundary Visualization")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis([-45, 45, -45, 35])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{path}/cause-tests/{jobname}/db/{seed}-{step}.png")
    plt.close()


# makes images of decision boundary into animation
def make_animation(seeds, training_steps):

    for i in range(seeds):
        # Path to images
        image_dir = f"{path}/cause-tests/{jobname}/db/" 
        image_filenames = [f"{i}-{j}.png" for j in range(training_steps)]  # 0.png to 50.png
        image_paths = [os.path.join(image_dir, fname) for fname in image_filenames]

        # Load all images
        frames = [Image.open(img_path) for img_path in image_paths]

        # Save as animated GIF
        frames[0].save(
            f"{path}/cause-tests/{jobname}/BA-seed:{i}.gif",
            save_all=True,
            append_images=frames[1:],  # all other frames
            duration=200,              # time between frames in ms
            loop=0                     # 0 = loop forever
        )


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