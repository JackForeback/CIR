import matplotlib.pyplot as plt

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
    plt.savefig("tests/test2/sample_plot.png")
    plt.close()