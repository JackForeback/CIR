import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Set variables
torch.manual_seed(42)
num_classes = 3
num_training_steps = 20
input_dim = 2
samples_per_class = 100
train_ratio = 0.7
total_samples = samples_per_class * num_classes

# Set means for 2D Gaussians
means = [torch.tensor([5.0, 0.0]),
         torch.tensor([-5.0, 0.0]),
         torch.tensor([0.0, 15.0])]

# Set covariance matrices. Establishes spread & direction of probability cluster
covs = [torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([[2.0, 0.0], [0.0, 2.0]])]

# Lists for Input and Output
X, Y = [], []

# Generate input data
for class_id in range(num_classes):
    x = torch.distributions.MultivariateNormal(means[class_id], covs[class_id]).sample((samples_per_class,))
    X.append(x)

# Copy for plotting
cp = X

# Create corresponding class labels
classes = [torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 1.0, 0.0]), torch.tensor([0.0, 0.0, 1.0])]
for i in range(num_classes):
    for j in range(samples_per_class):
        Y.append(classes[i])
               
# Combine to single tensors
X = torch.cat(X, dim=0)
Y = torch.stack(Y, dim=0)

# Shuffle the data, maintains correct labels
perm = torch.randperm(X.size(0))
X, Y = X[perm], Y[perm]

# Slice tensors to create train test split
split_idx = int(train_ratio * total_samples)
X_train, Y_train = X[:split_idx], Y[:split_idx]
X_test, Y_test = X[split_idx:], Y[split_idx:]

# Model definition
class LinearClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

model = LinearClassifier(input_dim, num_classes)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Dict to keep track of classification accuracy per class every step
accuracy_dict = {i: [] for i in range(num_classes)}

# Calculate number of samples from each class in the test set
test_samples_per_class = [0, 0, 0]
for i in Y_test:
    if (torch.equal(i, classes[0])):
        test_samples_per_class[0] += 1
    elif (torch.equal(i, classes[1])):
        test_samples_per_class[1] += 1
    else:
        test_samples_per_class[2] += 1

# Function to track accuracy of each class
def track_accuracy(predictions):
    tmp_arr = [0, 0, 0]
    # Check how many correct classifications made for each class
    for i in range(len(predictions)):
        idx = torch.argmax(predictions[i])
        if (idx == torch.argmax(Y_test[i])):
            tmp_arr[idx] += 1

    # Add percent of correctly clasified test data to accuracy_dict
    for i in range(num_classes):
        accuracy_dict[i].append(tmp_arr[i] / test_samples_per_class[i])

# Training loop
for _ in range(num_training_steps):
    y_pred = model(X_train)
    loss = criterion(y_pred, Y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Test loop
    with torch.no_grad():
        y_pred = model(X_test)
        track_accuracy(y_pred)

# print means and covariance to be piped into file with images to identify each test
print("Means:", means, "\n", "Covs:", covs)

# Accuracy plotting
plt.figure(figsize=(10, 6))
for class_id, accuracy_list in accuracy_dict.items():
    plt.plot(accuracy_list, label=f"Class {class_id}", linestyle='-')

plt.xlabel("Training Step")
plt.ylabel("Accuracy")
plt.title("Per-Class Test Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("tests/test1/accuracy_graph.png")
plt.close()

# Sample plotting
colors = ['green', 'blue', 'purple']
labels = [f"Class {i}" for i in range(num_classes)]

plt.figure(figsize=(10, 6))

# Plot original class samples
for class_id in range(num_classes):
    samples = cp[class_id]
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
plt.savefig("tests/test1/sample_plot.png")
plt.close()