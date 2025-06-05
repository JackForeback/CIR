import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sample_plotting import plot_samples

# Set variables
num_classes = 3
num_training_steps = 200
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

# Plot the data
plot_samples(X, num_classes)

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

# Calculate number of samples from each class in the test set
test_samples_per_class = [0, 0, 0]
for i in Y_test:
    if (torch.equal(i, classes[0])):
        test_samples_per_class[0] += 1
    elif (torch.equal(i, classes[1])):
        test_samples_per_class[1] += 1
    else:
        test_samples_per_class[2] += 1

# Dict to store average classification accuracy at each step
accuracy_dict = {i: [0] * num_training_steps for i in range(num_classes)}

# Function to track the totalcnumber of accurate classifications at each step
def track_accuracy(predictions, current_step):
    tmp_arr = [0, 0, 0]
    # Check how many correct classifications for each class
    for i in range(len(predictions)):
        idx = torch.argmax(predictions[i])
        if (idx == torch.argmax(Y_test[i])):
            tmp_arr[idx] += 1

    # Add number of correctly clasified test data to correct step spot in accuracy_dict
    for i in range(num_classes):
        accuracy_dict[i][j] += tmp_arr[i]

# Model definition
class LinearClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# Model Instantiation. Set seeds for 100 trials with random weights
for i in range(100):
    torch.manual_seed(i)
    model = LinearClassifier(input_dim, num_classes)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Training loop
    for j in range(num_training_steps):
        y_pred = model(X_train)
        loss = criterion(y_pred, Y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Test loop
        with torch.no_grad():
            y_pred = model(X_test)
            track_accuracy(y_pred, j)

# Divides the number of correct classifications at each step location in the list
# by the total number of possible correct classifications over the # of training steps
for i in range(3):
    div = test_samples_per_class[i]*num_training_steps
    for j in range(len(accuracy_dict[i])):
        accuracy_dict[i][j] /= div

# Print means and covariance to be piped into file with images to identify the distributions used in each test
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
plt.savefig("tests/test3/accuracy_graph.png")
plt.close()