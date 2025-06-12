import torch
import torch.nn as nn
import torch.optim as optim

from functions import plot_samples, count_samples, track_accuracy, plot_accuracy
from models import LinearClassifier

import math as m

# Set variables & seed for reproducible data generation
num_classes = 3
num_training_steps = 200
num_seeds = 100
input_dim = 2
samples_per_class = 10000
train_ratio = 0.7
total_samples = samples_per_class * num_classes
torch.manual_seed(42)

# Sets center of clusers, equidistant from the origin
height = 5
coord = m.sqrt((height**2)/2)

# Set means for 2D Gaussians
means = [torch.tensor([0.0, height]),
         torch.tensor([-coord, -coord]),
         torch.tensor([coord, -coord])]

# Set covariance matrices. Establishes spread & direction of probability cluster
covs = [torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([[1.0, 0.0], [0.0, 1.0]])]

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

# Calculate number of samples from each class in the test & train set
train_samples = count_samples(Y_train, classes)
test_samples = count_samples(Y_test, classes)

# Dict to store average classification accuracy at each step
train_dict = {i: [0] * num_training_steps for i in range(num_classes)}
test_dict = {i: [0] * num_training_steps for i in range(num_classes)}

# Model Instantiation. Set seeds for 100 trials with random weights
for i in range(num_seeds):
    torch.manual_seed(i)
    model = LinearClassifier(input_dim, num_classes)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Training loop
    for j in range(num_training_steps):
        y_pred = model(X_train)
        track_accuracy(y_pred, j, train_dict, Y_train, num_classes)
        loss = criterion(y_pred, Y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Test loop
        with torch.no_grad():
            y_pred = model(X_test)
            track_accuracy(y_pred, j, test_dict, Y_test, num_classes)

# Divides the number of correct classifications at each step location in the list
# by the total number of possible correct classifications over the total number of training steps
for i in range(num_classes):
    train_div = train_samples[i]*num_seeds
    test_div = test_samples[i]*num_seeds
    for j in range(num_training_steps):
        train_dict[i][j] /= train_div
        test_dict[i][j] /= test_div

plot_accuracy(train_dict, test_dict)

# Print means and covariance to be piped into file with images to verify distributions used in each test
print("Means:", means, "\n", "Covs:", covs)
