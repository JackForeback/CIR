import torch
import torch.nn as nn
import torch.optim as optim
import math as m
import copy

from models import LinearClassifier
from functions import (
    even_space, scalar_calculation, plot_samples, count_samples, scale_samples,
    track_accuracy, monitor_parameters, make_animation, plot_accuracy, seed_plot
    )

# Set variables & seed for reproducible data generation
num_classes = 3
num_training_steps = 50
num_seeds = 10
input_dim = 2
samples_per_class = 10000
train_ratio = 0.7
total_samples = samples_per_class * num_classes
torch.manual_seed(42)

# Sets coordinates used for cluster centers
height = 10
coord = m.sqrt((height**2)/2)
height = even_space(height)

# Set means for 2D Gaussians
means = [torch.tensor([0.0, 20]),
         torch.tensor([-coord, -coord]),
         torch.tensor([coord, -coord])]

# Set covariance matrices. Establishes spread & direction of probability cluster
covs = [torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([[1.0, 0.0], [0.0, 1.0]])]

# Projects clusters to create equidistant class means for equal convergence
# Can use max class mean and scale up, or scale all classes to the mean class centers
scalars = scalar_calculation(means, method='max')

# Print means, scalars, and covariance to output file to verify distributions used in each test
print(f'Means: {means} Covs: {covs} scalars: {scalars}')

# Lists for Input and Output
X, Y = [], []

# Generate input data
for class_id in range(num_classes):
    for _ in range(samples_per_class):
        x = torch.distributions.MultivariateNormal(means[class_id], covs[class_id]).sample()
        X.append(x)

# Create corresponding one hot encoding class labels
classes = [torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 1.0, 0.0]), torch.tensor([0.0, 0.0, 1.0])]
for i in range(num_classes):
    for _ in range(samples_per_class):
        Y.append(classes[i])
               
# Combine to single tensors
X = torch.stack(X, dim=0)
Y = torch.stack(Y, dim=0)

# copy for plotting
data_copy = copy.deepcopy(X)

# Plot the base data
plot_samples(X, num_classes, samples_per_class)

# Shuffle the data, maintains correct labels
perm = torch.randperm(X.size(0))
X, Y = X[perm], Y[perm]

# Slice tensors to create train test split
split_idx = int(train_ratio * total_samples)
X_train, Y_train = X[:split_idx], Y[:split_idx]
X_test, Y_test = X[split_idx:], Y[split_idx:]

# copy of training data
tdata_copy = copy.deepcopy(X_train)

# Calculate number of samples from each class in the test & train set
train_samples = count_samples(Y_train, classes)
test_samples = count_samples(Y_test, classes)

# Dict to store average classification accuracy at each step
train_dict = {i: [0] * num_training_steps for i in range(num_classes)}
test_dict = {i: [0] * num_training_steps for i in range(num_classes)}
# dict to store accuracy for each seed
per_seed = {'train': [[[] for _ in range(num_training_steps)] for _ in range(num_seeds)],
            'test':  [[[] for _ in range(num_training_steps)] for _ in range(num_seeds)]}


# Model Instantiation. Set seeds for num_seeds trials with random weights & 0 bias
for i in range(num_seeds):
    torch.manual_seed(i)
    model = LinearClassifier(input_dim, num_classes)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Training loop
    for j in range(num_training_steps):

        if j < 5:
            scale_samples(X_train, Y_train, scalars, decay_param=1)
            scale_samples(X, Y, scalars, decay_param=1)
        elif j < 10:
            scale_samples(X_train, Y_train, scalars, decay_param=0.5)
            scale_samples(X, Y, scalars, decay_param=0.5)

        y_pred = model(X_train)
        track_accuracy(y_pred, j, train_dict, Y_train, num_classes, per_seed, train_samples, i, key='train')
        loss = criterion(y_pred, Y_train)

        optimizer.zero_grad()
        loss.backward()

        # plotting the linear classifier, based on the weights and gradients
        w = model.linear.weight
        b = model.linear.bias


        monitor_parameters(X, X_train, num_classes, w.data.detach(), b.data.detach(),
                            j, i, samples_per_class, w.grad.detach(), b.grad.detach())

        # Can use same monitor params for thing or not? Maybe just change this shit up?
        X_train = tdata_copy
        X = data_copy
        tdata_copy = copy.deepcopy(tdata_copy)
        data_copy = copy.deepcopy(data_copy)

        # Log weights and gradients
        print(f"{chr(10)}Step: {j+1}")
        print("Weights:", model.linear.weight.data)
        print("Biases:", model.linear.bias.data)
        print("Weight Gradients:", model.linear.weight.grad)
        print("Bias Gradients:", model.linear.bias.grad)

        optimizer.step()
        
        # Test loop
        with torch.no_grad():
            y_pred = model(X_test)
            track_accuracy(y_pred, j, test_dict, Y_test, num_classes, per_seed, test_samples, i, key='test')


# make decision boundary animation
make_animation(num_seeds, num_training_steps)

# Divides the number of correct classifications at each step location in the list
# by the total number of possible correct classifications over the total number of training steps
for i in range(num_classes):
    train_div = train_samples[i]*num_seeds
    test_div = test_samples[i]*num_seeds
    for j in range(num_training_steps):
        train_dict[i][j] /= train_div
        test_dict[i][j] /= test_div

# print(test_dict)

plot_accuracy(train_dict, test_dict)
seed_plot(per_seed, num_seeds, num_training_steps)