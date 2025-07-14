import torch
import torch.nn as nn
import torch.optim as optim
import math as m
import copy

from models import LinearClassifier
from plotting import *
from utils import *

# Set variables & initial seed for reproducible data generation
num_classes = 3
num_training_steps = 50
num_seeds = 1
input_dim = 2
samples_per_class = 10000
train_ratio = 0.7
total_samples = samples_per_class * num_classes
torch.manual_seed(42)

# Sets coordinates used for cluster centers (can do manually)
height = 10
coord = m.sqrt((height**2)/2)
# height = even_space(height)

# Set means for 2D Gaussians
means = [torch.tensor([0.0, 20]),
         torch.tensor([-coord, -coord]),
         torch.tensor([coord, -coord])]

# Set covariance matrices. Establishes spread & direction of probability cluster
covs = [torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([[1.0, 0.0], [0.0, 1.0]])]

# Generate input data
X = generate_samples(means, covs, num_classes, samples_per_class)

# Create corresponding one hot encoding class labels
classes = [c for c in torch.eye(num_classes)]
Y = create_labels(num_classes, samples_per_class, classes)
               
# Combine to single tensors
X = torch.stack(X, dim=0)
Y = torch.stack(Y, dim=0)
means = torch.stack(means)
covs = torch.stack(covs)

# Determine if projection is necessary (i.e., not already ETF)
apply_projection = not is_regular_polygon(means)
projection_mode = 'shift'  # 'scale' or 'shift'

# Projects clusters to create ETF class means for equal convergence
# methods = max-norm, mean-norm, max-ETF, median-ETF, max-shift, median-shift
scalars_or_shifts = transform_to_even_space(means, mode=projection_mode, ref_mode='median')

# Print means, scalars, and covariance to output file to verify distributions used in each test
print(f'Means: {means} Covs: {covs} Projection ({projection_mode}): {scalars_or_shifts}')

# Plot initial data
plot_samples(X, num_classes, samples_per_class)

# Shuffle the data, maintains correct labels
perm = torch.randperm(X.size(0))
X, Y = X[perm], Y[perm]

# Slice tensors to create train test split
split_idx = int(train_ratio * total_samples)
X_train, Y_train = X[:split_idx], Y[:split_idx]
X_test, Y_test = X[split_idx:], Y[split_idx:]

# copy for plotting & projections
data_copy = X.clone()

# Calculate number of samples from each class in the test & train set
train_samples = count_samples(Y_train, classes)
test_samples = count_samples(Y_test, classes)

# Dict to store average classification accuracy at each step
train_dict, test_dict, per_seed = initialize_accuracy_tracking(num_classes, num_training_steps, num_seeds)


# Model Instantiation. Set seeds for num_seeds trials with random weights & 0 bias
for seed in range(num_seeds):
    torch.manual_seed(seed)
    model = LinearClassifier(input_dim, num_classes)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    previous_avg_percent_correct = 0

    # Training loop
    for step in range(num_training_steps):
        decay = 1 - previous_avg_percent_correct

        if apply_projection:
            if projection_mode == 'scale':
                scale_samples(X, Y, scalars_or_shifts, decay)
                projected_means = means * scalars_or_shifts[:, None]  # (num_classes, 2)
            elif projection_mode == 'shift':
                shift_samples(X, Y, scalars_or_shifts, decay)
                projected_means = means + scalars_or_shifts  # or scalars_or_shifts directly
            
            print(f'Equilateral after transform(decay):{decay} {is_regular_polygon(projected_means)}')
            print(f'projected means: {projected_means}')


        # make predictions, compute gradients
        y_pred = model(X_train)
        loss = criterion(y_pred, Y_train)
        optimizer.zero_grad()
        loss.backward()

        # Track and plot
        w, b = model.linear.weight.data, model.linear.bias.data
        wg, bg = model.linear.weight.grad.detach(), model.linear.bias.grad.detach()
        
        monitor_parameters(X, Y, classes, num_classes, w, b, step, seed, samples_per_class, wg, bg)

        previous_avg_percent_correct = track_accuracy(
            y_pred, step, train_dict, Y_train, num_classes, per_seed, train_samples, seed, key='train'
        )

        # Reset data for next step
        X = data_copy.clone()
        X_train = X[:split_idx]
        X_test = X[split_idx:]

        # update gradients
        optimizer.step()

        # Evaluation Step
        with torch.no_grad():
            y_pred_test = model(X_test)
            previous_avg_percent_correct = track_accuracy(
                y_pred_test, step, test_dict, Y_test, num_classes, per_seed, test_samples, seed, key='test'
            )


# make decision boundary animation
make_animation(num_seeds, num_training_steps)

# Divides the number of correct classifications at each step location in the list
# by the total number of possible correct classifications over the total number of training steps
for i in range(num_classes):
    train_div = train_samples[i]*num_seeds
    test_div = test_samples[i]*num_seeds
    # print(f'train_div ; {train_div}')
    for j in range(num_training_steps):
        train_dict[i][j] /= train_div
        test_dict[i][j] /= test_div

# call function to plot average and per seed accuracy
plot_avg_accuracy(train_dict, test_dict)
seed_plot(per_seed, num_seeds, num_training_steps)