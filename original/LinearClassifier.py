"""Original linear-classifier fairness experiment (driver script).

Part of the *original*, pre-refactor codebase; see ``README.md``. The
config-driven equivalent is ``../refactor/cir/experiments/linear.py``.

What it does:

1. Places `num_classes` cluster centres evenly on a circle -- a 2D simplex ETF --
   then deliberately breaks that symmetry by scaling one cluster outward.
2. Draws Gaussian samples around those centres and trains a linear classifier
   with MSE against one-hot targets, over `num_seeds` independent runs.
3. Records per-class accuracy at every step and plots the max-minus-min gap,
   which is the quantity the project cares about: equal geometry should mean
   equal convergence.

Three interventions can be switched on with the flags below: projecting the data
back onto the ETF, penalizing the between-class gap in the loss, and searching
for a low-entropy initialization instead of sampling one.

Run it through ``scripts/run_linear.sh``, which sets the arguments and creates
the output directories, or directly:

    python LinearClassifier.py --path output --num_classes 3 \\
        --num_training_steps 50 --num_seeds 1 --input_dim 2 \\
        --samples_per_class 10000 --train_ratio 0.7
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim

from models import LinearClassifier
from plotting import *
from utils import *

# --- Configuration -----------------------------------------------------------
# Set variables & initial seed for reproducible data generation
# The original used `locals().update(parsed_dict)`, which happens to work at
# module scope but silently does nothing inside a function and leaves every
# name undefined if an argument is omitted. Reading each key explicitly, with a
# default, is the same idea without either trap.
parsed_dict = parse_sysargs()

path = parsed_dict.get('path', 'output')
num_classes = parsed_dict.get('num_classes', 3)
num_training_steps = parsed_dict.get('num_training_steps', 50)
num_seeds = parsed_dict.get('num_seeds', 1)
input_dim = parsed_dict.get('input_dim', 2)
samples_per_class = parsed_dict.get('samples_per_class', 10000)
train_ratio = parsed_dict.get('train_ratio', 0.7)

total_samples = samples_per_class * num_classes
torch.manual_seed(42)

# Output directories used by the plotting functions.
for sub in ('db', 'ani', 'seed'):
    os.makedirs(f'{path}/{sub}', exist_ok=True)

# --- Experiment flags --------------------------------------------------------
per_class_gap = False        # penalize the gap in per-class MSE
soft_accuracy_gap = False    # penalize the gap in per-class softmax confidence
apply_projection = False     # warp the data toward the ETF each step
use_evo_weights = False      # search for a low-entropy initialization
plot_boundaries = True       # save a decision-boundary frame per step (slow)

ref = 'median'               # ref_mode (str): 'mean', 'median', or 'max'
projection_mode = 'scale'    # mode (str): 'shift' or 'scale'.

# --- Class geometry ----------------------------------------------------------
# Sets coordinates used for cluster centers (can do manually)
radius = 10
means = make_evenly_spaced_targets(num_classes, radius)

print(f'means: {means}')

# Break the ETF by pushing one cluster further from the origin.
cluster = 1
scalar = 2

means[cluster] *= scalar

print(f'after scale: {means}')
print(f'Equilateral before transform: {is_regular_polygon(means)}')

# Set covariance matrices. Establishes spread & direction of probability cluster
covs = [torch.eye(2) for _ in range(num_classes)]

# --- Data --------------------------------------------------------------------
# Generate input data
X = generate_samples(means, covs, num_classes, samples_per_class)

# Create corresponding one hot encoding class labels
classes = [c for c in torch.eye(num_classes)]
Y = create_labels(num_classes, samples_per_class, classes)

# Combine to single tensors
X = torch.stack(X, dim=0)
Y = torch.stack(Y, dim=0)

# Projects clusters to create ETF class means for equal convergence
scalars_or_shifts = transform_to_even_space(means, mode=projection_mode, ref_mode=ref)

# Print means, scalars, and covariance to output file to verify distributions used in each test
print(f'Means: {means} Covs: {covs} Projection ({projection_mode}): {scalars_or_shifts}')

# Plot initial data
plot_samples(X, num_classes, samples_per_class, path)

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

# --- Training ----------------------------------------------------------------
# Model Instantiation. Set seeds for num_seeds trials with random weights & 0 bias
for seed in range(num_seeds):
    torch.manual_seed(seed)
    model = LinearClassifier(input_dim, num_classes)

    if use_evo_weights:
        model.linear.weight.data = evo_weights(
            num_iter=1, pop_size=1000, tournament_size=100,
            weights=model.linear.weight.data, means=means
        )

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    previous_avg_percent_correct = 0

    # Training loop
    for step in range(num_training_steps):
        # Projection strength fades as the classifier improves.
        decay = 1 - previous_avg_percent_correct

        if apply_projection:
            if projection_mode == 'scale' or projection_mode == 'norm':
                scale_samples(X, Y, scalars_or_shifts, decay)
                projected_means = means * scalars_or_shifts[:, None]  # (num_classes, 2)
            elif projection_mode == 'shift':
                shift_samples(X, Y, scalars_or_shifts, decay)
                projected_means = means + scalars_or_shifts  # or scalars_or_shifts directly

            print(f'Equilateral after transform: {is_regular_polygon(projected_means)}')
            print(f'Projected means: {projected_means}')

            X_train, X_test = X[:split_idx], X[split_idx:]

        # make predictions, compute gradients
        y_pred = model(X_train)
        if per_class_gap:
            total_loss, mse_loss, fairness_loss = loss_with_per_class_gap(y_pred, Y_train, decay)
        elif soft_accuracy_gap:
            total_loss, mse_loss, fairness_loss = loss_with_soft_accuracy_gap(y_pred, Y_train, decay)
        else:
            total_loss = criterion(y_pred, Y_train)

        optimizer.zero_grad()
        total_loss.backward()

        # Track and plot
        w, b = model.linear.weight.data, model.linear.bias.data

        if plot_boundaries:
            plot_decision_boundaries(X, Y, classes, num_classes, w, b, step, seed, path)

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
            track_accuracy(
                y_pred_test, step, test_dict, Y_test, num_classes, per_seed, test_samples, seed, key='test'
            )

    print(f'seed {seed + 1}/{num_seeds} | loss {total_loss.item():.4f} | '
          f'train acc {previous_avg_percent_correct:.4f}')

# --- Output ------------------------------------------------------------------
# make decision boundary animation
if plot_boundaries:
    make_animation(num_seeds, num_training_steps, path)

# compute avg accuracy for plotting
compute_accuracies(num_classes, train_samples, test_samples, num_seeds, num_training_steps, train_dict, test_dict)

# call function to plot average and per seed accuracy
plot_avg_accuracy(train_dict, test_dict, num_classes, path)
seed_plot(per_seed, num_seeds, num_training_steps, path)

print(f'Figures written to {path}/')
