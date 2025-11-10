# experiments/linear_experiment.py
import torch
import copy
from torch.utils.data import DataLoader, TensorDataset
from refactor.experiments.base_experiment import BaseExperiment
from refactor.models.linear_classifier import LinearClassifier
from refactor.utils.functions import *

class LinearExperiment(BaseExperiment):
    def __init__(self):
            super().__init__()  # Call the parent's __init__
            self.input_dim = self.cfg["input_dim"],
            self.num_classes = self.cfg["num_classes"],
            self.radius = self.cfg["radius"],
            self.samples_per_class = self.cfg["samples_per_class"],
            self.scalars = self.cfg["scalars"],
            self.rotations = self.cfg["rotations"]
            self.train_ratio = self.cfg["train_ratio"]
            # flags for fairness projections
            self.flags = self.cfg["flags"]
            self.apply_projection = self.cfg["apply_projection"]
            self.target = self.cfg["target"]
            self.projection_mode = self.cfg["projection_mode"]

            self.means = 0

    def build_model(self):
        # reproducible model weight initialization
        torch.manual_seed(42)
        return LinearClassifier(
            input_dim=self.input_dim,
            num_classes=self.num_classes,
        )

    def get_dataloaders(self):

        means = make_evenly_spaced_targets(self.num_classes, self.radius)

        for i in range(self.num_classes):
            means[i] *= self.scalars[i]
            # FIXME make function
            # if need to rotate, then rotate
            if self.rotations != [0 for _ in range(self.num_classes)]:
                means = rotate_classes(means, self.rotations)

        self.means = copy.deepcopy(means)

        # Set covariance matrices. Establishes spread & direction of probability cluster
        covs = [torch.eye(2) for _ in range(num_classes)]

        # Generate input data
        X = generate_samples(means, covs, self.num_classes, self.samples_per_class)

        # Create corresponding one hot encoding class labels
        classes = [c for c in torch.eye(self.num_classes)]
        Y = create_labels(self.num_classes, self.samples_per_class, classes)
                    
        # Combine to single tensors
        X = torch.stack(X, dim=0)
        Y = torch.stack(Y, dim=0)

        # Projects clusters to create ETF class means for equal convergence
        scalars_or_shifts = transform_to_even_space(means, self.projection_mode, self.target)

        # Plot initial data
        plot_samples(X, self.num_classes, self.samples_per_class)

        # Shuffle the data, maintains correct labels
        perm = torch.randperm(X.size(0))
        X, Y = X[perm], Y[perm]

        # Slice tensors to create train test split
        split_idx = int(self.train_ratio * total_samples)
        X_train, Y_train = X[:split_idx], Y[:split_idx]
        X_test, Y_test = X[split_idx:], Y[split_idx:]

        # copy for plotting & projections
        data_copy = X.clone()

        # Calculate number of samples from each class in the test & train set
        train_samples = count_samples(Y_train, classes)
        test_samples = count_samples(Y_test, classes)

        # Dict to store average classification accuracy at each step
        train_dict, test_dict, per_seed = initialize_accuracy_tracking(num_classes, num_training_steps, num_seeds)
        # train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=32, shuffle=True)
        # val_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=32)
        return train_samples, test_samples, data_copy

    def compute_loss(self, batch):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        preds = self.model(x)
        return torch.nn.functional.mse(preds, y)

    def train_epoch(self, loader):
       # Model Instantiation. Set seeds for num_seeds trials with random weights & 0 bias
        for seed in range(self.num_seeds):
            torch.manual_seed(seed)
            model = LinearClassifier(self.input_dim, self.num_classes)

            if self.flags["evo_weights"]:
                model.linear.weight.data = evo_weights(num_iter=1, pop_size=100000, weights=model.linear.weight.data, means=self.means)

            # Loss function and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.SGD(model.parameters(), lr=0.01)

            previous_avg_percent_correct = 0

            # Training loop
            for step in range(num_training_steps):
                decay = 1 - previous_avg_percent_correct

                if self.apply_projection:
                    if projection_mode == 'scale' or projection_mode == 'norm':
                        scale_samples(X, Y, scalars_or_shifts, decay)
                        projected_means = means * scalars_or_shifts[:, None]  # (num_classes, 2)
                    elif projection_mode == 'shift':
                        shift_samples(X, Y, scalars_or_shifts, decay)
                        projected_means = means + scalars_or_shifts  # or scalars_or_shifts directly
                    
                    print(f'Equilateral after transform: {is_regular_polygon(projected_means)}')
                    print(f'Projected means: {projected_means}')

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
                
                plot_decision_boundaries(X, Y, classes, num_classes, w, b, step, seed)

                previous_avg_percent_correct = track_accuracy(
                    y_pred, step, train_dict, Y_train, num_classes, per_seed, train_samples, seed, key='train'
                )

                # Reset data for next step
                X = data_copy.clone()
                X_train = X[:split_idx]
                X_test = X[split_idx:]

                # update gradients
                optimizer.step()
    
    def validate_epoch(self, loader):
        # Evaluation Step
        with torch.no_grad():
            y_pred_test = model(X_test)
            previous_avg_percent_correct = track_accuracy(
                y_pred_test, step, test_dict, Y_test, num_classes, per_seed, test_samples, seed, key='test'
            )

    
    def on_run_end(self, batch):
        # make decision boundary animation
        make_animation(num_seeds, num_training_steps)

        # compute avg accuracy for plotting
        compute_accuracies(num_classes, train_samples, test_samples, num_seeds, num_training_steps, train_dict, test_dict)

        # call function to plot average and per seed accuracy
        plot_avg_accuracy(train_dict, test_dict, num_classes)
        seed_plot(per_seed, num_seeds, num_training_steps)
        return 0
