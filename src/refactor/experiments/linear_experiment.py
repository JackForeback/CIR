# experiments/linear_experiment.py
import torch
from torch.utils.data import DataLoader, TensorDataset
from experiments.base_experiment import BaseExperiment
from models.linear_classifier import LinearClassifier
from utils import *

class LinearExperiment(BaseExperiment):
    def build_model(self):
        return LinearClassifier(
            input_dim=self.cfg["input_dim"],
            radius=self.cfg["radius"],
            num_classes=self.cfg["num_classes"],
            means = make_evenly_spaced_targets(num_classes, radius)
        )

    def get_dataloaders(self):

        means, covs, num_classes, samples_per_class

        X_train = torch.randn(1000, self.cfg["input_dim"])
        y_train = torch.randint(0, self.cfg["num_classes"], (1000,))
        X_val = torch.randn(200, self.cfg["input_dim"])
        y_val = torch.randint(0, self.cfg["num_classes"], (200,))
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)
        return train_loader, val_loader

    def compute_loss(self, batch):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        preds = self.model(x)
        return torch.nn.functional.cross_entropy(preds, y)

