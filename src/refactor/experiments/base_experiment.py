# experiments/base_experiment.py
import torch
import torch.nn as nn
from abc import ABC, abstractmethod


# FIXME MAYBE MAKE THESE ALL ABSTRATC METHODS SO THAT THEY HAVE TO BE DEFINED FOR EACH THING SPECIFICALLY.
# FIXME KEEP OPTIMS AND STUFF BECAUSE YOU CAN JUST SET THOSE FROM CONFIG.

class BaseExperiment(ABC):
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.build_model().to(self.device)
        self.optimizer = self.get_optimizer()
        self.loss_function = self.get_loss_function()
        self.history = {"train_loss": [], "val_loss": []}

    @abstractmethod
    def build_model(self):
        """Return the PyTorch model instance."""
        pass

    @abstractmethod
    def get_dataloaders(self):
        """Return train_loader, val_loader."""
        pass

    @abstractmethod
    def compute_loss(self, batch):
        """Define how to compute loss from a batch."""
        pass

    @abstractmethod
    def train_epoch(self, loader):
        """Define a train epoch for your model."""
        pass

    @abstractmethod
    def validate_epoch(self, loader):
        """Define a validation epoch for your model."""
        pass

    @abstractmethod
    def on_run_end(self):
        """Define any steps occuring after main training loops if necessary."""
        pass

    def get_optimizer(self):
        """Default optimizer setup."""
        optim = self.cfg.get("optimizer", "adam").lower()
        lr = self.cfg.get("lr", 1e-3)
        if optim == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=lr)
        elif optim == "sgd":
            return torch.optim.SGD(self.model.parameters(), lr=lr, momentum=int(self.cfg.get("momentum", 0)))
        elif optim == "rmsprop":
            return torch.optim.RMSProp(self.model.parameters(), lr=lr)
        elif optim == "adagrad":
            return torch.optim.Adagrad(self.model.parameters(), lr=lr)
        elif optim == "adadelta":
            return torch.optim.Adagrad(self.model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer '{optim}'")
        
    def get_loss_function(self):
        """Default loss function setup."""
        loss_function = self.cfg.get("loss_function").lower()
        if loss_function == "mse":
            return nn.MSELoss()
        elif loss_function == "cross_entropy":
            return nn.CrossEntropyLoss()
        elif loss_function == "bce":
            return nn.BCELoss()
        elif loss_function == "nll":
            return nn.NLLLoss()
        elif loss_function == "l1":
            return nn.L1Loss()
        else:
            raise ValueError(f"Unsupported optimizer '{loss_function}'")

    def run(self):
        """Main training loop"""
        train_loader, val_loader = self.get_dataloaders()
        epochs = self.cfg.get("epochs", 1)

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate_epoch(val_loader)
            self.logger.log(epoch, train_loss, val_loss)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        self.on_run_end()

