# experiments/base_experiment.py
import torch
from abc import ABC, abstractmethod


# FIXME MAYBE MAKE THESE ALL ABSTRATC METHODS SO THAT THEY HAVE TO BE DEFINED FOR EACH THING SPECIFICALLY.
# FIXME KEEP OPTIMS AND STUFF BECAUSE YOU CAN JUST SET THOSE FROM CONFIG.

class BaseExperiment(ABC):
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.build_model().to(self.device)
        self.optimizer = self.configure_optimizers()
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
        """Define how to compute loss from a batch."""
        pass

    @abstractmethod
    def validate_epoch(self, loader):
        """Define how to compute loss from a batch."""
        pass


    def configure_optimizers(self):
        """Default optimizer setup."""
        return torch.optim.Adam(self.model.parameters(), lr=self.cfg.get("lr", 1e-3))

    # FIXME maybe need run individualized too?????
    def run(self):
        """Main training loop"""
        train_loader, val_loader = self.get_dataloaders()
        epochs = self.cfg.get("epochs", 10)

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate_epoch(val_loader)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

