# experiments/vae_experiment.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from experiments.base_experiment import BaseExperiment
from models.vae import VAE
from utils import *

class VAEExperiment(BaseExperiment):
    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)    # Call the parent's init

    def build_model(self):
        return VAE(
            input_dim=self.cfg["input_dim"],
            latent_dim=self.cfg["latent_dim"],
            encoder_layers=self.cfg["encoder_layers"],
            decoder_layers=self.cfg["decoder_layers"],
            activation=self.cfg["activation"]
        )

    def get_dataloaders(self):
        # Transform
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # Data
        train_dataset = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
        test_dataset  = datasets.MNIST(root='./data', train=False, download=False, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)
        return train_loader, test_loader

    def compute_loss(self, batch):
        x = batch
        x = x.to(self.device)
        x_hat, kl_loss, mu, log_var = self.model.forward(x, self.cfg.get('kl_reduction', 'batchmean').lower())

        recon = self.loss_function(x_hat, x)
        total = recon + self.cfg.get("kl_weight", 0.001) * kl_loss

        return total

    def train_epoch(self, loader):
        self.model.train()
        for batch_idx, (inputs, _) in enumerate(loader):
            inputs = inputs.view(inputs.size(0), -1)  # Flatten

            loss = self.compute_loss(inputs)  # reconstruct input

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch_idx % 100 == 0:
                print(f"Train Batch {batch_idx}, Loss: {loss.item():.4f}")

        

    def validate_epoch(self, loader):
        self.model.eval()
        total_loss = 0
        for batch_idx, (inputs, _) in enumerate(loader):
            inputs = inputs.view(inputs.size(0), -1)  # Flatten

            loss = self.compute_loss(inputs)  # reconstruct input
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Test Loss: {avg_loss:.4f}")


    def on_run_end(self):
        """Define any steps occuring after main training loops if necessary."""
        print("Run Finished!")