# experiments/vae_experiment.py
import torch
from torch.utils.data import DataLoader, TensorDataset
from experiments.base_experiment import BaseExperiment
from models.vae import VAE
from utils import *

class VAEExperiment(BaseExperiment):
    def build_model(self):
        return VAE(
            input_dim=self.cfg["input_dim"],
            latent_dim=self.cfg["latent_dim"],
            encoder_layers=self.cfg["encoder_layers"],
            decoder_layers=self.cfg["decoder_layers"],
        )

    def get_dataloaders(self):
        # Replace this with your real data generation
        # MNIST and shiffle yada yada yada
        X = torch.randn(2000, self.cfg["input_dim"])
        loader = DataLoader(TensorDataset(X, X), batch_size=64, shuffle=True)
        return loader, loader

    def compute_loss(self, batch):
        x, _ = batch
        x = x.to(self.device)
        x_hat, mu, log_var = self.model(x)

        recon = torch.nn.functional.mse_loss(x_hat, x)
        kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / len(x)
        total = recon + self.cfg.get("kl_weight", 0.001) * kl

        return total
