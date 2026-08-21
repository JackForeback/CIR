"""Original VAE reconstruction script for MNIST.

Part of the *original*, pre-refactor codebase; see ``README.md``. The
config-driven equivalent is ``../refactor/cir/experiments/vae.py``.

Run it through ``scripts/run_vae.sh``, or directly:

    python vae.py --epochs 1
"""

import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import VAE
from utils import parse_sysargs

# --- Configuration -----------------------------------------------------------
args = parse_sysargs()

num_epochs = args.get('epochs', 1)
batch_size = args.get('batch_size', 64)
latent_dim = args.get('latent_dim', 16)
learning_rate = args.get('lr', 0.01)
kl_weight = args.get('kl_weight', 0.001)

# MNIST is committed at the repository root so runs work without a download.
data_root = args.get('data_root', os.path.join(os.path.dirname(__file__), '..', 'data'))

torch.manual_seed(42)

# Transform. Kept in [0, 1] rather than standardized: the reconstruction target
# has to stay in a range the decoder can actually reach.
transform = transforms.ToTensor()

# Data
train_dataset = datasets.MNIST(root=data_root, train=True, download=False, transform=transform)
test_dataset = datasets.MNIST(root=data_root, train=False, download=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model
model = VAE(784, latent_dim)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# --- Training ----------------------------------------------------------------
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}")
    model.train()
    for batch_idx, (inputs, _) in enumerate(train_loader):
        inputs = inputs.view(inputs.size(0), -1)  # Flatten

        y_pred, kl_loss, mean, log_var = model(inputs)
        loss = criterion(y_pred, inputs)  # reconstruct input
        # The KL term is summed over the batch while the reconstruction term is
        # averaged, so it needs a weight or it swamps the reconstruction.
        loss = loss + kl_weight * kl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Train Batch {batch_idx}, Loss: {loss.item():.4f}")

    # Evaluation
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.view(inputs.size(0), -1)
            y_pred, kl_loss, mean, log_var = model(inputs)
            loss = criterion(y_pred, inputs)
            loss = loss + kl_weight * kl_loss
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    print(f"Test Loss: {avg_loss:.4f}")
