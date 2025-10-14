import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models import VAE

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

# Model (you should define VAE elsewhere)
model = VAE(784, 16)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

num_epochs = 1

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}")
    model.train()
    for batch_idx, (inputs, _) in enumerate(train_loader):
        inputs = inputs.view(inputs.size(0), -1)  # Flatten

        y_pred, kl_loss, mean, log_var = model(inputs)
        loss = criterion(y_pred, inputs)  # reconstruct input
        loss += kl_loss

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
            loss += kl_loss
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    print(f"Test Loss: {avg_loss:.4f}")
