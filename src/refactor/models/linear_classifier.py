import torch
import torch.nn as nn

# Model definition
class LinearClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        nn.init.zeros_(self.linear.bias)  # Set bias to zero

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
