import torch
import torch.nn as nn
import math

class DCTLinear(nn.Module):
    """
    A fixed linear layer using Discrete Cosine Transform (DCT) basis functions.

    Args:
        input_dim (int): Input feature dimension (e.g. 784)
        output_dim (int): Output feature dimension (e.g. 128)
        norm (bool): Whether to apply orthonormal normalization (default=True)
    """
    def __init__(self, input_dim, output_dim, norm=True):
        super(DCTLinear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = norm

        # Build the fixed DCT weight matrix (output_dim × input_dim)
        weight = self._build_dct_weights(input_dim, output_dim, norm)
        self.register_buffer('weight', weight)  # fixed, not learnable

    @staticmethod
    def _build_dct_weights(N, K, norm=True):
        """
        Build a (K x N) DCT basis matrix where each row is a DCT basis vector.
        """
        # Initialize matrix
        W = torch.zeros(K, N)

        # Compute DCT-II basis functions
        n = torch.arange(N).float()
        for k in range(K):
            W[k, :] = torch.cos(math.pi / N * (n + 0.5) * k)
            if norm:
                if k == 0:
                    W[k, :] *= math.sqrt(1 / N)
                else:
                    W[k, :] *= math.sqrt(2 / N)

        return W

    def forward(self, x):
        """
        x: (batch_size, input_dim)
        returns: (batch_size, output_dim)
        """
        return torch.matmul(x, self.weight.t())

