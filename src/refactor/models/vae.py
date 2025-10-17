import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_layer_sizes, latent_dim, activation="relu"):
        """
        A flexible MLP encoder that maps input → latent parameters (mu, log_var)
        Args:
            input_dim (int): dimensionality of input data
            hidden_layer_sizes (list[int]): list of hidden layer sizes
            latent_dim (int): size of latent space
            activation (str): activation function to use ("relu", "leakyrelu", "gelu", "sigmoid")
        """
        super().__init__()

        # pick activation function
        act_fn = self._get_activation(activation)

        # build the hidden stack
        layers = []
        prev_dim = input_dim
        for hidden_size in hidden_layer_sizes:
            layers.append(nn.Linear(prev_dim, hidden_size))
            layers.append(act_fn)
            prev_dim = hidden_size

        # combine layers into one sequential module
        self.hidden_layers = nn.Sequential(*layers)

        # final layers that output distribution parameters
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

    def _get_activation(self, name):
        name = name.lower()
        if name == "relu":
            return nn.ReLU()
        elif name == "leakyrelu":
            return nn.LeakyReLU(0.2)
        elif name == "gelu":
            return nn.GELU()
        elif name == "sigmoid":
            return nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation '{name}'")

    def forward(self, x):
        hidden = self.hidden_layers(x)
        mu = self.fc_mu(hidden)
        log_var = self.fc_logvar(hidden)
        return mu, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_layer_sizes, output_dim, activation="relu"):
        """
        Build a flexible MLP decoder that maps latent → reconstructed output.
        """
        super().__init__()
        act_fn = self._get_activation(activation)

        layers = []
        prev_dim = latent_dim
        for hidden_size in hidden_layer_sizes:
            layers.append(nn.Linear(prev_dim, hidden_size))
            layers.append(act_fn)
            prev_dim = hidden_size

        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, output_dim)

    def _get_activation(self, name):
        name = name.lower()
        if name == "relu":
            return nn.ReLU()
        elif name == "leakyrelu":
            return nn.LeakyReLU(0.2)
        elif name == "gelu":
            return nn.GELU()
        elif name == "sigmoid":
            return nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation '{name}'")

    def forward(self, z):
        hidden = self.hidden_layers(z)
        reconstruction = torch.sigmoid(self.output_layer(hidden))
        return reconstruction


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, encoder_layers, decoder_layers, activation="relu"):
        """
        Full VAE wrapper that builds encoder, decoder, and handles reparameterization.
        """
        super().__init__()
        self.encoder = Encoder(input_dim, encoder_layers, latent_dim, activation)
        self.decoder = Decoder(latent_dim, decoder_layers, input_dim, activation)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def get_kl_loss(self, mu, log_var, reduction='batchmean'):
        # computes kl_loss. defaults for mse is mean, which equals batchmean
        kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
        if reduction == "sum":
            return kl.sum()
        elif reduction == "mean":
            return kl.mean()  # average over batch and latent dims
        elif reduction == "batchmean":
            return kl.sum() / mu.size(0)  # average per sample
        else:
            return kl

    def forward(self, x, kl):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decoder(z)

        # KL divergence loss
        kl_loss = self.get_kl_loss(mu, log_var, kl)

        return x_hat, kl_loss, mu, log_var
