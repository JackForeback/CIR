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
    

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.Encoder = nn.Sequential(
        nn.Linear(input_dim, 128),  
        nn.ReLU()
        )
        self.mu = nn.Linear(128, latent_dim)
        self.log_var = nn.Linear(128, latent_dim)
        self.Decoder = nn.Sequential(
        nn.Linear(latent_dim, 128),
        nn.ReLU(),
        nn.Linear(128, input_dim)
        )
        
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)      # sampling epsilon        
        z = mean + var*epsilon                          # reparameterization trick
        return z
        
                
    def forward(self, x):
        x = self.Encoder(x)
        mean = self.mu(x)
        log_var = self.log_var(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)
        x_hat            = self.Decoder(z)

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        
        return x_hat, kl_loss, mean, log_var
    

# Fixed Output Layer Variational AutoEncoder
class FOLVAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.Encoder = nn.Sequential(
        nn.Linear(input_dim, 128),  
        nn.ReLU()
        )
        self.mu = nn.Linear(128, latent_dim)
        self.log_var = nn.Linear(128, latent_dim)
        self.Decoder = nn.Sequential(
        nn.Linear(latent_dim, 128),
        nn.ReLU(),
        nn.Linear(128, input_dim)
        )
        self.linear128 = nn.Linear(input_dim,128)

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)      # sampling epsilon        
        z = mean + var*epsilon                          # reparameterization trick
        return z
        
                
    def forward(self, x, current_step):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)

        if (current_step % 2):
            x_hat = self.Decoder(z)  # DIMENSION 128. z is dim 10
        else:
            #LINEAR SOLVER AND THEN BACKPROP. HOW DO I DO THIS BEST?
            x_hat = self.linear128(z)
            self.Decoder.weights.freeze()

            soln = solver(x_hat)
            self.output(x_hat)
        
        return x_hat, mean, log_var
    

# Linear Alternating Variational AutoEncoder
class LAVAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.Encoder = nn.Sequential(
        nn.Linear(input_dim, 128),  
        nn.ReLU()
        )
        self.mu = nn.Linear(128, latent_dim)
        self.log_var = nn.Linear(128, latent_dim)
        self.Decoder = nn.Sequential(
        nn.Linear(latent_dim, 128),
        nn.ReLU(),
        nn.Linear(128, input_dim)
        )
        self.linear128 = nn.Linear(input_dim,128)

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)      # sampling epsilon        
        z = mean + var*epsilon                          # reparameterization trick
        return z
        
                
    def forward(self, x, current_step):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)

        if (current_step % 2):
            x_hat            = self.Decoder(z)  # DIMENSION 128. z is dim 10
        else:
            #LINEAR SOLVER AND THEN BACKPROP. HOW DO I DO THIS BEST?
            x_hat = self.linear128(z)
            self.Decoder.weights.freeze()

            soln = solver(x_hat)
            self.output(x_hat)
        
        return x_hat, mean, log_var


# Added Loss Variational AutoEncoder
class ALVAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.Encoder = nn.Sequential(
        nn.Linear(input_dim, 128),  
        nn.ReLU()
        )
        self.mu = nn.Linear(128, latent_dim)
        self.log_var = nn.Linear(128, latent_dim)
        self.Decoder = nn.Sequential(
        nn.Linear(latent_dim, 128),
        nn.ReLU(),
        nn.Linear(128, input_dim)
        )
        self.linear128 = nn.Linear(input_dim,128)

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)      # sampling epsilon        
        z = mean + var*epsilon                          # reparameterization trick
        return z
        
                
    def forward(self, x, current_step):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)

        if (current_step % 2):
            x_hat            = self.Decoder(z)  # DIMENSION 128. z is dim 10
        else:
            #LINEAR SOLVER AND THEN BACKPROP. HOW DO I DO THIS BEST?
            x_hat = self.linear128(z)
            self.Decoder.weights.freeze()

            soln = solver(x_hat)
            self.output(x_hat)
        
        return x_hat, mean, log_var