"""
VAE Training with PyTorch Profiling and TensorBoard support.
Optimized for Windows Multiprocessing.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST
from torchvision.utils import save_image
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler, schedule

# --- Hyperparameters ---
dataset_path = "datasets"
device_name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
DEVICE = torch.device(device_name)
batch_size = 100
x_dim = 784
hidden_dim = 400
latent_dim = 20
lr = 1e-3
epochs = 2 

# --- Model Definitions ---
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim) -> None:
        super().__init__()
        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_mean = nn.Linear(hidden_dim, latent_dim)
        self.FC_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h_ = torch.relu(self.FC_input(x))
        mean = self.FC_mean(h_)
        log_var = self.FC_var(h_)
        std = torch.exp(0.5 * log_var)
        z = self.reparameterization(mean, std)
        return z, mean, log_var

    def reparameterization(self, mean, std):
        epsilon = torch.randn_like(std)
        return mean + std * epsilon

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim) -> None:
        super().__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = torch.relu(self.FC_hidden(x))
        return torch.sigmoid(self.FC_output(h))

class VAE(nn.Module):
    def __init__(self, encoder, decoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        z, mean, log_var = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, mean, log_var

def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return reproduction_loss + kld

# --- Main Training Function ---
def train():
    # 1. Data Loading (Inside train() to prevent Windows spawn issues)
    train_dataset = MNIST(dataset_path, train=True, download=True)
    train_dataset = TensorDataset(train_dataset.data.float() / 255.0, train_dataset.targets)
    
    # Using num_workers > 0 to profile parallel data loading
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2,         
        pin_memory=True        
    )

    # 2. Model Initialization
    encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=x_dim)
    model = VAE(encoder=encoder, decoder=decoder).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=lr)

    # 3. Profiling Schedule
    my_schedule = schedule(wait=5, warmup=2, active=6, repeat=1)

    print(f"Start profiling VAE on {device_name}...")
    model.train()

    # 4. Profiler Context
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if torch.cuda.is_available() else [ProfilerActivity.CPU],
        schedule=my_schedule,
        on_trace_ready=tensorboard_trace_handler("./log/vae_mnist"),
        record_shapes=True,
        profile_memory=True,  
        with_stack=True       
    ) as prof:
        
        for epoch in range(epochs):
            overall_loss = 0
            for batch_idx, (x, _) in enumerate(train_loader):
                # Non-blocking move to device
                x = x.view(batch_size, x_dim).to(DEVICE, non_blocking=True)
                
                optimizer.zero_grad()
                x_hat, mean, log_var = model(x)
                loss = loss_function(x, x_hat, mean, log_var)
                
                # Detach to avoid CPU-GPU sync stalls
                overall_loss += loss.detach() 
                
                loss.backward()
                optimizer.step()
                
                # Update profiler state
                prof.step()
                
                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f}")
            
            print(f"Epoch {epoch+1} Complete. Average Loss: {overall_loss.item() / (len(train_loader)*batch_size):.4f}")

    print("Profiling Finished! Run 'uv run tensorboard --logdir=./log' to see results.")

# --- Windows Safe Entry Point ---
if __name__ == '__main__':
    train()