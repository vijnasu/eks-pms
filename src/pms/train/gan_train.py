import yaml
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Load configuration
with open('gan_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Load and preprocess the data
df_idle_random = pd.read_csv('./data/MLC_Idle_Memory_Latency_Local_Random.csv')

# Example for selecting features for core 0
features = df_idle_random[['core0_base_freq', 'core0_min_freq', 'core0_max_freq', 'c0_CoreTmp', 'pkg0_PkgWatt']]
features_scaled = StandardScaler().fit_transform(features)


# Convert scaled features to PyTorch tensors
tensor_data = torch.Tensor(features_scaled)
dataset = TensorDataset(tensor_data)
dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

# GAN Components

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, config['generator']['first_layer_size']),
            nn.LeakyReLU(config['leaky_relu_alpha']),
            nn.Linear(config['generator']['first_layer_size'], config['generator']['second_layer_size']),
            nn.LeakyReLU(config['leaky_relu_alpha']),
            nn.Linear(config['generator']['second_layer_size'], 5)  # Output dimension adjusted to 5
        )
    
    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(5, config['discriminator']['first_layer_size']),  # Input dimension adjusted to 5
            nn.LeakyReLU(config['leaky_relu_alpha']),
            nn.Linear(config['discriminator']['first_layer_size'], config['discriminator']['second_layer_size']),
            nn.LeakyReLU(config['leaky_relu_alpha']),
            nn.Linear(config['discriminator']['second_layer_size'], 1),
            nn.Sigmoid()
        )
    
    def forward(self, z):
        return self.model(z)

# Initialize models and optimizers
generator = Generator(config['latent_dim'])
discriminator = Discriminator()

optimizer_G = optim.Adam(generator.parameters(), lr=config['learning_rate'], betas=(config['beta1'], config['beta2']))
optimizer_D = optim.Adam(discriminator.parameters(), lr=config['learning_rate'], betas=(config['beta1'], config['beta2']))

# Loss function
adversarial_loss = nn.BCELoss()

# Training Loop
for epoch in range(config['epochs']):
    for i, (real_data,) in enumerate(dataloader):
        valid = torch.ones((real_data.size(0), 1), requires_grad=False)
        fake = torch.zeros((real_data.size(0), 1), requires_grad=False)

        # Train Generator
        optimizer_G.zero_grad()
        z = torch.randn(real_data.size(0), config['latent_dim'])
        gen_data = generator(z)
        g_loss = adversarial_loss(discriminator(gen_data), valid)
        g_loss.backward()
        optimizer_G.step()

        # Train Discriminator
        optimizer_D.zero_grad()
        real_loss = adversarial_loss(discriminator(real_data), valid)
        fake_loss = adversarial_loss(discriminator(gen_data.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

    # Print progress for every 1000 epochs or on the last epoch
    if epoch % 1000 == 0 or epoch == config['epochs'] - 1:
        print(f"Epoch: {epoch} | D Loss: {d_loss.item()} | G Loss: {g_loss.item()}")

# Save the trained models
torch.save(generator.state_dict(), config['gen_model_path'])
torch.save(discriminator.state_dict(), config['dis_model_path'])
print("Trained models have been saved.")
