import yaml
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable, grad
from termcolor import colored
import matplotlib.pyplot as plt
import os

# Function to compute the gradient penalty for WGAN-GP
def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    alpha = torch.rand(real_samples.size(0), 1).to(real_samples.device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(torch.Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False).to(real_samples.device)
    gradients = grad(outputs=d_interpolates, inputs=interpolates,
                     grad_outputs=fake, create_graph=True, retain_graph=True,
                     only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# Define a function to encapsulate the training process
# Define a function to encapsulate the GAN training process
def train_gan(config, data_path='./data/MLC_Idle_Memory_Latency_Local_Random.csv'):
    print(colored("Starting GAN training...", "cyan"))

    # Load and preprocess the data
    df = pd.read_csv(data_path)
    print(colored("Data loaded and preprocessing initiated...", "yellow"))

    # If 'DateTime' column exists, convert it to datetime and extract features
    if 'DateTime' in df.columns:
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df['Year'] = df['DateTime'].dt.year
        df['Month'] = df['DateTime'].dt.month
        df['Day'] = df['DateTime'].dt.day
        df['Hour'] = df['DateTime'].dt.hour
        df['Minute'] = df['DateTime'].dt.minute
        df['Second'] = df['DateTime'].dt.second
        df.drop(columns=['DateTime'], inplace=True)

    # Ensure all features are numeric, exclude non-numeric
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    non_numeric_columns = set(df.columns) - set(numeric_columns) - {'MLC_Idle_Memory_Latency_Local_Random'}
    if non_numeric_columns:
        print(f"Excluding non-numeric columns: {non_numeric_columns}")
        df.drop(columns=non_numeric_columns, inplace=True)

    # Selecting relevant features and the target
    input_features = df.drop(columns=['MLC_Idle_Memory_Latency_Local_Random'])
    target = df[['MLC_Idle_Memory_Latency_Local_Random']]

    # Standardize the features
    scaler = StandardScaler()
    input_features_scaled = scaler.fit_transform(input_features)
    target_scaled = scaler.fit_transform(target)

    # Convert to PyTorch tensors
    tensor_input_features = torch.Tensor(input_features_scaled)
    tensor_target = torch.Tensor(target_scaled)
    dataset = TensorDataset(tensor_input_features, tensor_target)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    # Generator
    class Generator(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(Generator, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim, config['generator']['first_layer_size']),
                nn.LeakyReLU(config['leaky_relu_alpha']),
                nn.Linear(config['generator']['first_layer_size'], config['generator']['second_layer_size']),
                nn.LeakyReLU(config['leaky_relu_alpha']),
                nn.Linear(config['generator']['second_layer_size'], output_dim),
            )

        def forward(self, z):
            return self.model(z)

    # Discriminator
    class Discriminator(nn.Module):
        def __init__(self, input_dim):
            super(Discriminator, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim, config['discriminator']['first_layer_size']),
                nn.LeakyReLU(config['leaky_relu_alpha']),
                nn.Linear(config['discriminator']['first_layer_size'], config['discriminator']['second_layer_size']),
                nn.LeakyReLU(config['leaky_relu_alpha']),
                nn.Linear(config['discriminator']['second_layer_size'], 1),  # No sigmoid activation
            )

        def forward(self, data):
            return self.model(data)

    # Initialize models and optimizers
    input_dim = input_features.shape[1] + target.shape[1]
    output_dim = target.shape[1]
    generator = Generator(config['latent_dim'], output_dim)
    discriminator = Discriminator(input_dim)

    optimizer_G = optim.Adam(generator.parameters(), lr=config['learning_rate'], betas=(config['beta1'], config['beta2']))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=config['learning_rate'], betas=(config['beta1'], config['beta2']))

    # Initialize learning Rate Schedulers
    lr_scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=config['step_size'], gamma=config['gamma'])
    lr_scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=config['step_size'], gamma=config['gamma'])

    print(colored("Models initialized. Starting training loop...", "green"))
    
    # Initialize lists to track losses
    d_losses = []
    g_losses = []

    for epoch in range(config['epochs']):
        for i, (features, targets) in enumerate(dataloader):
            # Configure input
            real_imgs = torch.cat((features, targets), 1)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Sample noise as generator input
            z = torch.randn(features.size(0), config['latent_dim'])

            # Generate a batch of images
            fake_imgs = generator(z)
            fake_imgs = torch.cat((features, fake_imgs), 1)

            # Real images
            real_validity = discriminator(real_imgs)
            # Fake images
            fake_validity = discriminator(fake_imgs.detach())
            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data)
            # Discriminator loss
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + config['lambda_gp'] * gradient_penalty

            d_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()

            # Train the generator every n_critic steps
            if i % config['n_critic'] == 0:
                # -----------------
                #  Train Generator
                # -----------------
                # Generate a batch of images
                gen_imgs = generator(z)
                gen_imgs = torch.cat((features, gen_imgs), 1)
                # Loss measures generator's ability to fool the discriminator
                g_loss = -torch.mean(discriminator(gen_imgs))

                g_loss.backward()
                optimizer_G.step()

                print(colored(f"Epoch {epoch} | Batch {i} | D Loss: {d_loss.item()} | G Loss: {g_loss.item()}", "blue"))

        d_losses.append(d_loss.item())
        g_losses.append(g_loss.item())

        # Update learning rate schedulers
        lr_scheduler_G.step()
        lr_scheduler_D.step()

        if epoch % config['save_interval'] == 0:  # save_interval can be defined as per your preference
            torch.save(generator.state_dict(), os.path.join('models', f'generator_epoch_{epoch}.pth'))
            torch.save(discriminator.state_dict(), os.path.join('models', f'discriminator_epoch_{epoch}.pth'))
            print(f"Checkpoint saved at epoch {epoch}")

    print(colored("Training completed!", "cyan"))

    if not os.path.isdir('models'):
        os.makedirs('models')  # Create directory for saving models if it doesn't exist

    # Save the final models
    torch.save(generator.state_dict(), os.path.join('models', 'generator_final.pth'))
    torch.save(discriminator.state_dict(), os.path.join('models', 'discriminator_final.pth'))
    print("Final models saved!")

    # Evaluate performance
    avg_d_loss = np.mean(d_losses[-100:])
    avg_g_loss = np.mean(g_losses[-100:])
    performance_metric = (avg_d_loss + avg_g_loss) / 2
    print(colored(f"Final Performance Metric: {performance_metric}", "magenta"))

    return performance_metric

# main function
def main():
    # Load configuration
    with open('gan_config_final.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Load and preprocess the data
    df_idle_random = pd.read_csv('./data/MLC_Idle_Memory_Latency_Local_Random.csv')

    # Train the GAN with the generated configuration
    performance = train_gan(config)

if __name__ == "__main__":
    main()
