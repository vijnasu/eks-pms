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

# Load configuration
#with open('gan_config.yaml', 'r') as file:
#    config = yaml.safe_load(file)

def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(real_samples.device)
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
def train_gan(config, data_path='./data/MLC_Idle_Memory_Latency_Local_Random.csv'):

    print(colored("Starting GAN training...", "cyan"))
    
    # Load and preprocess the data
    df = pd.read_csv('./data/MLC_Idle_Memory_Latency_Local_Random.csv')
    
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
        # Now you can drop the original 'DateTime' column
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
                nn.Linear(config['generator']['second_layer_size'], output_dim)
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
                nn.Linear(config['discriminator']['second_layer_size'], 1)
                # Removed Sigmoid activation
            )

        def forward(self, data):
            return self.model(data)

    # Initialize models and optimizers
    input_dim = input_features.shape[1] + target.shape[1]  # Adding target dimension for discriminator
    output_dim = target.shape[1]  # Only target dimension for generator
    generator = Generator(config['latent_dim'], output_dim)
    discriminator = Discriminator(input_dim)

    optimizer_G = optim.Adam(generator.parameters(), lr=config['learning_rate'], betas=(config['beta1'], config['beta2']))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=config['learning_rate'], betas=(config['beta1'], config['beta2']))

    # Initialize learning Rate Schedulers
    lr_scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=config['step_size'], gamma=config['gamma'])  # Example decay
    lr_scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=config['step_size'], gamma=config['gamma'])  # Example decay

    # Loss function
    adversarial_loss = nn.BCELoss()

    print(colored("Models initialized. Starting training loop...", "green"))
    
    # Initialize lists to track losses
    d_losses = []
    g_losses = []

    # Training loop with gradient clipping and dynamic LR scheduler update
    
    for epoch in range(config['epochs']):
        for i, (features, targets) in enumerate(dataloader):
            # Train Discriminator
            optimizer_D.zero_grad()

            # Generate a batch of images
            z = torch.randn(features.size(0), config['latent_dim'])
            generated_targets = generator(z)

            # Real images
            real_loss = discriminator(torch.cat((features, targets), 1))
            # Fake images
            fake_loss = discriminator(torch.cat((features, generated_targets.detach()), 1))
            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(discriminator, torch.cat((features, targets), 1).data, torch.cat((features, generated_targets), 1).data)
            # Discriminator loss
            d_loss = -torch.mean(real_loss) + torch.mean(fake_loss) + config['lambda_gp'] * gradient_penalty

            d_loss.backward()
            optimizer_D.step()

            # Train the generator every n_critic iterations
            if i % config['n_critic'] == 0:
                optimizer_G.zero_grad()

                # Generate a batch of images
                generated_targets = generator(z)
                # Generator loss
                g_loss = -torch.mean(discriminator(torch.cat((features, generated_targets), 1)))

                g_loss.backward()
                optimizer_G.step()

            # Logging
            if i % 100 == 0:
                print(colored(f"Batch {i} | Epoch: {epoch} | D Loss: {d_loss.item()} | G Loss: {g_loss.item()}", "blue"))

        # Epoch-level logging
        if epoch % 100 == 0:
            print(colored(f"Epoch {epoch} | Avg D Loss: {np.mean(d_losses[-100:])} | Avg G Loss: {np.mean(g_losses[-100:])}", "magenta"))

    print(colored("Training completed!", "cyan"))

    # Performance evaluation
    avg_d_loss = np.mean(d_losses[-100:])  # Adjust the range as needed
    avg_g_loss = np.mean(g_losses[-100:])  # Adjust the range as needed
    performance_metric = (avg_d_loss + avg_g_loss) / 2
    print(colored(f"Final Performance Metric: {performance_metric}", "magenta"))

    # Return the performance metric for further evaluation if needed
    return performance_metric