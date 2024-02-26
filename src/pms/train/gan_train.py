import yaml
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from termcolor import colored
import matplotlib.pyplot as plt

# Load configuration
#with open('gan_config.yaml', 'r') as file:
#    config = yaml.safe_load(file)

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
                nn.Linear(config['discriminator']['second_layer_size'], 1),
                nn.Sigmoid()
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
            valid = torch.ones((features.size(0), 1), requires_grad=False)
            fake = torch.zeros((features.size(0), 1), requires_grad=False)

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = torch.randn(features.size(0), config['latent_dim'])

            # Generate a batch of images
            generated_targets = generator(z)

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(torch.cat((features, generated_targets), 1)), valid)

            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1)  # Gradient clipping for the generator
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(torch.cat((features, targets), 1)), valid)
            fake_loss = adversarial_loss(discriminator(torch.cat((features, generated_targets.detach()), 1)), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1)  # Gradient clipping for the discriminator
            optimizer_D.step()

            # Update learning rate based on the scheduler
            lr_scheduler_G.step()
            lr_scheduler_D.step()
            
            # Logging
            if i % 100 == 0:  # Log every 100 batches
                print(colored(f"Batch {i} | Epoch: {epoch} | D Loss: {d_loss.item()} | G Loss: {g_loss.item()}", "blue"))
            
        # Epoch-level logging
        if epoch % 100 == 0:  # Adjust the epoch logging frequency as needed
            print(colored(f"Epoch {epoch} | Avg D Loss: {np.mean(d_losses[-100:])} | Avg G Loss: {np.mean(g_losses[-100:])}", "magenta"))

    print(colored("Training completed!", "cyan"))
    
    # Performance evaluation
    avg_d_loss = np.mean(d_losses[-100:])  # Adjust the range as needed
    avg_g_loss = np.mean(g_losses[-100:])  # Adjust the range as needed
    performance_metric = (avg_d_loss + avg_g_loss) / 2
    print(colored(f"Final Performance Metric: {performance_metric}", "magenta"))

    # Return the performance metric for further evaluation if needed
    return performance_metric