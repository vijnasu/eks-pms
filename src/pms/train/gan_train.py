import yaml
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Load configuration
#with open('gan_config.yaml', 'r') as file:
#    config = yaml.safe_load(file)

# Define a function to encapsulate the training process
def train_gan(config, data_path='./data/MLC_Idle_Memory_Latency_Local_Random.csv'):

    # Load and preprocess the data
    df = pd.read_csv('./data/MLC_Idle_Memory_Latency_Local_Random.csv')

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

    # Loss function
    adversarial_loss = nn.BCELoss()

    # Training loop
    for epoch in range(config['epochs']):
        for i, (features, targets) in enumerate(dataloader):
            valid = torch.ones((features.size(0), 1), requires_grad=False)
            fake = torch.zeros((features.size(0), 1), requires_grad=False)

            # Train Generator
            optimizer_G.zero_grad()
            z = torch.randn(features.size(0), config['latent_dim'])
            generated_targets = generator(z)
            g_loss = adversarial_loss(discriminator(torch.cat((features, generated_targets), 1)), valid)
            g_loss.backward()
            optimizer_G.step()

            # Train Discriminator
            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(torch.cat((features, targets), 1)), valid)
            fake_loss = adversarial_loss(discriminator(torch.cat((features, generated_targets.detach()), 1)), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

        # Print progress
        if epoch % 1000 == 0 or epoch == config['epochs'] - 1:
            print(f"Epoch: {epoch} | D Loss: {d_loss.item()} | G Loss: {g_loss.item()}")

    # Save the trained models
    torch.save(generator.state_dict(), config['gen_model_path'])
    torch.save(discriminator.state_dict(), config['dis_model_path'])
    print("Trained models have been saved.")
