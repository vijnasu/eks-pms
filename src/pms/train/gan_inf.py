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
from gan_train import Generator

# Load configuration
with open('gan_config_final.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Load the test data
df = pd.read_csv('./data/MLC_Idle_Memory_Latency_Local_Random.csv')

# Preprocess the test data
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
# No need to scale target for inference in GAN, as we're generating new data

# Convert to PyTorch tensors
tensor_input_features = torch.Tensor(input_features_scaled)
# Create dummy tensor for targets since we're only interested in generating data, not pairing it with real targets
dummy_targets = torch.zeros(tensor_input_features.size(0), 1)  # Adjust the size if necessary
dataset = TensorDataset(tensor_input_features, dummy_targets)
dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)

# Initialize the model
input_dim = config['latent_dim']
output_dim = target.shape[1]  # Adjust according to your model's expected output dimension
model = Generator(input_dim, output_dim)

# Load the state dictionary
state_dict = torch.load('./models/generator_final.pth')
model.load_state_dict(state_dict)

# Set the model to evaluation mode
model.eval()

# Perform inference on the test data
generated_data_list = []
with torch.no_grad():
    for batch in dataloader:
        inputs, _ = batch  # We don't need the targets
        generated_batch = model(inputs)
        generated_data_list.append(generated_batch)

# Concatenate all generated data batches
generated_data = torch.cat(generated_data_list, dim=0)

# Process the generated data (if needed)
# ...

# Print or save the generated data
print(generated_data)
