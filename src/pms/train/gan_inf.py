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

# Load configuration
with open('gan_config_final.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Load the GAN model
model = torch.load('./models/generator_final.pth')

# Load the test data
test_data = pd.read_csv('./data/MLC_Idle_Memory_Latency_Local_Random.csv')

# Preprocess the test data
df = pd.read_csv(test_data)
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

# Set the model to evaluation mode
model.eval()

# Perform inference on the test data
with torch.no_grad():
    generated_data = model(dataloader)

# Process the generated data (if needed)
# ...

# Print the generated data
print(generated_data)