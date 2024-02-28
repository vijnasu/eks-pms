import torch
import torch.nn as nn
import pandas as pd

# Load the GAN model
model = torch.load('./models/generator_final.pth')

# Load the test data
test_data = pd.read_csv('./data/MLC_Idle_Memory_Latency_Local_Random.csv')

# Preprocess the test data (if needed)
# ...

# Convert the test data to PyTorch tensors
test_data = torch.tensor(test_data.values, dtype=torch.float32)

# Set the model to evaluation mode
model.eval()

# Perform inference on the test data
with torch.no_grad():
    generated_data = model(test_data)

# Process the generated data (if needed)
# ...

# Print the generated data
print(generated_data)