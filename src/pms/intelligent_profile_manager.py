import numpy as np
from termcolor import colored

import torch
import torch.nn as nn
import yaml

# Load configuration
with open('gan_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(config['latent_dim'], config['generator']['first_layer_size']),
            nn.LeakyReLU(config['leaky_relu_alpha']),
            nn.Linear(config['generator']['first_layer_size'], config['generator']['second_layer_size']),
            nn.LeakyReLU(config['leaky_relu_alpha']),
            nn.Linear(config['generator']['second_layer_size'], 3)  # Assuming 3 output features
        )

    def forward(self, z):
        return self.model(z)

# Initialize and load the generator model
generator = Generator(config)
generator.load_state_dict(torch.load(config['gen_model_path']))
generator.eval()  # Set the model to evaluation mode

def preprocess_data(core_metrics):
    """
    Preprocess the core metrics for model input.
    
    Parameters:
        core_metrics (dict): A dictionary containing metrics of a core.
    
    Returns:
        np.array: An array of processed data ready for model input.
    """
    # Example for core0, adjust indices/names based on actual usage
    processed_data = np.array([core_metrics['core0_base_freq'], core_metrics['core0_min_freq'],
                               core_metrics['core0_max_freq'], core_metrics['c0_CoreTmp'],
                               core_metrics['pkg0_PkgWatt']])
    return processed_data

def predict_profile(processed_data):
    """
    Predict the optimal profile using the pre-trained PyTorch model.
    
    Parameters:
        processed_data (np.array): The preprocessed data for prediction.
    
    Returns:
        str: The predicted profile.
    """
    # Convert numpy array to PyTorch tensor
    processed_data_tensor = torch.from_numpy(processed_data).float()

    # Use the generator model for prediction
    with torch.no_grad():
        predicted_metrics = generator(processed_data_tensor.unsqueeze(0)).numpy().flatten()

    # Interpret the predicted metrics to decide on the power profile
    # This is a sample placeholder; we need to replace it with our logic based on the model's output
    # Use the threshold from the config to interpret the model's output
    if predicted_metrics[0] > config['prediction_threshold']:
        return "high_performance"
    else:
        return "power_saving"

def determine_profile_for_core(core):
    """
    Determine the optimal profile for a core based on its real-time metrics.

    Parameters:
        core: The core object with its current metrics.

    Returns:
        str: The optimal profile for the core.
    """
    # Refresh core stats to get updated values
    core.refresh_stats()

    # Gather real-time metrics
    core_metrics = {
        'current_frequency': core.curr_freq,  # Current frequency of the core
        'min_frequency': core.min_freq,  # Minimum frequency the core is allowed to run at
        'max_frequency': core.max_freq,  # Maximum frequency the core is allowed to run at
        'core_temp': core.thread_siblings[0].temperature,  # Assuming temperature can be obtained from the first thread sibling
        'power_consumption': core.cpu.power_consumption  # Power consumption from the associated CPU object
    }

    # Note: The 'core_temp' attribute is hypothetical as the pwr module documentation does not specify a direct way to obtain core temperature.
    # We might need to use an external tool or library to get the core temperature.

    # Preprocess data
    processed_data = preprocess_data(core_metrics)

    # Predict the optimal profile using the pre-trained model
    profile = predict_profile(processed_data)

    return profile

def intelligent_apply_profiles(cores, default_profile="default"):
    """
    Apply profiles intelligently to each core based on real-time metrics and a pre-trained ML model.
    
    Parameters:
        cores (list): A list of core objects.
        default_profile (str): The default profile to use if prediction fails.
    """
    for core in cores:
        try:
            profile = determine_profile_for_core(core)
        except Exception as e:
            print(colored(f"Error determining profile for core {core.core_id}: {e}", "red"))
            profile = default_profile
        core.commit(profile)
        print(colored(f"Applied '{profile}' profile to Core {core.core_id} based on real-time metrics.", "green"))
