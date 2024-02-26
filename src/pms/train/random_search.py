import random
from gan_train import train_gan

# Define hyperparameter search space
hyperparams_space = {
    'latent_dim': [50, 100, 150],
    'batch_size': [16, 32, 64],
    'learning_rate': [1e-4, 1e-3, 1e-2],
    'beta1': [0.1, 0.5, 0.9],
    'generator': {
        'first_layer_size': [64, 128, 256],
        'second_layer_size': [64, 128, 256],
    },
    'discriminator': {
        'first_layer_size': [64, 128, 256],
        'second_layer_size': [64, 128, 256],
    },
    'epochs': [5000, 10000, 15000],  # Reduced for quicker iterations during hyperparameter search
}

# Function to generate a random configuration
def random_config(space):
    config = {}
    for param, values in space.items():
        if isinstance(values, dict):
            config[param] = random_config(values)  # Recursive call for nested dictionaries
        else:
            config[param] = random.choice(values)
    return config

# Number of configurations to try
num_configs = 10

best_performance = float('inf')  # Assuming lower is better; adjust accordingly
best_config = None

print("Starting random search for best hyperparameters...")

for i in range(num_configs):
    print(f"\nTesting configuration {i+1}/{num_configs}...")
    config = random_config(hyperparams_space)
    # Update config with fixed parameters if any
    config.update({
        'leaky_relu_alpha': 0.2,
        'beta2': 0.999,
        'gen_model_path': './models/generator_state_dict.pth',
        'dis_model_path': './models/discriminator_state_dict.pth'
    })

    print(f"Current configuration: {config}")

    # Train the GAN with the generated configuration
    performance = train_gan(config)

    print(f"Configuration {i+1} performance: {performance}")

    # Evaluate the performance and update the best configuration
    if performance < best_performance:
        best_performance = performance
        best_config = config
        print(f"New best configuration found: {best_config} with performance: {best_performance}")

print(f"\nRandom search completed. Best configuration: {best_config} with performance: {best_performance}")
