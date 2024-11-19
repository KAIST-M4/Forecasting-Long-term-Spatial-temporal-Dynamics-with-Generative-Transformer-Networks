# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 14:13:25 2024

@author: user
"""

# -*- coding: utf-8 -*-
"""
Title: Forecasting Long-term Spatial-temporal Dynamics with Generative Transformer NetworksAuthor: Donggeun Park
Description: 
This project aims to train and deploy a Spatiotemporal Vision Transformer (ViT) for various scientific and engineering applications, including climate science, crack propagation, and 3D reaction diffusion. The code covers the training of both a GAN model for data generation and a Vision Transformer for spatiotemporal prediction.

Key Features:
- GAN-based generator and discriminator models
- Vision Transformer (ViT) for spatiotemporal analysis
- Configurations for different scenarios, including climate science, crack propagation, etc.
- Visualization of results comparing true vs predicted data


How to Run:
1. Clone the repository.
2. Install the required dependencies using pip:
    pip install -r requirements.txt
3. Run the script by specifying the scenario as a command-line argument (e.g., 'python main.py climate_science'):
    python main.py climate_science
4. The script will train the models and visualize the results.

Dependencies:
- Please refer "DynamicGPT.yml" file

"""
# -*- coding: utf-8 -*-

# Import necessary libraries
import argparse
# Other imports
import sys
import os
import numpy as np
import tensorflow as tf

# Add paths to sys.path
script_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(script_dir, 'Neural Network Architecture'))
sys.path.append(os.path.join(script_dir, 'Configuration'))
sys.path.append(os.path.join(script_dir, 'util'))
sys.path.append(os.path.join(script_dir, 'Results'))

# Argument Parsing for Scenario Input

def parse_args():
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Run Spatiotemporal ViT model with the specified scenario.")
    
    # Add the 'scenario' argument with choices to limit the input options
    parser.add_argument("scenario", type=str, choices=["climate_science", "crack_propagation", "3D_reaction_diffusion", "flow_past_cylinder"],
                        help="Scenario to run (choose from 'climate_science', 'crack_propagation', '3D_reaction_diffusion', 'flow_past_cylinder').")
    
    # Parse the arguments and return them
    return parser.parse_args()

# Call the argument parser
args = parse_args()

# Get the selected scenario
scenario = args.scenario

print(f"Running scenario: {scenario}")
# Import modules for GAN training and model architecture
from gan_trainer import GANTrainer
from discriminator import Discriminator
from mnet_models import MNetGenerator
from get_configuration import get_configuration
from DatasetLoader import DatasetLoader

# Load the scenario configuration
print(f"Loading configuration for the scenario: {scenario}")
config = get_configuration(scenario)  # Retrieve configuration for the given scenario
print("Scenario configuration loaded successfully!")

# Stage 1: Train GAN (Generator and Discriminator)
print("Initializing the GAN model...")

generator = MNetGenerator(config["input_shape"], config["depth"], config["feature_maps"], config["upsample_configs"], config["downsample_configs"])
discriminator = Discriminator(config["dis_config"])

trainer = GANTrainer(generator, discriminator, learning_rate=2e-4, beta_1=0.5)

# Load dataset and prepare for training
module_dir = os.path.join(os.path.dirname(__file__), 'Results')
npy_file_pathX = os.path.join(module_dir, 'SST_train_x.npy')
npy_file_pathY = os.path.join(module_dir, 'SST_train_y.npy')

X = np.load(npy_file_pathX)
Y = np.load(npy_file_pathY)

dataset = tf.data.Dataset.from_tensor_slices((X, Y))
dataset = dataset.batch(4)

# Train the models using the GANTrainer
print("Training cGAN model...")
gen_loss_history, disc_loss_history = trainer.train(dataset, epochs=1000)
print("GAN training completed!")

#Stage 2: Train Vision Transformer (ViT) model
from vit_config import ViTConfig
from vit_model_ import SpatiotemporalViT
from data_preprocessing_vit import DataPreprocessor
from vit_model_compiler import ModelCompiler

# Create a ViTConfig object based on the scenario configuration
print("Initializing ViT model...")
vit_config = ViTConfig(**config["vit_config"])

# Create the SpatiotemporalViT model
vit_model = SpatiotemporalViT(vit_config)

# Set batch size and max patches
batch_size = 4
max_patches = config["vit_config"]["max_patches"]
data_preprocessor = DataPreprocessor(batch_size=batch_size, max_patches=max_patches, patch_shape=config["vit_config"]["patch_shape"])

dataset = data_preprocessor.prepare_dataset()

# Initialize and compile the model
model_compiler = ModelCompiler(model=vit_model)
model_compiler.compile_model()
print("ViT model compiled successfully!")

# Train the ViT model
epochs = 10
print(f"Training ViT model for {epochs} epochs...")
vit_model.fit(dataset, epochs=epochs)
print("ViT training completed!")

# Visualize the results
from Visualizer import Visualizer

pre = generator.predict(X)  # Replace with actual prediction data
visualizer = Visualizer(pre, X)
visualizer.visualize()
print("Visualization completed!")
