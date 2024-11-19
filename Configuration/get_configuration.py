# -*- coding: utf-8 -*-
"""
Title: Configuration for Spatiotemporal Vision Transformer (ViT) and GAN Models
Author: Donggeun Park
Description:
This module provides a function to load configurations for different scientific and engineering scenarios 
involving spatiotemporal analysis using GAN-based and Vision Transformer models. The available scenarios 
are 'crack_propagation', '3D_reaction_diffusion', 'flow_past_cylinder', and 'climate_science'.
It returns the corresponding configuration dictionaries required for the model training, including 
the input shapes, feature maps, layer configurations, and ViT-specific settings.

Key Features:
- Handles configurations for different research scenarios such as crack propagation and climate science.
- Configures GAN architecture and Vision Transformer (ViT) settings for spatiotemporal prediction.
- Provides the ability to customize each scenario with layers, feature maps, and downsampling/upsampling settings.

Usage:
- Import this function in your training pipeline.
- Use the function `get_configuration(scenario)` where `scenario` can be one of the following: 
  'crack_propagation', '3D_reaction_diffusion', 'flow_past_cylinder', 'climate_science'.

Dependencies:
- No external dependencies required for this specific module.

"""

def get_configuration(scenario):
    """
    Load configuration settings for a given scenario.

    Args:
    scenario (str): The name of the scenario to retrieve configurations for. 
                     Options: 'crack_propagation', '3D_reaction_diffusion', 
                              'flow_past_cylinder', 'climate_science'.

    Returns:
    dict: A dictionary containing the configuration settings for the specified scenario, 
          including input shapes, depth, feature maps, ViT configurations, and discriminator settings.
    """

    # Configuration for the 'crack_propagation' scenario
    if scenario == "crack_propagation":
        return {
            "input_shape": (132, 132, 3),  # Input shape for crack propagation (1 channel)
            "depth": 3,  # Layer depth
            "feature_maps": [32, 32, 16],  # Feature maps across layers
            "downsample_configs": [(66, 66), (33, 33), (16, 16)],  # Downsampling configurations for each layer
            "upsample_configs": [(16, 4, (16, 16)), (32, 4, (33, 33)), (32, 4, (66, 66))],  # Upsampling configurations
            "vit_config": {
                "image_size": 16,
                "patch_size": 16,
                "patch_shape": (16, 16, 16),  # Updated feature map size
                "num_layers": 6,
                "hidden_size": 2048,
                "num_heads": 6,
                "mlp_dim": 2048,
                "max_patches": 3  # Max patches should be equal to the depth of the input
            },
            "dis_config": {
                "input_shape": (132, 132, 3),
                "target_shape": (132, 132, 1),
                "num_layers": 3,
                "base_filters": 32,
                "filter_multiplier": 2,
                "kernel_size": 4
            }
        }

    # Configuration for '3D_reaction_diffusion' scenario
    elif scenario == "3D_reaction_diffusion":
        return {
            "input_shape": (80, 80, 30),  # Input shape for the 3D reaction-diffusion process
            "depth": 3,  # Model depth
            "feature_maps": [64, 32, 16, 8],
            "downsample_configs": [(40, 40), (20, 20), (10, 10)],
            "upsample_configs": [(16, 4, (10, 10)), (32, 4, (20, 20)), (64, 4, (40, 40)), (64, 4, (80, 80))],
            "vit_config": {
                "image_size": 8,
                "patch_size": 8,
                "patch_shape": (8, 8, 16),  # Adjusted feature map size for 3D data
                "num_layers": 8,
                "hidden_size": 1024,
                "num_heads": 8,
                "mlp_dim": 1024,
                "max_patches": 30
            },
            "dis_config": {
                "input_shape": (80, 80, 30),
                "target_shape": (80, 80, 1),
                "num_layers": 3,
                "base_filters": 32,
                "filter_multiplier": 2,
                "kernel_size": 4
            }
        }

    # Configuration for 'flow_past_cylinder' scenario
    elif scenario == "flow_past_cylinder":
        return {
            "input_shape": (256, 384, 10),  # Input shape for flow past cylinder
            "depth": 10,  # Input depth
            "feature_maps": [64, 64, 64, 64, 64, 64],  # Consistent feature maps
            "downsample_configs": [(128, 192), (64, 96), (32, 48), (16, 24), (8, 12), (4, 4)],
            "upsample_configs": [(64, 4, (8, 12)), (64, 4, (16, 24)), (64, 4, (32, 48)), (64, 4, (64, 96)), (64, 4, (128, 192))],
            "vit_config": {
                "image_size": 4,
                "patch_size": 4,
                "patch_shape": (4, 4, 64),  # Updated patch shape after encoding
                "num_layers": 4,
                "hidden_size": 1024,
                "num_heads": 4,
                "mlp_dim": 1024,
                "max_patches": 10
            },
            "dis_config": {
                "input_shape": (256, 384, 10),
                "target_shape": (256, 384, 1),
                "num_layers": 3,
                "base_filters": 32,
                "filter_multiplier": 2,
                "kernel_size": 4
            }
        }

    # Configuration for 'climate_science' scenario
    elif scenario == "climate_science":
        return {
            "input_shape": (180, 360, 12),  # Input shape for climate science
            "depth": 7,  # Depth based on the number of layers
            "feature_maps": [64, 64, 64, 64, 64, 64, 64],  # Feature maps for each layer
            "downsample_configs": [(90, 180), (45, 90), (23, 45), (12, 23), (6, 12), (3, 6), (2, 2)],
            "upsample_configs": [(64, 4, (3, 6)), (64, 4, (6, 12)), (64, 4, (12, 23)), (64, 4, (23, 45)), (64, 4, (45, 90)), (64, 4, (90, 180)), (64, 4, (180, 360))],
            "vit_config": {
                "image_size": 2,
                "patch_size": 2,
                "patch_shape": (2, 2, 64),  # Patch shape for ViT, based on the final encoder output
                "num_layers": 2,
                "hidden_size": 256,
                "num_heads": 2,
                "mlp_dim": 256,
                "max_patches": 12  # Max patches equals the depth of input
            },
            "dis_config": {
                "input_shape": (180, 360, 12),
                "target_shape": (180, 360, 1),
                "num_layers": 3,
                "base_filters": 32,
                "filter_multiplier": 2,
                "kernel_size": 4
            }
        }

    # Raise error if an unknown scenario is passed
    else:
        raise ValueError(f"Unknown scenario: {scenario}")