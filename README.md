# Spatiotemporal Vision Transformer for Forecasting Dynamics

## Project Overview
This project aims to forecast long-term spatiotemporal dynamics using **Generative Transformer Networks** (GTNs). The core of this project is the combination of **conditional Generative Adversarial Networks (cGANs)** and **Vision Transformers (ViT)** for various scientific and engineering applications such as **climate science**, **crack propagation**, and **3D reaction diffusion**.

The framework trains a **GAN model** to generate synthetic data and a **Vision Transformer (ViT)** model to predict spatiotemporal dynamics over time. The models are designed to handle large spatial and temporal data, making them suitable for tasks involving scientific simulations and real-world predictions.

## Key Features in DynamicGPT.py (Main python file):
- **conditional GAN-based Generator and Discriminator Models**: Used for generating synthetic data and training the discriminator to differentiate between real and generated data.
- **Spatiotemporal Vision Transformer (ViT)**: A Transformer-based model that processes spatial and temporal dynamics simultaneously for tasks such as forecasting and prediction.
- **Scenario-Based Configurations**: Configurations for various scientific and engineering scenarios (e.g., climate science, crack propagation, reaction diffusion, etc.).
- **Visualization**: Visualization of the results, comparing true data with predicted data, including error metrics.

## Supported Scenarios (Configuration:
- **crack_propagation**: Used for modeling the progression of cracks in materials.
- **3D_reaction_diffusion**: Simulates reaction-diffusion processes in 3D environments.
- **flow_past_cylinder**: Models fluid dynamics around objects like cylinders.
- **climate_science**: Simulates and forecasts climate-related data, such as temperature and precipitation patterns.

## Requirements:
- **TensorFlow** (>= 2.6)
- **NumPy** (1.23.5)
- **Additional Modules**: Refer to the `requirements.txt` file for the full list of dependencies.

## Setup and Installation:

### 1. Clone the repository:
```bash
git clone https://github.com/yourusername/spatiotemporal-forecasting.git
cd spatiotemporal-forecasting
