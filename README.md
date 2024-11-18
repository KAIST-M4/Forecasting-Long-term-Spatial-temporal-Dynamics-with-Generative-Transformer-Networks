# Spatiotemporal Vision Transformer for Forecasting Dynamics

## Code Overview
This project aims to forecast long-term spatiotemporal dynamics using **Generative Transformer Networks** (GTNs). The core of this code is the combination of **conditional Generative Adversarial Networks (cGANs)** and **Vision Transformers (ViT)** for long-term spatiotemporal prediction across various scientific and engineering applications such as **climate science**, **crack propagation in composite material**, and **3D reaction diffusion** as shown **Figure 1**.

The framework trains a **cGAN model** to generate synthetic data and a **Vision Transformer (ViT)** model to predict spatiotemporal dynamics over time. The models are designed to handle large spatial and temporal data, making them suitable for tasks involving **scientific simulations** and **real-world predictions**.

## Key Features in DynamicGPT.py (Main python file):
- **conditional GAN-based Generator and Discriminator Models**: Used for generating synthetic data and training the discriminator to differentiate between real and generated data.
- **Spatiotemporal Vision Transformer (ViT)**: A Transformer-based model that processes spatial and temporal dynamics simultaneously for tasks such as forecasting and prediction.
- **Scenario-Based Configurations**: Configurations for various scientific and engineering scenarios (e.g., climate science, crack propagation, reaction diffusion, etc.).
- **Visualization**: Visualization of the results, comparing true data with predicted data, including error metrics.

## Supported Scenarios (Configurations of optimized hyper paramters):
- **crack_propagation**: Used for modeling the progression of cracks in materials.
- **3D_reaction_diffusion**: Simulates reaction-diffusion processes in 3D environments.
- **flow_past_cylinder**: Models fluid dynamics around objects like cylinders.
- **climate_science**: Simulates and forecasts climate-related data, such as temperature and precipitation patterns.

## Requirements:
- **TensorFlow** (>= 2.6)
- **NumPy** (1.23.5)
- **Additional Modules**: Refer to the `requirements.txt` file for the full list of dependencies.

## Setup and Installation:

### Step 1. Clone the repository:
```bash
git clone https://github.com/DonggeunPark/DynamicGPT.git
```

### Step 2. Navigate into the project folder and create the Conda environment from "DynamciGPT.yml":
```bash
cd DynamicGPT
conda env create -f DynamicGPT.yml
conda activate DynamicGPT
```

### Step 3. Run and enjoy **DynamicGPT** (training, inference, visualization):
```bash
python main.py climate_science
```

## Code Structure

- **main.py**: The main script to train and test the models. It takes a scenario as input (e.g., `climate_science`) and executes the training process.
- **gan_trainer.py**: Defines the `GANTrainer` class, which is responsible for handling the training of the generator and discriminator models within the GAN framework.
- **discriminator.py**: Contains the `Discriminator` class, which is used in GANs to classify whether the input data is real or generated.
- **mnet_models.py**: Defines the `MNetGenerator` class, which represents the generator model used in GAN training for generating data.
- **vit_model.py**: Contains the `SpatiotemporalViT` class, which uses a Vision Transformer (ViT) for spatiotemporal forecasting tasks, handling both the encoder and the decoder in the transformer architecture.
- **data_preprocessing_vit.py**: Provides functions for preprocessing the data to prepare it for the Vision Transformer model, including tasks such as patch extraction.
- **visualizer.py**: Contains functions for visualizing the model’s predictions, comparing them with the true values to assess performance.

## Detailed Project Structure
```bash
project/
├── Neural Network Architecture/
│   ├── vit_model_.py
│   ├── discriminator.py
│   ├── mnet_models.py
├── Configuration/
│   ├── vit_config.py
│   ├── get_configuration.py
├── util/
│   ├── data_preprocessing_vit.py
│   ├── DatasetLoader.py
│   ├── gan_trainer.py
│   ├── Visualizer.py
│   ├── vit_model_compiler
├── Results/
│   ├── SST_train-x.npy
│   ├── SST_train-y.npy
├── main.py
```
