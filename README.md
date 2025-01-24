# Forecasting Long-term Spatial-temporal Dynamics with Generative Transformer Networks
![License](https://img.shields.io/badge/license-MIT-green) [![DOI](https://zenodo.org/badge/888896208.svg)](https://zenodo.org/badge/latestdoi/888896208)
## Code Overview
We introduce **DynamicGPT**, a generative model designed specifically for **dynamic** modeling by integrating **G**enerative learning with a **P**re-trained multi-scale kernel-based autoencoder and a vision **T**ansformer framework’. DynamicGPT distinguishes itself by predicting long-term spatiotemporal responses without the need for physical constraints or governing equations, unlike physics-informed neural networks (PINN). The core of this code is the combination of **conditional Generative Adversarial Networks (cGANs)** and **Vision Transformers (ViT)** for long-term spatiotemporal prediction across various scientific and engineering applications such as **crack propagation in composite material**, **3D reaction diffusion**, **unsteady flow** and **climate science**. The framework trains a **cGAN model** to generate synthetic data and a **Vision Transformer (ViT)** model to predict spatiotemporal dynamics over time as shown **Figure 1**. The models are designed to handle large spatial and temporal data, making them suitable for tasks involving **scientific simulations** and **real-world predictions** as shown **Figure 2**..
![CoverLetterF](https://github.com/user-attachments/assets/8d87ca89-d827-4ccf-ac5b-1c7a75bff6f9)

## Supported Scenarios (Configurations of optimized hyper paramters):
- **crack_propagation**: Used for modeling the progression of cracks in composite materials, even for configurations outside the training data distribution, demonstrating its robustness and generalization capabilities.
- **3D_reaction_diffusion**: Simulates reaction-diffusion processes in 3D environments, handling complex 3D data with high accuracy, showcasing its scalability and adaptability to high-dimensional spatiotemporal problems.
- **flow_past_cylinder**: Models fluid dynamics around objects like cylinders, including turbulent flows and Rayleigh-Bénard convection, which are critical for engineering safety and efficiency.
- **climate_science**: Simulates and forecasts climate-related data, such as sea surface temperature patterns, capturing seasonal and long-term trends, essential for climate modeling and forecasting phenomena like El Niño.
![CoverLetterFigure1](https://github.com/user-attachments/assets/9e175b01-ad90-41c3-9a13-200c75873704)

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

## Summary of optimized DynamicGPT's architecture
![Supple figure 8](https://github.com/user-attachments/assets/a0adf687-b44c-465e-807c-07906529bb97)

