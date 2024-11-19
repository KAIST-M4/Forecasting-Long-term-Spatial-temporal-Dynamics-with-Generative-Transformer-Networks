"""
MNet-based Generator Class for Spatiotemporal Prediction

This module defines a generator network architecture based on an M-Net structure. The generator is designed for 
spatiotemporal prediction tasks, where it learns to generate data (e.g., images or feature maps) that follows a 
spatiotemporal distribution. The architecture includes downsampling and upsampling blocks, as well as a latent 
space sampled via a Variational Autoencoder (VAE).

Key Components:
- `MNetDownsample`: A downsampling layer that uses parallel convolutions with different kernel sizes (2, 4, and 8).
- `CustomUpsample`: An upsampling layer that uses transposed convolutions with skip connections and dropout.
- `MNetGenerator`: A function that ties together the encoder and decoder with VAE latent space, forming the full generator model.

The generator is typically used in Generative Adversarial Networks (GANs) for data generation tasks, such as image 
generation or spatiotemporal sequence prediction.

"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers, Input

# Define the MNetDownsample class
class MNetDownsample(layers.Layer):
    def __init__(self, filters, target_output_shape=None, apply_batchnorm=True):
        super(MNetDownsample, self).__init__()
        initializer = tf.random_normal_initializer(0., 0.02)
        
        # Parallel paths
        self.path2 = layers.Conv2D(filters, kernel_size=2, strides=2, padding='same', kernel_initializer=initializer, use_bias=False)
        self.path4 = layers.Conv2D(filters, kernel_size=4, strides=2, padding='same', kernel_initializer=initializer, use_bias=False)
        self.path8 = layers.Conv2D(filters, kernel_size=8, strides=2, padding='same', kernel_initializer=initializer, use_bias=False)
        
        # Batch Normalization
        self.batchnorm = layers.BatchNormalization() if apply_batchnorm else None
        
        # 1x1 conv to merge paths
        self.merge_conv = layers.Conv2D(filters, (1, 1), padding='same', kernel_initializer=initializer)
        self.target_output_shape = target_output_shape

    def call(self, x):
        x2 = self.path2(x)
        x4 = self.path4(x)
        x8 = self.path8(x)
        
        # Combine the outputs
        combined = layers.Add()([x2, x4, x8])
        combined = self.merge_conv(combined)
        
        if self.batchnorm:
            combined = self.batchnorm(combined)
        combined = layers.LeakyReLU()(combined)
        
        if self.target_output_shape:
            combined = tf.image.resize(combined, self.target_output_shape)
        
        return combined

# Dummy data and MNetGenerator for testing
def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=tf.shape(z_mean))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class CustomUpsample(layers.Layer):
    def __init__(self, filters, size, custom_output_shape=None, apply_dropout=False):
        super(CustomUpsample, self).__init__()
        initializer = tf.random_normal_initializer(0., 0.02)
        self.custom_output_shape = custom_output_shape
        self.up_conv = layers.Conv2DTranspose(filters, size, strides=(2, 2), padding='same', kernel_initializer=initializer, use_bias=False)
        self.batchnorm = layers.BatchNormalization()
        self.dropout = layers.Dropout(0.5) if apply_dropout else None

    def call(self, x, skip=None):
        x = self.up_conv(x)
        x = self.batchnorm(x)
        if self.dropout:
            x = self.dropout(x)
        if skip is not None:
            x = tf.image.resize(x, (skip.shape[1], skip.shape[2]))
            x = layers.Concatenate()([x, skip])
        elif self.custom_output_shape is not None:
            x = tf.image.resize(x, self.custom_output_shape)
        return x

def MNetGenerator(input_shape, depth, feature_maps, upsample_configs, downsample_configs):
    inputs = Input(shape=input_shape)
    x = inputs
    down_layers = []

    # Encoder (M-Net based downsampling)
    for i in range(depth):
        filters = feature_maps[i]
        target_output_shape = downsample_configs[i] if downsample_configs else None
        x = MNetDownsample(filters, target_output_shape=target_output_shape)(x)
        down_layers.append(x)

    # VAE latent space
    z_mean = layers.Conv2D(feature_maps[-1], (1, 1), padding='same')(x)
    z_log_var = layers.Conv2D(feature_maps[-1], (1, 1), padding='same')(x)
    z = layers.Lambda(sampling, name = 'vae_latent_space')([z_mean, z_log_var])

    # Decoder (upsampling)
    for i in range(len(upsample_configs)):
        filters, size, custom_output_shape = upsample_configs[i]
        z = CustomUpsample(filters, size, custom_output_shape=custom_output_shape)(z, down_layers[-(i + 1)])

    # Final output layer
    outputs = layers.Conv2DTranspose(1, (4, 4), strides=2, padding='same', activation='elu')(z)

    model = Model(inputs, outputs)
    return model
