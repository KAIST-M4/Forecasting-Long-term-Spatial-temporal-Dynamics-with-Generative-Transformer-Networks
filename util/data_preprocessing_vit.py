# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 09:43:44 2024

@author: user
"""

import numpy as np
import tensorflow as tf

class DataPreprocessor:
    def __init__(self, batch_size, max_patches, patch_shape=(2, 2, 64)):
        self.batch_size = batch_size
        self.max_patches = max_patches
        self.patch_shape = patch_shape  # Shape of the patches (height, width, channels)
    
    # Function to create dummy data patches
    def generate_dummy_data(self):
        patches_A = [np.random.rand(1, *self.patch_shape) for _ in range(self.batch_size)]  # (batch_size, 1, 16, 16, 128)
        patches_B = [np.random.rand(3, *self.patch_shape) for _ in range(self.batch_size)]  # (batch_size, 3, 16, 16, 128)
        patches_C = [np.random.rand(5, *self.patch_shape) for _ in range(self.batch_size)]  # (batch_size, 5, 16, 16, 128)
        
        return patches_A, patches_B, patches_C
    
    # Data preprocessing function
    def preprocess_data(self, patches):
        batch_size = len(patches)
        input_data = np.zeros((batch_size, self.max_patches, np.prod(self.patch_shape)))  # Initialize input data array
        attention_mask = np.zeros((batch_size, self.max_patches))  # Initialize attention mask array

        for i, patch_seq in enumerate(patches):
            for j, patch in enumerate(patch_seq):
                input_data[i, j, :] = patch.flatten()  # Flatten each patch and store in input_data
                attention_mask[i, j] = 1  # Mark valid data locations as 1 in attention mask

        # Convert 2D mask to 3D mask
        attention_mask = tf.expand_dims(attention_mask, axis=1)  # (batch_size, 1, max_patches)
        attention_mask = tf.matmul(attention_mask, attention_mask, transpose_b=True)  # (batch_size, max_patches, max_patches)
        
        return input_data, attention_mask
    
    # Method to prepare the dataset
    def prepare_dataset(self):
        # Generate dummy patches
        patches_A, patches_B, patches_C = self.generate_dummy_data()

        # Preprocess each dataset
        input_A, mask_A = self.preprocess_data(patches_A)
        input_B, mask_B = self.preprocess_data(patches_B)
        input_C, mask_C = self.preprocess_data(patches_C)

        # Combine datasets
        inputs = np.concatenate([input_A, input_B, input_C], axis=0)
        masks = np.concatenate([mask_A, mask_B, mask_C], axis=0)

        # Generate labels (target patches)
        labels_A = input_A[:, 0, :]
        labels_B = input_B[:, 0, :]
        labels_C = input_C[:, 0, :]
        labels = np.concatenate([labels_A, labels_B, labels_C], axis=0)

        # Convert to TensorFlow dataset
        dataset = tf.data.Dataset.from_tensor_slices(((inputs, masks), labels))
        # Shuffle and batch the dataset
        dataset = dataset.batch(self.batch_size).shuffle(buffer_size=10)

        return dataset
