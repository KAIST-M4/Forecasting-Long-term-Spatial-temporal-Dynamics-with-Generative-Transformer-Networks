# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 09:46:17 2024

@author: user
"""
import tensorflow as tf

class ModelCompiler:
    def __init__(self, model, loss_fn=None, optimizer=None):
        """
        Initializes the ModelCompiler class with a model, loss function, and optimizer.
        :param model: The model to be compiled (e.g., SpatiotemporalViT).
        :param loss_fn: The loss function to use for the model. Defaults to Mean Squared Error if None.
        :param optimizer: The optimizer to use for the model. Defaults to Adam if None.
        """
        self.model = model
        self.loss_fn = loss_fn if loss_fn is not None else tf.keras.losses.MeanSquaredError()  # Default to MSE
        self.optimizer = optimizer if optimizer is not None else tf.keras.optimizers.Adam()  # Default to Adam
    
    def compile_model(self):
        """
        Compiles the model with the given loss function and optimizer.
        """
        self.model.compile(optimizer=self.optimizer, loss=self.loss_fn)
        print("Model has been compiled with optimizer: {} and loss function: {}".format(
            self.optimizer.__class__.__name__, self.loss_fn.__class__.__name__))
        
    def get_loss_function(self):
        """
        Returns the loss function.
        """
        return self.loss_fn
    
    def get_optimizer(self):
        """
        Returns the optimizer.
        """
        return self.optimizer