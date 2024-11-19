# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 10:41:29 2024

@author: user
"""
import numpy as np
class DatasetLoader:
    def __init__(self, x_file='X.npy', y_file='Y.npy'):
        """
        Class to load X.npy and Y.npy files from a specific directory.
        :param x_file: The path to the X.npy file.
        :param y_file: The path to the Y.npy file.
        :param directory: The base directory where the files are located.
        """
        self.x_file = x_file
        self.y_file = y_file

    def load_data(self):
        """
        Loads X and Y numpy arrays from the specified files.
        :return: X and Y numpy arrays
        """
        X = np.load(self.x_file)
        Y = np.load(self.y_file)
        return X, Y