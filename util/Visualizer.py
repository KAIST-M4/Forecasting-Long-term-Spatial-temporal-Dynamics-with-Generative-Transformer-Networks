# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 13:31:49 2024

@author: user
"""
import numpy as np
import matplotlib.pyplot as plt

class Visualizer:
    def __init__(self, pre, X, threshold=48, cmap='seismic', vmax=38, nan_value=50):
        """
        Initializes the visualizer with the necessary parameters.

        Parameters:
        - pre: The predicted data to be visualized.
        - X: The true data (ground truth).
        - threshold: The value above which data will be set to NaN.
        - cmap: The colormap to be used for the images.
        - vmax: The maximum value for color mapping.
        - nan_value: The value to replace NaN values in the data.
        """
        self.pre = pre
        self.X = X
        self.threshold = threshold
        self.cmap = cmap
        self.vmax = vmax
        self.nan_value = nan_value

    def visualize(self):
        """
        Visualizes the true data, predicted data, and the difference between them.
        """
        # Create subplots
        fig, ax = plt.subplots(3, 12, figsize=(5 * 12, 12), dpi=300)
        fig.patch.set_facecolor('k')

        for j in range(12):
            for i in range(3):
                ax[i, j].set_facecolor('black')
                for spine in ax[i, j].spines.values():
                    spine.set_edgecolor('white')

                ax[i, j].tick_params(axis='x', colors='white')
                ax[i, j].tick_params(axis='y', colors='white')
                ax[i, j].xaxis.label.set_color('white')
                ax[i, j].yaxis.label.set_color('white')
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])

            # Apply thresholding to remove values above the threshold (set to NaN)
            X_filtered = self.X[j, :, :, 0]
            X_filtered[X_filtered >= self.threshold] = np.nan

            pre_filtered = self.pre[j, :, :, 0]
            pre_filtered[pre_filtered >= self.threshold] = np.nan

            # Replace NaN values with a placeholder value (nan_value)
            t1 = np.array(X_filtered)
            t1[np.isnan(t1)] = self.nan_value

            p1 = np.array(pre_filtered)
            p1[np.isnan(p1)] = self.nan_value

            # Plot the true data, predicted data, and the absolute difference
            ax[0, j].imshow(t1, cmap=self.cmap, vmin=np.min(t1), vmax=self.vmax)
            ax[1, j].imshow(p1, cmap=self.cmap, vmin=np.min(p1), vmax=self.vmax)
            ax[2, j].imshow(np.abs(t1 - p1), cmap=self.cmap, vmin=np.min(np.abs(t1 - p1)), vmax=self.vmax)

        # Show the plot
        plt.tight_layout()
        plt.show()