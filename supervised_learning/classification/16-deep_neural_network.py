#!/usr/bin/env python3
"""
Module for Deep Neural Network class
Defines a deep neural network performing binary classification
"""
import numpy as np


class DeepNeuralNetwork:
    """
    A class that defines a deep neural
    network performing binary classification
    """

    def __init__(self, nx, layers):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        if not all(isinstance(layer, int) and layer > 0 for layer in layers):
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        layer_sizes = [nx] + layers

        for i in range(1, self.L + 1):
            self.weights['W' + str(i)] = (
                np.random.randn(layer_sizes[i], layer_sizes[i - 1]) *
                np.sqrt(2 / layer_sizes[i - 1])
            )
            self.weights['b' + str(i)] = np.zeros((layer_sizes[i], 1))
