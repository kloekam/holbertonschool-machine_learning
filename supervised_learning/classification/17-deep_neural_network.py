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

        for layer in layers:
            if not isinstance(layer, int) or layer <= 0:
                raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        layer_sizes = [nx] + layers

        for i in range(1, self.__L + 1):
            self.__weights['W' + str(i)] = (
                np.random.randn(layer_sizes[i], layer_sizes[i - 1]) *
                np.sqrt(2 / layer_sizes[i - 1])
            )
            self.__weights['b' + str(i)] = np.zeros((layer_sizes[i], 1))

    @property
    def L(self):
        """Getter for L"""
        return self.__L

    @property
    def cache(self):
        """Getter for cache"""
        return self.__cache

    @property
    def weights(self):
        """Getter for weights"""
        return self.__weights
