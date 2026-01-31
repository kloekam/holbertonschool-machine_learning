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

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network
        """
        self.__cache['A0'] = X

        for i in range(1, self.__L + 1):
            Z = (np.matmul(self.__weights['W' + str(i)],
                 self.__cache['A' + str(i - 1)]) +
                 self.__weights['b' + str(i)])
            self.__cache['A' + str(i)] = 1 / (1 + np.exp(-Z))

        return self.__cache['A' + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        """
        m = Y.shape[1]
        cost = (-1 / m) * np.sum(Y * np.log(A) +
                                 (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network's predictions
        """
        A, cache = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost
