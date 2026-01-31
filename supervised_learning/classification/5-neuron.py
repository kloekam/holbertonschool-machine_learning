#!/usr/bin/env python3
"""
A script that defines a single neuron performing
binary classification and calculates one
pass of gradient descent on the neuron
"""


import numpy as np


class Neuron:
    """A class that defines a single neuron performing binary classification"""
    def __init__(self, nx):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A

    def forward_prop(self, X):
        """A function that calculates the forward propagation of the neuron"""
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """
        A function that calculates the cost of the model
        using logistic regression
        """
        m = Y.shape[1]
        log_loss = -1/m*np.sum(Y * np.log(A) \
                               + (1-Y)*(np.log(1.0000001 - A)))
        return log_loss

    def evaluate(self, X, Y):
        """A function that evaluates the neuron’s predictions"""
        self.__A = self.forward_prop(X)
        cost = self.cost(Y, self.__A)
        result = np.where(self.__A >= 0.5, 1, 0)
        return result, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """A function calculates one pass of gradient descent on the neuron"""
        m = Y.shape[1]
        dW = (1/m) * np.matmul((A - Y), X.T)
        db = (1/m) * np.sum(A - Y)
        self.__W = self.__W - alpha * dW
        self.__b = self.__b - alpha * db
