#!/usr/bin/env python3
"""Optimization Module Tasks"""

import numpy as np


def shuffle_data(X, Y):
    """
    A function that shuffles the data points
    into 2 matrices the same way
    """

    permutation = np.random.permutation(X.shape[0])

    return X[permutation], Y[permutation]
