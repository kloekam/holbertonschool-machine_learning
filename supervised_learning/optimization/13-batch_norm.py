#!/usr/bin/env python3
"""Optimization Module Tasks"""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    A function that normalizes an unactivated output of
    a neural network using batch normalization
    """
    mean = np.mean(Z, axis=0)
    variance = np.var(Z, axis=0)
    Z_normalized = (Z - mean) / np.sqrt(
        variance + epsilon
    )
    Z_norm = gamma * Z_normalized + beta
    return Z_norm
