#!/usr/bin/env python3
"""Optimization Module Tasks"""

import numpy as np


def normalization_constants(X):
    """
    A function that calculates the normalization
    (standardization) constants of a matrix
    """
    m = np.mean(X, axis=0)
    s = np.std(X, axis=0)

    return m, s
