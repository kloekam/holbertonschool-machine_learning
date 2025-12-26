#!/usr/bin/env python3
"""A script that initializes cluster centroids for K-means"""
import numpy as np


def initialize(X, k):
    """A function that initializes cluster centroids for K-means"""

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None

    if not isinstance(k, int) or k <= 0:
        return None

    n, d = X.shape

    if k > n:
        return None

    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)

    centroids = np.random.uniform(low=min_vals, high=max_vals, size=(k, d))

    return centroids
