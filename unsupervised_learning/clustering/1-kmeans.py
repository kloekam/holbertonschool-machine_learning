#!/usr/bin/env python3
"""A script that performs K-means on a dataset"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """A function that performs K-means on a dataset"""

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None

    if not isinstance(k, int) or k <= 0:
        return None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape

    if k > n:
        return None, None

    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)

    C = np.random.uniform(low=min_vals, high=max_vals, size=(k, d))

    for _ in range(iterations):
        C_prev = C.copy()
        sum_cluster_points = np.zeros_like(C)
        n_cluster_points = np.zeros((k, 1))
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(distances, axis=1)

        for i in range(k):
            cluster_points = X[clss == i]

            if len(cluster_points) == 0:
                C[i] = np.random.uniform(low=min_vals, high=max_vals, size=d)
            else:
                sum_cluster_points[i] = np.sum(cluster_points, axis=0)
                n_cluster_points[i] = cluster_points.shape[0]

        non_empty_clusters = n_cluster_points.flatten() != 0
        C[non_empty_clusters] = (sum_cluster_points[non_empty_clusters]
                                 / n_cluster_points[non_empty_clusters])

        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(distances, axis=1)

        if np.array_equal(C, C_prev):
            return C, clss

    return C, clss
