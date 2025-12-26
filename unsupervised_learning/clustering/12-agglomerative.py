#!/usr/bin/env python3
"""A script that performs agglomerative clustering on a dataset"""

from curses.ascii import alt
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """A function that performs agglomerative clustering on a dataset"""

    dendrogram = scipy.cluster.hierarchy.linkage(X, method='ward')
    scipy.cluster.hierarchy.dendrogram(dendrogram, color_threshold=dist)

    plt.show()

    return scipy.cluster.hierarchy.fcluster(dendrogram,
                                            dist,
                                            criterion='distance'
                                            )
