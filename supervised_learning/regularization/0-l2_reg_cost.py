#!/usr/bin/env python3
"""Regularization Module"""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    A function that calculates the cost of
    a neural network with L2 regularization
    """
    l2_sum = 0
    for i in range(1, L + 1):
        W = weights["W" + str(i)]
        l2_sum += np.sum(np.square(W))

    l2_cost = lambtha * l2_sum / (2 * m)
    return cost + l2_cost
