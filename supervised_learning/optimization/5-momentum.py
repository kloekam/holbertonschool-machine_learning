#!/usr/bin/env python3
"""Optimization Module Tasks"""

import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    A function  that updates a variable using
    the gradient descent with momentum
    optimization algorithm
    """
    v_new = beta1 * v + (1 - beta1) * grad
    var_updated = var - alpha * v_new

    return v_new, var_updated
