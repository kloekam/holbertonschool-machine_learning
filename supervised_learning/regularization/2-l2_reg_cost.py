#!/usr/bin/env python3
"""Regularization Module"""

import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    A function that calculates the
    cost of a neural network with L2 regularization
    """
    if model.losses:
        return cost + tf.add_n(model.losses)
    return cost
