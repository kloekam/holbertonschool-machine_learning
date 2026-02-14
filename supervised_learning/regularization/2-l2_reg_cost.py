#!/usr/bin/env python3
"""Regularization Module"""

import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    A function that calculates the
    cost of a neural network with L2 regularization
    """
    l2_costs = []
    for layer in model.layers:
        layer_l2 = tf.math.reduce_sum(
            layer.losses) if layer.losses else tf.constant(0.0)
        l2_costs.append(layer_l2)
    return tf.stack(l2_costs)
