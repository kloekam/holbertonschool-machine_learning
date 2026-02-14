#!/usr/bin/env python3
"""Regularization Module"""

import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    A function that calculates the
    cost of a neural network with L2 regularization
    """
    l2_loss = tf.add_n(model.losses) if model.losses else 0.0
    return cost + l2_loss
