#!/usr/bin/env python3
"""Optimization Module Tasks"""

import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """
    A function that sets up the
    RMSProp optimization algorithm
    """
    optimizer = tf.keras.optimizers.RMSProp(
        learning_rate=alpha,
        momentum=beta2,
        epsilon=epsilon
    )

    return optimizer
