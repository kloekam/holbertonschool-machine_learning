#!/usr/bin/env python3
"""Optimization Module Tasks"""

import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """
    A function that sets up the gradient descent
    with momentum optimization algorithm
    """
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=alpha,
        momentum=beta1
    )

    return optimizer
