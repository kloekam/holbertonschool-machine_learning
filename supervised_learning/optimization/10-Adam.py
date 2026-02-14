#!/usr/bin/env python3
"""Optimization Module Tasks"""

import tensorflow as tf


def create_Adam_op(alpha, beta1, beta2, epsilon):
    """
    A function hat sets up the RMSProp
    optimization algorithm in TensorFlow
    """
    return tf.optimizers.Adam(
        learning_rate=alpha,
        beta_1=beta1,
        beta_2=beta2,
        epsilon=epsilon
    )
