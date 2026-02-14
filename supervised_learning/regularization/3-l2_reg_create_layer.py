#!/usr/bin/env python3
"""Regularization Module"""

import numpy as np
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    A function that creates a neural network
    layer in tensorFlow that includes L2 regularization
    """
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer,
        kernel_regularizer=tf.keras.regularizers.l2(lambtha)
    )

    return layer(prev)
