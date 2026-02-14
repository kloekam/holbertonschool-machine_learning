#!/usr/bin/env python3
"""Regularization Module"""

import tensorflow as tf


def dropout_create_layer(prev,
                         n,
                         activation,
                         keep_prob,
                         training=True):
    """
    A function  that creates a layer of a
    neural network using dropout
    """
    init = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_avg')

    dense = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=init
    )

    x = dense(prev)

    if training:
        x = tf.nn.dropout(x, rate=1 - keep_prob)

    return x
