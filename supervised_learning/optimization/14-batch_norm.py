#!/usr/bin/env python3
"""Optimization Module Tasks"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    A function that creates a batch normalization
    layer for a neural network in tensorflow
    """
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    dense = tf.keras.layers.Dense(n, kernel_initializer=initializer)
    z = dense(prev)

    gamma = tf.Variable(tf.ones([n]), trainable=True)
    beta = tf.Variable(tf.zeros([n]), trainable=True)

    mean, variance = tf.nn.moments(z, axes=[0])

    epsilon = 1e-7
    z_norm = tf.nn.batch_normalization(
        z,
        mean,
        variance,
        offset=beta,
        scale=gamma,
        variance_epsilon=epsilon
    )

    if activation is not None:
        return activation(z_norm)
    return z_norm
