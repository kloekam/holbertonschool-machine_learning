#!/usr/bin/env python3
"""Optimization Module Tasks"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    A function that creates a batch normalization
    layer for a neural network in tensorflow
    """
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    Z = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=initializer,
        trainable=True
    )(prev)

    A = tf.keras.layers.BatchNormalization(
        epsilon=1e-7,
        gamma_initializer=tf.keras.initializers.Ones(),
        beta_initializer=tf.keras.initializers.Zeros()
    )(Z)

    return activation(A)
