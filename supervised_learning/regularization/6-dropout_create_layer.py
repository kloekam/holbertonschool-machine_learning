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
    dense = tf.keras.layers.Dense(
        units=n,
        activation=activation
    )(prev)

    dropout = tf.keras.layers.Dropout(
        rate=1.0 - keep_prob,
    )(dense,
      training=training)

    return dropout
