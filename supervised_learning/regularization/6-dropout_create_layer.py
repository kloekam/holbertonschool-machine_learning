#!/usr/bin/env python3
"""Regularization Module"""

import numpy as np
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
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    dense = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer
    )

    A = dense(prev)

    dropout = tf.keras.layers.Dropout(rate=1 - keep_prob)

    return dropout(A, training=training)
