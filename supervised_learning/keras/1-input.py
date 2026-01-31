#!/usr/bin/env python3
"""Keras Module"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Builds a neural network with the Keras library"""
    inputs = tf.keras.Input(shape=(nx,))
    x = inputs

    for i in range(len(layers)):
        x = tf.keras.layers.Dense(
            units=layers[i],
            activation=activations[i],
            kernel_regularizer=tf.keras.regularizers.L2(lambtha)
        )(x)

        if i != len(layers) - 1 and keep_prob is not None:
            x = tf.keras.layers.Dropout(1-keep_prob)(x)

    model = keras.Model(inputs, x)

    return model
