#!/usr/bin/env python3
"""Keras Module"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Builds a neural network with the Keras library"""
    model = K.Sequential()
    for i in range(1, len(layers)):
        model.add(K.layers.Dense(
            units=layers[i],
            activation=activations[i],
            kernel_regularizer=K.regularizers.L2(lambtha),
            input_shape=(nx,)
        ))

        if i != len(layers) - 1 and keep_prob is not None:
            model.add(K.layers.Dropout(1-keep_prob))

    return model
