#!/usr/bin/env python3
"""Projection block implementation for ResNet"""
from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """Builds a projection block as described
    in Deep Residual Learning for Image Recognition (2015)
    """
    F11, F3, F12 = filters
    initializer = K.initializers.HeNormal(seed=0)

    # First 1x1 convolution (with stride s)
    X = K.layers.Conv2D(F11, (1, 1), strides=(s, s), padding='same',
                        kernel_initializer=initializer)(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # 3x3 convolution
    X = K.layers.Conv2D(F3, (3, 3), padding='same',
                        kernel_initializer=initializer)(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # Second 1x1 convolution (no activation yet)
    X = K.layers.Conv2D(F12, (1, 1), padding='same',
                        kernel_initializer=initializer)(X)
    X = K.layers.BatchNormalization(axis=3)(X)

    # 1x1 convolution to match dimensions (stride s, F12 filters)
    shortcut = K.layers.Conv2D(F12, (1, 1), strides=(s, s), padding='same',
                               kernel_initializer=initializer)(A_prev)
    shortcut = K.layers.BatchNormalization(axis=3)(shortcut)

    X = K.layers.Add()([X, shortcut])
    X = K.layers.Activation('relu')(X)

    return X
