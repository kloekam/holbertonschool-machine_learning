#!/usr/bin/env python3
"""Identity block implementation for ResNet"""
from tensorflow import keras as K


def identity_block(A_prev, filters):
    """Builds an identity block as described
    in Deep Residual Learning for Image Recognition (2015)
    """
    F11, F3, F12 = filters
    initializer = K.initializers.HeNormal(seed=0)

    # First 1x1 convolution
    X = K.layers.Conv2D(F11, (1, 1), padding='same',
                        kernel_initializer=initializer)(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # 3x3 convolution
    X = K.layers.Conv2D(F3, (3, 3), padding='same',
                        kernel_initializer=initializer)(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # Second 1x1 convolution
    X = K.layers.Conv2D(F12, (1, 1), padding='same',
                        kernel_initializer=initializer)(X)
    X = K.layers.BatchNormalization(axis=3)(X)

    # Add shortcut (identity) connection
    X = K.layers.Add()([X, A_prev])

    # Final ReLU activation
    X = K.layers.Activation('relu')(X)

    return X
