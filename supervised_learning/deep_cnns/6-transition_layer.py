#!/usr/bin/env python3
"""Transition layer implementation for DenseNet"""
from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """Builds a transition layer as described in
    Densely Connected Convolutional Networks
    """
    initializer = K.initializers.HeNormal(seed=0)

    # Compute compressed filter count (DenseNet-C)
    nb_filters = int(nb_filters * compression)

    # BN → ReLU → 1x1 Conv (reduce channels by compression factor)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)
    X = K.layers.Conv2D(nb_filters, (1, 1), padding='same',
                        kernel_initializer=initializer)(X)

    # 2x2 Average Pooling with stride 2 (halve spatial dimensions)
    X = K.layers.AveragePooling2D((2, 2), strides=(2, 2), padding='same')(X)

    return X, nb_filters
