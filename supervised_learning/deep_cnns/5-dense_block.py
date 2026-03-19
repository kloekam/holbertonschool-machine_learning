#!/usr/bin/env python3
"""Dense block implementation for DenseNet"""
from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """Builds a dense block as described in
    Densely Connected Convolutional Networks
    """
    initializer = K.initializers.HeNormal(seed=0)

    for _ in range(layers):

        # BN → ReLU → 1x1 Conv (produces 4 * growth_rate feature maps)
        X_new = K.layers.BatchNormalization(axis=3)(X)
        X_new = K.layers.Activation('relu')(X_new)
        X_new = K.layers.Conv2D(4 * growth_rate, (1, 1), padding='same',
                                kernel_initializer=initializer)(X_new)

        # BN → ReLU → 3x3 Conv (produces growth_rate feature maps)
        X_new = K.layers.BatchNormalization(axis=3)(X_new)
        X_new = K.layers.Activation('relu')(X_new)
        X_new = K.layers.Conv2D(growth_rate, (3, 3), padding='same',
                                kernel_initializer=initializer)(X_new)

        # Concatenate input with new output along channel axis
        X = K.layers.Concatenate(axis=3)([X, X_new])

        # Update filter count
        nb_filters += growth_rate

    return X, nb_filters
