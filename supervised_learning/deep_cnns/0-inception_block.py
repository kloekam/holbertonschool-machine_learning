#!/usr/bin/env python3
"""Inception block implementation"""
from tensorflow import keras as K


def inception_block(A_prev, filters):
    """Builds an inception block as described
    in Going Deeper with Convolutions (2014)
    """
    F1, F3R, F3, F5R, F5, FPP = filters

    # Branch 1: 1x1 convolution
    conv1x1 = K.layers.Conv2D(F1, (1, 1), padding='same',
                              activation='relu')(A_prev)

    # Branch 2: 1x1 reduction then 3x3 convolution
    conv3x3_reduce = K.layers.Conv2D(F3R, (1, 1), padding='same',
                                     activation='relu')(A_prev)
    conv3x3 = K.layers.Conv2D(F3, (3, 3), padding='same',
                              activation='relu')(conv3x3_reduce)

    # Branch 3: 1x1 reduction then 5x5 convolution
    conv5x5_reduce = K.layers.Conv2D(F5R, (1, 1), padding='same',
                                     activation='relu')(A_prev)
    conv5x5 = K.layers.Conv2D(F5, (5, 5), padding='same',
                              activation='relu')(conv5x5_reduce)

    # Branch 4: 3x3 max pooling then 1x1 convolution
    max_pool = K.layers.MaxPooling2D((3, 3), strides=(1, 1),
                                     padding='same')(A_prev)
    conv_pool = K.layers.Conv2D(FPP, (1, 1), padding='same',
                                activation='relu')(max_pool)

    # Concatenate all branches along the channel axis
    output = K.layers.Concatenate(axis=-1)([conv1x1, conv3x3,
                                            conv5x5, conv_pool])

    return output
