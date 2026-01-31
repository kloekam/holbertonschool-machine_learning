#!/usr/bin/env python3
"""Keras Module"""

import tensorflow.keras as K


def one_hot(labels, classes=None):
    """Converts a label vector into a one-hot matrix"""

    return tf.keras.utils.to_categorical(labels, classes)
