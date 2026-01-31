#!/usr/bin/env python3
"""Keras Module"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os


def one_hot(labels, classes=None):
    """Converts a label vector into a one-hot matrix"""

    return tf.keras.utils.to_categorical(labels, classes)
