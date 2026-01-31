#!/usr/bin/env python3
"""Keras Module"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os


def train_model(
        network, data, labels, batch_size, epochs,
        verbose=True, shuffle=False):
    """Trains a model"""
    history = network.fit(
        data, labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle
    )
    return history
