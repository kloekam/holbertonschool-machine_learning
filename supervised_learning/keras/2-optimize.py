#!/usr/bin/env python3
"""Keras Module"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
from keras.optimizers import Adam


def optimize_model(network, alpha, beta1, beta2):
    """
    Sets up Adam optimization for a keras model
    with categorical crossentropy loss and accuracy metrics
    """
    cust_adam = Adam(learning_rate=alpha, beta_1=beta1, beta_2=beta2)
    network.compile(
        optimizer=cust_adam,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return None
