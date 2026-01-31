#!/usr/bin/env python3
"""Keras Module"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os


def predict(network, data, verbose=False):
    """Makes a prediction using a neural network"""
    predictions = network.predict(
        data,
        verbose=verbose
    )
    return predictions
