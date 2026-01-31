#!/usr/bin/env python3
"""Keras Module"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os


def test_model(network, data, labels, verbose=True):
    """Tests a neural network"""
    results = network.evaluate(
        data,
        labels,
        verbose=verbose
    )
    return results
