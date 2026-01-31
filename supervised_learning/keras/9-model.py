#!/usr/bin/env python3
"""Keras Module"""

import tensorflow.keras as K


def save_model(network, filename):
    """Saves an entire model"""
    network.save(filename)
    return None


def load_model(filename):
    """Loads an entire model"""
    return keras.models.load_model(filename)
