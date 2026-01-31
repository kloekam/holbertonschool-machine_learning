#!/usr/bin/env python3
"""Keras Module"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os


def save_model(network, filename):
    """Saves an entire model"""
    network.save(filename)
    return None


def load_model(filename):
    """Loads an entire model"""
    return keras.models.load_model(filename)
