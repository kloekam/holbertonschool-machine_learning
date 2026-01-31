#!/usr/bin/env python3
"""Keras Module"""


import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os


def save_weights(network, filename, save_format='keras'):
    """"Saves a model's weights"""
    network.save_weights(filename, save_format=save_format)

    return None


def load_weights(network, filename):
    """Loads a model's weights"""
    network.load_weights(filename)

    return None
