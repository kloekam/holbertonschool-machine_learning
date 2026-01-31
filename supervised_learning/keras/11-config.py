#!/usr/bin/env python3
"""Keras Module"""

import tensorflow.keras as K


def save_config(network, filename):
    """Saves a model's configuration in JSON format"""
    json_config = network.to_json()
    with open(filename, 'w') as json_file:
        json_file.write(json_config)
    return None


def load_config(filename):
    """Loads a model's configuration in JSON format"""
    with open(filename, 'r') as json_file:
        json_config = json_file.read()
    return keras.models.model_from_json(json_config)
