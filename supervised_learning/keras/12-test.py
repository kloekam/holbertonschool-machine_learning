#!/usr/bin/env python3
"""Keras Module"""

import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """Tests a neural network"""
    results = network.evaluate(
        data,
        labels,
        verbose=verbose
    )
    return results
