#!/usr/bin/env python3
"""Optimization Module Tasks"""

import tensorflow.keras as K


def create_mini_batches(X, Y, batch_size):
    """
    A function that creates mini-batches to be used for training
    a neural network using mini-batch gradient descent
    """
    m = X.shape[0]

    X_shuffled, Y_shuffled = __import__('2-shuffle_data').shuffle_data

    mini_batches = []

    num_complete_batches = m // batch_size

    for k in range(num_complete_batches):
        start = k * batch_size
        end = (k + 1) * batch_size
        X_batch = X_shuffled[start:end]
        Y_batch = Y_shuffled[start:end]
        mini_batches.append((X_batch, Y_batch))

    if m % batch_size != 0:
        start = num_complete_batches * batch_size
        X_batch = X_shuffled[start:]
        Y_batch = Y_shuffled[start:]
        mini_batches.append((X_batch, Y_batch))

    return mini_batches
