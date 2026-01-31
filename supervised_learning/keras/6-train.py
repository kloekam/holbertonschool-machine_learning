#!/usr/bin/env python3
"""Keras Module"""

import tensorflow.keras as K


def train_model(network,
                data,
                labels,
                batch_size,
                epochs,
                validation_data=None,
                early_stopping=False,
                patience=0,
                verbose=True,
                shuffle=False):
    """Trains a model"""
    callbacks = []

    if early_stopping and validation_data is not None:
        early_stop = K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience
        )
        callbacks.append(early_stop)

    history = network.fit(
        data, labels,
        validation_data=validation_data,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=verbose,
        shuffle=shuffle
    )

    return history
