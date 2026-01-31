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
                learning_rate_decay=False,
                alpha=0.1,
                decay_rate=1,
                save_best=False,
                filepath=None,
                verbose=True,
                shuffle=False):
    """Trains a model"""
    callbacks = []

    if learning_rate_decay and validation_data is not None:
        def lr_schedule(epoch):
            return alpha / (1 + decay_rate * epoch)

        lr_scheduler = K.callbacks.LearningRateScheduler(
            schedule=lr_schedule,
            verbose=1
        )
        callbacks.append(lr_scheduler)

    if save_best and validation_data is not None and filepath is not None:
        checkpoint = K.callbacks.ModelCheckpoint(
            filepath=filepath,
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=0
        )
        callbacks.append(checkpoint)

    history = network.fit(
        x=data,
        y=labels,
        validation_data=validation_data,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=verbose,
        shuffle=shuffle
    )

    return history
