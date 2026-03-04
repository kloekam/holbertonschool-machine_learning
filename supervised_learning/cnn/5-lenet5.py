#!/usr/bin/env python3
"""
Convolutional Neural Networks Module
"""

from tensorflow import keras as K


def lenet5(X):
    """
    A function  that builds a modified version
    of the LeNet-5 architecture using keras
    """
    initializer = K.initializers.HeNormal(seed=0)

    conv1 = K.layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        padding='same',
        kernel_initializer=initializer,
        activation='relu'
    )(X)

    pool1 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(conv1)

    conv2 = K.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding='valid',
        kernel_initializer=initializer,
        activation='relu'
    )(pool1)

    pool2 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(conv2)

    flatten = K.layers.Flatten()(pool2)

    fc1 = K.layers.Dense(
        120,
        kernel_initializer=initializer,
        activation='relu'
    )(flatten)

    fc2 = K.layers.Dense(
        84,
        kernel_initializer=initializer,
        activation='relu'
    )(fc1)

    output = K.layers.Dense(
        10,
        kernel_initializer=initializer,
        activation='softmax'
    )(fc2)

    model = K.models.Model(inputs=X, outputs=output)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return model
