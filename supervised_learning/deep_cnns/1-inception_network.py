#!/usr/bin/env python3
"""Inception Network implementation"""
from tensorflow import keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """Builds the inception network as described
    in Going Deeper with Convolutions (2014)
    """
    X = K.Input(shape=(224, 224, 3))

    # Conv1: 7x7, stride 2
    X1 = K.layers.Conv2D(64, (7, 7), strides=(2, 2),
                         padding='same', activation='relu')(X)

    # MaxPool1: 3x3, stride 2
    X1 = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(X1)

    # Conv2: 1x1 reduction
    X1 = K.layers.Conv2D(64, (1, 1), padding='same', activation='relu')(X1)

    # Conv2: 3x3
    X1 = K.layers.Conv2D(192, (3, 3), padding='same', activation='relu')(X1)

    # MaxPool2: 3x3, stride 2
    X1 = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(X1)

    # Inception 3a
    X1 = inception_block(X1, [64, 96, 128, 16, 32, 32])

    # Inception 3b
    X1 = inception_block(X1, [128, 128, 192, 32, 96, 64])

    # MaxPool3: 3x3, stride 2
    X1 = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(X1)

    # Inception 4a
    X1 = inception_block(X1, [192, 96, 208, 16, 48, 64])

    # Inception 4b
    X1 = inception_block(X1, [160, 112, 224, 24, 64, 64])

    # Inception 4c
    X1 = inception_block(X1, [128, 128, 256, 24, 64, 64])

    # Inception 4d
    X1 = inception_block(X1, [112, 144, 288, 32, 64, 64])

    # Inception 4e
    X1 = inception_block(X1, [256, 160, 320, 32, 128, 128])

    # MaxPool4: 3x3, stride 2
    X1 = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(X1)

    # Inception 5a
    X1 = inception_block(X1, [256, 160, 320, 32, 128, 128])

    # Inception 5b
    X1 = inception_block(X1, [384, 192, 384, 48, 128, 128])

    # AvgPool: 7x7, stride 1
    X1 = K.layers.AveragePooling2D((7, 7), strides=(1, 1), padding='valid')(X1)

    # Dropout 40%
    X1 = K.layers.Dropout(0.4)(X1)

    # Softmax output (1000 classes)
    X1 = K.layers.Dense(1000, activation='softmax')(X1)

    model = K.models.Model(inputs=X, outputs=X1)

    return model
