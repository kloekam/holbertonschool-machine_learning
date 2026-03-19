#!/usr/bin/env python3
"""DenseNet-121 implementation"""
from tensorflow import keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """Builds the DenseNet-121 architecture as described in
    Densely Connected Convolutional Networks
    """
    initializer = K.initializers.HeNormal(seed=0)

    X_input = K.Input(shape=(224, 224, 3))

    # Initial convolution layer
    nb_filters = 2 * growth_rate

    X = K.layers.BatchNormalization(axis=3)(X_input)
    X = K.layers.Activation('relu')(X)
    X = K.layers.Conv2D(nb_filters, (7, 7), strides=(2, 2), padding='same',
                        kernel_initializer=initializer)(X)

    # MaxPool: 3x3, stride 2
    X = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(X)

    # Dense Block 1 + Transition (6 layers)
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 6)
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense Block 2 + Transition (12 layers)
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 12)
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense Block 3 + Transition (24 layers)
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 24)
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense Block 4 (16 layers)
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 16)

    # Global average pooling → 1x1
    X = K.layers.AveragePooling2D((7, 7), strides=(1, 1), padding='valid')(X)

    # Fully connected: 1000 classes softmax
    X = K.layers.Dense(1000, activation='softmax',
                       kernel_initializer=initializer)(X)

    model = K.models.Model(inputs=X_input, outputs=X)

    return model
