#!/usr/bin/env python3
"""ResNet-50 implementation"""
from tensorflow import keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """Builds the ResNet-50 architecture as described in
    Deep Residual Learning for Image Recognition (2015)
    """
    initializer = K.initializers.HeNormal(seed=0)

    X_input = K.Input(shape=(224, 224, 3))

    # Stage 1: Conv1 - 7x7, 64 filters, stride 2
    X = K.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same',
                        kernel_initializer=initializer)(X_input)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # MaxPool: 3x3, stride 2
    X = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(X)

    # Stage 2: Conv2_x - output 56x56x256
    X = projection_block(X, [64, 64, 256], s=1)
    X = identity_block(X, [64, 64, 256])
    X = identity_block(X, [64, 64, 256])

    # Stage 3: Conv3_x - output 28x28x512
    X = projection_block(X, [128, 128, 512], s=2)
    X = identity_block(X, [128, 128, 512])
    X = identity_block(X, [128, 128, 512])
    X = identity_block(X, [128, 128, 512])

    # Stage 4: Conv4_x - output 14x14x1024
    X = projection_block(X, [256, 256, 1024], s=2)
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])

    # Stage 5: Conv5_x - output 7x7x2048
    X = projection_block(X, [512, 512, 2048], s=2)
    X = identity_block(X, [512, 512, 2048])
    X = identity_block(X, [512, 512, 2048])

    # Average pooling: 7x7
    X = K.layers.AveragePooling2D((7, 7), strides=(1, 1), padding='valid')(X)

    # Fully connected output: 1000 classes
    X = K.layers.Dense(1000, activation='softmax',
                       kernel_initializer=initializer)(X)

    model = K.models.Model(inputs=X_input, outputs=X)

    return model
