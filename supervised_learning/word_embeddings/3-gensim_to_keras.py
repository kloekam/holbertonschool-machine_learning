#!/usr/bin/env python3
"""Gensim Word2Vec to Keras Embedding layer conversion module."""
import tensorflow as tf


def gensim_to_keras(model):
    """Convert a trained gensim Word2Vec model to a Keras Embedding layer.

    Args:
        model: a trained gensim Word2Vec model

    Returns:
        a trainable Keras Embedding layer initialized with the model weights
    """
    weights = model.wv.vectors
    vocab_size, vector_size = weights.shape
    return tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=vector_size,
        weights=[weights],
        trainable=True
    )
