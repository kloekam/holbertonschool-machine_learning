#!/usr/bin/env python3
"""Gensim Word2Vec to Keras Embedding layer conversion module."""
import tensorflow as tf


def gensim_to_keras(model):
    """Convert a trained gensim Word2Vec model to a Keras Embedding layer."""
    weights = model.wv.vectors
    vocab_size, vector_size = weights.shape
    embedding = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=vector_size,
        weights=[weights],
        trainable=True
    )
    return embedding
