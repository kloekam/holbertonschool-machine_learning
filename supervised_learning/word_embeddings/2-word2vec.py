#!/usr/bin/env python3
"""Word2Vec model training module."""
from gensim.models import Word2Vec


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """Create, build, and train a gensim Word2Vec model."""
    sg = 0 if cbow else 1
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=sg,
        epochs=epochs,
        seed=seed,
        workers=workers
    )
    return model
#!/usr/bin/env python3
"""Gensim Word2Vec to Keras Embedding layer conversion module."""
from tensorflow import keras


def gensim_to_keras(model):
    """Convert a trained gensim Word2Vec model to a Keras Embedding layer.

    Args:
        model: a trained gensim Word2Vec model.

    Returns:
        A trainable Keras Embedding layer initialized with the model weights.
    """
    weights = model.wv.vectors
    vocab_size, vector_size = weights.shape
    embedding = keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=vector_size,
        weights=[weights],
        trainable=True
    )
    return embedding
