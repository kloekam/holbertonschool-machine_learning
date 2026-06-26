#!/usr/bin/env python3
"""Word2Vec model training module."""
import os
import sys
if 'PYTHONHASHSEED' not in os.environ:
    os.environ['PYTHONHASHSEED'] = '0'
    os.execv(sys.executable, [sys.executable] + sys.argv)
import gensim


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """
    Create, build, and train a gensim Word2Vec model.
    Returns the trained model
    """

    sg = 0 if cbow else 1
    model = gensim.models.Word2Vec(
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
  
