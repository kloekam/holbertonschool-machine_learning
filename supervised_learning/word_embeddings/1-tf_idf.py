#!/usr/bin/env python3
"""TF-IDF embedding module."""
import re
import numpy as np


def tf_idf(sentences, vocab=None):
    """Create a TF-IDF embedding matrix."""

    def tokenize(sentence):
        """Tokenize a sentence into lowercase words."""
        sentence = sentence.lower()
        sentence = re.sub(r"'s\b", "", sentence)
        sentence = re.sub(r"[^a-z ]", "", sentence)
        return [w for w in sentence.split() if w]

    tokenized = [tokenize(s) for s in sentences]
    s = len(sentences)

    if vocab is None:
        word_set = set()
        for tokens in tokenized:
            word_set.update(tokens)
        features = np.array(sorted(word_set))
    else:
        features = np.array(vocab)

    f = len(features)
    feature_index = {word: i for i, word in enumerate(features)}

    embeddings = np.zeros((s, f))

    for i, tokens in enumerate(tokenized):
        if not tokens:
            continue
        tf = {}
        for word in tokens:
            if word in feature_index:
                tf[word] = tf.get(word, 0) + 1
        for word, count in tf.items():
            tf[word] = count / len(tokens)

        for word, tf_val in tf.items():
            docs_with_word = sum(
                1 for t in tokenized if word in t
            )
            idf = np.log((1 + s) / (1 + docs_with_word)) + 1
            j = feature_index[word]
            embeddings[i][j] = tf_val * idf

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms

    return embeddings, features
