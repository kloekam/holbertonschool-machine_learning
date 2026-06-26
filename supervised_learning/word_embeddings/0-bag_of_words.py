#!/usr/bin/env python3
"""Bag of words embedding module."""
import re
import numpy as np


def bag_of_words(sentences, vocab=None):
    """Create a bag of words embedding matrix."""

    def tokenize(sentence):
        """Tokenize a sentence into lowercase words."""
        sentence = sentence.lower()
        sentence = re.sub(r"'s\b", "", sentence)
        sentence = re.sub(r"[^a-z ]", "", sentence)
        return [w for w in sentence.split() if w]

    if vocab is None:
        word_set = set()
        for sentence in sentences:
            word_set.update(tokenize(sentence))
        features = np.array(sorted(word_set))
    else:
        features = np.array(vocab)

    feature_index = {word: i for i, word in enumerate(features)}

    embeddings = np.zeros((len(sentences), len(features)), dtype=int)
    for i, sentence in enumerate(sentences):
        for word in tokenize(sentence):
            if word in feature_index:
                embeddings[i][feature_index[word]] += 1

    return embeddings, features
