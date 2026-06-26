#!/usr/bin/env python3
"""Bag of Words embedding module."""
import re
import numpy as np


def bag_of_words(sentences, vocab=None):
    """Create a bag of words embedding matrix."""
    def tokenize(sentence):
        """Lowercase, strip possessives and non-alpha characters, split."""
        sentence = sentence.lower()
        sentence = re.sub(r"'s\b", "", sentence)
        sentence = re.sub(r"[^a-z\s]", "", sentence)
        return sentence.split()

    if vocab is None:
        word_set = set()
        for s in sentences:
            word_set.update(tokenize(s))
        features = sorted(word_set)
    else:
        features = sorted(vocab)

    feature_index = {word: i for i, word in enumerate(features)}
    embeddings = np.zeros((len(sentences), len(features)), dtype=int)

    for i, sentence in enumerate(sentences):
        for word in tokenize(sentence):
            if word in feature_index:
                embeddings[i][feature_index[word]] += 1

    return embeddings, features
