#!/usr/bin/env python3
"""TF-IDF embedding module."""
import re
import numpy as np


def tf_idf(sentences, vocab=None):
    """Create a TF-IDF embedding matrix."""
    def tokenize(sentence):
        """Lowercase, strip possessives and non-alpha characters, split."""
        sentence = sentence.lower()
        sentence = re.sub(r"'s\b", "", sentence)
        sentence = re.sub(r"[^a-z\s]", "", sentence)
        return sentence.split()

    tokenized = [tokenize(s) for s in sentences]

    if vocab is None:
        word_set = set()
        for tokens in tokenized:
            word_set.update(tokens)
        features = sorted(word_set)
    else:
        features = list(vocab)

    s = len(sentences)
    f = len(features)
    feature_index = {word: i for i, word in enumerate(features)}

    # Compute TF: term count / total terms in sentence
    tf = np.zeros((s, f))
    for i, tokens in enumerate(tokenized):
        total = len(tokens)
        if total == 0:
            continue
        for word in tokens:
            if word in feature_index:
                tf[i][feature_index[word]] += 1
        tf[i] /= total

    # Compute IDF: log((1 + s) / (1 + df)) + 1  (sklearn smooth variant)
    idf = np.zeros(f)
    for j, word in enumerate(features):
        df = sum(1 for tokens in tokenized if word in tokens)
        idf[j] = np.log((1 + s) / (1 + df)) + 1

    # TF-IDF = TF * IDF, then L2-normalize each row
    tfidf = tf * idf

    norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    norms[norms == 0] = 1
    tfidf = tfidf / norms

    return tfidf, features
