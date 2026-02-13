#!/usr/bin/env python3
"""Error Analysis Module"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """A function that creates a confusion matrix"""
    m, classes = labels.shape
    confusion = np.zeros((classes, classes), dtype=float)

    for i in range(m):
        true = np.where(labels[i] == 1)[0][0]
        pred = np.where(logits[i] == 1)[0][0]
        confusion[true, pred] += 1

    return confusion
