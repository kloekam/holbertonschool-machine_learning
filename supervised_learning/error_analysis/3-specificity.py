#!/usr/bin/env python3
"""Error Analysis Module"""

import numpy as np


def specificity(confusion):
    """
    A function  that calculates the specificity for
    each class in a confusion matrix
    """
    classes = confusion.shape[0]
    specific = np.zeros(classes)

    for i in range(classes):
        true_n = np.sum(confusion) - np.sum(
            confusion[i, :]) - np.sum(
                confusion[:, i]) + confusion[i, i]
        total_n = np.sum(confusion) - np.sum(confusion[i, :])
        if total_n > 0:
            specific[i] = true_n / total_n
        else:
            specific[i] = 1.0

    return specific
