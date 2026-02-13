#!/usr/bin/env python3
"""Error Analysis Module"""

import numpy as np


def precision(confusion):
    """
    A function that calculates the precision for
    each class in a confusion matrix
    """
    true_p = np.diag(confusion)
    total_pred = np.sum(confusion, axis=0)

    return true_p / total_pred
