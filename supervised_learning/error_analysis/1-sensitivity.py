#!/usr/bin/env python3
"""Error Analysis Module"""

import numpy as np


def sensitivity(confusion):
    """
    A function  that calculates the sensitivity
    for each class in a confusion matrix
    """
    true_p = np.diag(confusion)
    total_p = np.sum(confusion, axis=1)

    return true_p / total_p
