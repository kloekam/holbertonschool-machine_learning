#!/usr/bin/env python3
"""Error Analysis Module"""

import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    A function that calculates the
    F1 score of a confusion matrix
    """
    precis = precision(confusion)
    sensit = sensitivity(confusion)

    return 2 * (precis * sensit) / (precis + sensit)
