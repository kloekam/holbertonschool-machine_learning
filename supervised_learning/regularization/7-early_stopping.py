#!/usr/bin/env python3
"""Regularization Module"""

import numpy as np


def early_stopping(cost,
                   opt_cost,
                   threshold,
                   patience,
                   count):
    """A function that determines if
    you should stop gradient descent early"""
    if opt_cost - cost > threshold:
        return False, 0
    else:
        count += 1
        return count >= patience, count
