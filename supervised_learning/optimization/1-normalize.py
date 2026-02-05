#!/usr/bin/env python3
"""Optimization Module Tasks"""

import numpy as np


def normalize(X, m, s):
    """A function that normalizes (standardizes) a matrix"""

    z = (X - m) / s

    return z
