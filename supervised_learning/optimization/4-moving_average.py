#!/usr/bin/env python3
"""Optimization Module Tasks"""

import numpy as np


def moving_average(data, beta):
    """
    A function that calculates the
    weighted moving average of a data set
    """
    moving_avgs = []
    v = 0

    for t in range(len(data)):
        v = beta * v + (1 - beta) * data[t]
        v_corrected = v / (1 - np.power(beta, t + 1))
        moving_avgs.append(v_corrected)

    return moving_avgs
