#!/usr/bin/env python3
"""Optimization Module Tasks"""

import numpy as np


def learning_rate_decay(alpha,
                        decay_rate, global_step, decay_step):
    """
    A function that updates
    the learning rate using inverse time decay in numpy
    """
    updated_alpha = alpha / (
        1 + decay_rate * np.floor(
            global_step / decay_step
        )
    )
    return updated_alpha
