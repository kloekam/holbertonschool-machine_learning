#!/usr/bin/env python3
"""Optimization Module Tasks"""

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """
    A function that creates a learning rate decay operation in
    tensorflow using inverse time decay
    """
    return tf.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True
    )
