#!/usr/bin/env python3
"""
Data Augmentation Module
"""

import tensorflow as tf


def change_brightness(image, max_delta):
    """
    Randomly changes the brightness of an image.
    """
    return tf.image.random_brightness(image, max_delta=max_delta)
