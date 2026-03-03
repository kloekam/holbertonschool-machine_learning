#!/usr/bin/env python3
"""
Convolution and Pooling Module
"""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    A function tha performs a same
    convolution on grayscale images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    pad_h = kh // 2
    pad_w = kw // 2

    padded = np.pad(
        images,
        pad_width=((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
        mode='constant',
        constant_values=0
    )

    output = np.zeros((m, h, w))

    for i in range(h):
        for j in range(w):
            region = padded[:, i:i+kh, j:j+kw]
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2))

    return output
