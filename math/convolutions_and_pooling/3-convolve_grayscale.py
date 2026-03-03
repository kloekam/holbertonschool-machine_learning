#!/usr/bin/env python3
"""
Convolution and Pooling Module
"""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    A function that performs a convolution on
    grayscale images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    if padding == 'valid':
        ph, pw = 0, 0
    elif padding == 'same':
        ph = (kh - 1) // 2
        pw = (kw - 1) // 2
    else:
        ph, pw = padding

    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (w + 2 * pw - kw) // sw + 1

    padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), mode='constant')

    output = np.zeros((m, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            i_start = i * sh
            j_start = j * sw
            region = padded[:, i_start:i_start + kh, j_start:j_start + kw]
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2))

    return output
