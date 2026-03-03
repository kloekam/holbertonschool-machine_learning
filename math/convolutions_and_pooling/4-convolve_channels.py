#!/usr/bin/env python3
"""
Convolution and Pooling Module
"""

import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    A function that performs a convolution
    on images with channels.
    """
    m, h, w, c = images.shape
    kh, kw, _ = kernel.shape
    sh, sw = stride

    if padding == 'valid':
        ph, pw = 0, 0
    elif padding == 'same':
        ph = (kh - 1) // 2
        pw = (kw - 1) // 2
    else:
        ph, pw = padding

    out_h = 1 + (h + 2 * ph - kh) // sh
    out_w = 1 + (w + 2 * pw - kw) // sw

    padded = np.pad(images, ((0, 0), (ph, ph),
                             (pw, pw), (0, 0)), mode='constant')

    output = np.zeros((m, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            i_start = i * sh
            j_start = j * sw
            region = padded[:, i_start:i_start + kh, j_start:j_start + kw, :]
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2, 3))

    return output
