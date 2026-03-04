#!/usr/bin/env python3
"""
Convolutional Neural Networks Module
"""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    A function that performs forward propagation over a
    convolutional layer of a neural network
    """
    (m, h_prev, w_prev, c_prev) = A_prev.shape
    (kh, kw, c_prev, c_new) = W.shape
    sh, sw = stride

    if padding == 'same':
        ph = max((h_prev - 1) * sh + kh - h_prev) // 2
        pw = max((w_prev - 1) * sw + kw - w_prev) // 2
    else:
        ph, pw = 0, 0

    padded = np.pad(A_prev, ((0, 0),
                             (ph, ph),
                             (pw, pw),
                             (0, 0)), mode='constant')

    out_h = (h_prev + 2 * ph - kh) // sh + 1
    out_w = (w_prev + 2 * pw - kw) // sw + 1

    Z = np.zeros((m, out_h, out_w, c_new))

    for k in range(c_new):
        for i in range(out_h):
            for j in range(out_w):
                region = padded[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]
                Z[:, i, j, k] = np.sum(region * W[:, :, :, k],
                                       axis=(1, 2, 3))

    Z += b

    return activation(Z)
