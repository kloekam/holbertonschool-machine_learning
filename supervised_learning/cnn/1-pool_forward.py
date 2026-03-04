#!/usr/bin/env python3
"""
Convolutional Neural Networks Module
"""

import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    A function that performs forward propagation over
    a pooling layer of a neural network
    """
    (m, h_prev, w_prev, c_prev) = A_prev.shape
    (kh, kw) = kernel_shape
    (sh, sw) = stride

    out_h = 1 + (h_prev - kh) // sh
    out_w = 1 + (w_prev - kw) // sw

    A = np.zeros((m, out_h, out_w, c_prev))

    for i in range(out_h):
        for j in range(out_w):
            i_start = i * sh
            j_start = j * sw

            window = A_prev[:, i_start:i_start+kh, j_start:j_start+kw, :]

            if mode == 'max':
                A[:, i, j, :] = np.max(window, axis=(1, 2))
            else:
                A[:, i, j, :] = np.mean(window, axis=(1, 2))

    return A
