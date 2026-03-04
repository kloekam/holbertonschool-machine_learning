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

    if padding == 'valid':
        ph, pw = 0, 0
    else:
        ph = (kh - 1) // 2
        pw = (kw - 1) // 2

    out_h = 1 + (h_prev + 2 * ph - kh) // sh
    out_w = 1 + (w_prev + 2 * pw - kw) // sw

    A_padded = np.pad(A_prev, (
        (0, 0),
        (ph, ph),
        (pw, pw),
        (0, 0)),
        mode='constant')

    Z = np.zeros((m, out_h, out_w, c_new))

    for i in range(out_h):
        for j in range(out_w):
            for k in range(c_new):
                i_start = i * sh
                j_start = j * sw
                region = A_padded[:, i_start:i_start+kh, j_start:j_start+kw, :]
                Z[:, i, j, k] = (np.sum(region * W[:, :, :, k],
                                        axis=(1, 2, 3)) + b[0, 0, 0, k])

    A = activation(Z)

    return A
