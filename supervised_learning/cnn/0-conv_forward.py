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
        ph = max((kh - 1), 0)
        pw = max((kw - 1), 0)

    pad_top = ph // 2
    pad_bottom = ph - pad_top
    pad_left = pw // 2
    pad_right = pw - pad_left

    out_h = (h_prev + ph - kh) // sh + 1
    out_w = (w_prev + pw - kw) // sw + 1

    A_padded = np.pad(A_prev,
                      ((0, 0),
                       (pad_top, pad_bottom),
                       (pad_left, pad_right),
                       (0, 0)),
                      mode='constant')

    Z = np.zeros((m, out_h, out_w, c_new))

    for i in range(out_h):
        for j in range(out_w):
            for k in range(c_new):
                region = A_padded[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]
                Z[:, i, j, k] = np.sum(region * W[:, :, :, k],
                                       axis=(1, 2, 3)) + b[0, 0, 0, k]

    return activation(Z)
