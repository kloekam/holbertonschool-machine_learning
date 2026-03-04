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
        pad_top = pad_bottom = pad_left = pad_right = 0
    else:
        ph = max((int(np.ceil(h_prev / sh)) - 1) * sh + kh - h_prev, 0)
        pw = max((int(np.ceil(w_prev / sw)) - 1) * sw + kw - w_prev, 0)
        pad_top = ph // 2
        pad_bottom = ph - pad_top
        pad_left = pw // 2
        pad_right = pw - pad_left

    out_h = (h_prev + pad_top + pad_bottom - kh) // sh + 1
    out_w = (w_prev + pad_left + pad_right - kw) // sw + 1

    A_padded = np.pad(A_prev, (
        (0, 0),
        (pad_top, pad_bottom),
        (pad_left, pad_right),
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
