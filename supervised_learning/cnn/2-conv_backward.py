#!/usr/bin/env python3
"""
Convolutional Neural Networks Module
"""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    A function that performs back propagation over a
    convolutional layer of a neural network
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    m, h_new, w_new, c_new = dZ.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == "same":
        ph = ((h_prev - 1) * sh + kh - h_prev) // 2 + 1
        pw = ((w_prev - 1) * sw + kw - w_prev) // 2 + 1
    else:
        ph, pw = 0, 0

    A_prev_padded = np.pad(
        A_prev,
        pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode="constant"
    )

    dA_prev_padded = np.zeros_like(A_prev_padded)
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for i in range(h_new):
        for j in range(w_new):
            h_start = i * sh
            h_end = h_start + kh
            w_start = j * sw
            w_end = w_start + kw

            A_slice = A_prev_padded[:, h_start:h_end, w_start:w_end, :]

            for c in range(c_new):
                dZ_slice = dZ[:, i, j, c][:, np.newaxis,
                                          np.newaxis, np.newaxis]

                dA_prev_padded[:, h_start:h_end,
                               w_start:w_end, :] += W[:, :, :, c] * dZ_slice
                dW[:, :, :, c] += np.sum(A_slice * dZ_slice, axis=0)

    if padding == "same":
        dA_prev = dA_prev_padded[:, ph:-ph, pw:-pw, :]
    else:
        dA_prev = dA_prev_padded

    return dA_prev, dW, db
