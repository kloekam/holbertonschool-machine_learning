#!/usr/bin/env python3
"""
Convolutional Neural Networks Module
"""

import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    A function that performs back propagation over
    a pooling layer of a neural network
    """
    m, h_prev, w_prev, c = A_prev.shape
    _, h_new, w_new, _ = dA.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros_like(A_prev)

    for i in range(h_new):
        for j in range(w_new):
            h_start = i * sh
            h_end = h_start + kh
            w_start = j * sw
            w_end = w_start + kw

            A_slice = A_prev[:, h_start:h_end, w_start:w_end, :]

            if mode == 'max':
                mask = A_slice == np.max(A_slice, axis=(1, 2), keepdims=True)
                dA_prev[:, h_start:h_end, w_start:w_end, :] += mask * dA[
                    :, i, j, :][:, None, None, :]
            elif mode == 'avg':
                da = dA[:, i, j, :][:, None, None, :]
                average = da / (kh * kw)
                dA_prev[:, h_start:h_end, w_start:w_end, :] += average
            else:
                raise ValueError("mode must be 'max' or 'avg'")

    return dA_prev
