#!/usr/bin/env python3
"""A script that performs matrix multiplication"""


def mat_mul(mat1, mat2):
    """A function that performs matrix multiplication"""
    if len(mat1[0]) != len(mat2):
        return None
    result = []
    for i in range(len(mat1)):
        row_result = []
        for j in range(len(mat2[0])):
            dot_product = 0
            for k in range(len(mat1[0])):
                dot_product += mat1[i][k] * mat2[k][j]
            row_result.append(dot_product)
        result.append(row_result)
    return result
