#!/usr/bin/env python3
"""A script that adds two matrices element-wise"""


def add_matrices2D(mat1, mat2):
    """A function that adds two matrices element-wise"""
    if len(mat1) != len(mat2):
        return None
    rows = len(mat1)
    result = []
    for i in range(rows):
        row1 = mat1[i]
        row2 = mat2[i]
        if len(row1) != len(row2):
            return None
        new_row = []
        cols = len(row1)
        for j in range(cols):
            col1 = row1[j]
            col2 = row2[j]
            new_row.append(col1+col2)
        result.append(new_row)
    return result
