#!/usr/bin/env python3
"""A script that concatenates two matrices along a specific axis"""


def cat_matrices2D(mat1, mat2, axis=0):
    """A function that concatenates two matrices along a specific axis"""
    mat = []
    if axis == 0:
        cols = len(mat1[0])
        for row in mat1:
            if len(row) != cols:
                return None
            mat.append(row[:])
        for row in mat2:
            if len(row) != cols:
                return None
            mat.append(row[:])
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        for i in range(len(mat1)):
            row1 = mat1[i]
            row2 = mat2[i]
            new_row = []
            for j in range(len(row1)):
                new_row.append(row1[j])
            for j in range(len(row2)):
                new_row.append(row2[j])
            mat.append(new_row)
    return mat
    