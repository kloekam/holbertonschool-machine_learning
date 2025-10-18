#!/usr/bin/env python3
"""A script that concatenates two arrays"""


def cat_arrays(arr1, arr2):
    """A function that concatenates two arrays"""
    arr = []
    for i in range(len(arr1)):
        arr.append(arr1[i])
    for i in range(len(arr2)):
        arr.append(arr2[i])
    return arr
