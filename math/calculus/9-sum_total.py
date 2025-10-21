#!/usr/bin/env python3
"""A script that calculates the sum of squares"""


def summation_i_squared(n):
    """A function that calculates the sum of squares"""
    if type(n) is not int or n < 1:
        return None
    return n * (n + 1) * (2 * n + 1) // 6
