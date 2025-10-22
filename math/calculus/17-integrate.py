#!/usr/bin/env python3
"""A script that calculates the integral  of a polynomial"""


def poly_integral(poly, C=0):
    """A function that calculates the integral  of a polynomial"""
    if type(poly) is not list or len(poly) == 0:
        return None
    if type(C) not in (int, float):
        return None
    result = [C]
    for p in range(len(poly)):
        if type(poly[p]) not in (int, float):
            return None
        result.append(poly[p] / (p+1))
    for i in range(len(result)):
        if isinstance(result[i], float) and result[i].is_integer():
            result[i] = int(result[i])
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return result
