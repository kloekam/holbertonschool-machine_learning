#!/usr/bin/env python3
"""A script that represents a Poisson distribution"""


class Poisson:
    """A class that represents a Poisson distribution"""

    def __init__(self, data=None, lambtha=1.):
        """A functiont that initalizes a Poisson distribution"""
        if data is None:
            if type(lambtha) not in (int, float) or lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of successes"""
        if type(k) not in (int, float):
            return 0
        k = int(k)
        if k < 0:
            return 0
        e = 2.7182818285
        fact = 1
        for i in range(1, k + 1):
            fact *= i
        pmf = (e ** (-self.lambtha) * (self.lambtha ** k) / fact)
        return pmf

    def cdf(self, k):
        """Calculates the value of the CDF for a given number of successes"""
        if type(k) not in (int, float):
            return 0
        k = int(k)
        if k < 0:
            return 0
        cdf = 0
        for i in range(k + 1):
            cdf += self.pmf(i)
        return cdf
