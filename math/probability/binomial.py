#!/usr/bin/env python3
"""A script that represents Binomial distribution"""


class Binomial:
    """"A class that represents Binomial distribution"""

    def __init__(self, data=None, n=1, p=0.5):
        """A function that initializes a Normal distribution instance"""

        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if not (0 < p < 1):
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)
            p_est = 1 - (variance / mean)
            n_est = round(mean / p_est)
            p_est = mean / n_est
            self.n = int(n_est)
            self.p = float(p_est)

    def factorial(self, num):
        """Calculates the factorial"""
        if num == 0 or num == 1:
            return 1
        fact = 1
        for i in range(2, num + 1):
            fact *= i
        return fact

    def combination(self, n, k):
        """Calculates the number of combinations (n choose k)"""
        if k > n or k < 0:
            return 0
        return self.factorial(n) // (self.factorial(k) * self.factorial(n - k))

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of successes"""
        k = int(k)
        if k < 0 or k > self.n:
            return 0
        comb = self.combination(self.n, k)
        return comb * (self.p ** k) * ((1 - self.p) ** (self.n - k))

    def cdf(self, k):
        """Calculates the value of the CDF for a given number of successes"""
        k = int(k)
        if k < 0:
            return 0
        if k >= self.n:
            k = self.n
        cdf = 0
        for i in range(0, k + 1):
            cdf += self.pmf(i)
        return cdf
