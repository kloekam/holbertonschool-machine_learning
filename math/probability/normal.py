#!/usr/bin/env python3
"""A script that implements a Normal distribution class"""


class Normal:
    """A class that represents a Normal distribution"""
    def __init__(self, data=None, mean=0., stddev=1.):
        """A function that initializes a Normal distribution instance"""
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data) / len(data)
            squared_differences = []
            for x in data:
                squared_differences.append((x - self.mean) ** 2)
            variance = sum(squared_differences) / len(data)
            self.stddev = variance ** 0.5
