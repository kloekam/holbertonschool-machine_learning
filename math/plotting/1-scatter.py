#!/usr/bin/env python3
"""A script that plots student grade frequencies"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def scatter():
    """A function that plots student grade frequencies"""
    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x, y = np.random.multivariate_normal(mean, cov, 2000).T
    y += 180
    plt.figure(figsize=(6.4, 4.8))

    # Set color to magenta
    plt.scatter(x, y, color="magenta")
    # Set title
    plt.title("Men's Height vs Weight")
    # Label to x-axis
    plt.xlabel("Height (in)")
    # Label to y-axis
    plt.ylabel("Weight (lbs)")

    plt.show()
