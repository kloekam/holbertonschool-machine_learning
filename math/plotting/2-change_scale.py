#!/usr/bin/env python3
"""A script that plots the exponential decay of C-14 on a log scale"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def change_scale():
    """A function that plots the exponential decay of C-14 on a log scale"""
    x = np.arange(0, 28651, 5730)
    r = np.log(0.5)
    t = 5730
    y = np.exp((r / t) * x)
    plt.figure(figsize=(6.4, 4.8))

    plt.plot(x, y)
    # Set y-axis to logarithmic scale
    plt.yscale("log")
    # Set the x-axis limits
    plt.xlim(0, 28650)
    # Label x-axis
    plt.xlabel("Time (years)")
    # Label y-axis
    plt.ylabel("Fraction Remaining")
    # Set title
    plt.title("Exponential Decay of C-14")

    plt.show()
