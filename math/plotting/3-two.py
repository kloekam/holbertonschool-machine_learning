#!/usr/bin/env python3
"""A script that plots the exponential decay of C-14 and Ra-226"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def two():
    """A function that plots the exponential decay of C-14 and Ra-226"""
    x = np.arange(0, 21000, 1000)
    r = np.log(0.5)
    t1 = 5730
    t2 = 1600
    y1 = np.exp((r / t1) * x)
    y2 = np.exp((r / t2) * x)
    plt.figure(figsize=(6.4, 4.8))

    # Plot the C-14 and the Ra-226 graph
    plt.plot(x, y1, color="red", linestyle="dashed", label="C-14")
    plt.plot(x, y2, color="green", label="Ra-226")

    # Set x-axis and y-axis limits
    plt.xlim(0, 20000)
    plt.ylim(0, 1)

    # Display legend
    plt.legend()

    plt.xlabel("Time (years)")
    plt.ylabel("Fraction Remaining")
    plt.title("Exponential Decay of Radioactive Elements")

    plt.show()
