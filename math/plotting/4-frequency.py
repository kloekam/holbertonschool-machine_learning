#!/usr/bin/env python3
"""A script that plots a histogram of student grades"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def frequency():
    """A function that plots a histogram of student grades"""
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    # Plot the histogram
    plt.hist(student_grades, bins=np.arange(0, 101, 10),
             edgecolor="black")

    # Set x-axis and y-axis limits
    plt.xlim(0, 100)
    plt.ylim(0, 30)

    # Set x-axis ticks every 10 units
    plt.xticks(np.arange(0, 101, step=10))

    plt.xlabel("Grades")
    plt.ylabel("Number of Students")
    plt.title("Project A")

    plt.show()
