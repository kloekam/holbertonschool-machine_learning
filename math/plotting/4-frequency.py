#!/usr/bin/env python3
"""A script that plots a histogram of student grades"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def frequency():

    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    plt.hist(student_grades, bins=[x*10 for x in range(1, 11)],
             edgecolor="black")

    plt.xlim(0, 100)
    plt.ylim(0, 30)

    plt.xticks(np.arange(0, 101, step=10))

    plt.xlabel("Grades")
    plt.ylabel("Number of Students")
    plt.title("Project A")

    plt.show()
