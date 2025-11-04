#!/usr/bin/env python3
"""A script that plots all five previous graphs in one figure"""


import numpy as np
import matplotlib.pyplot as plt


def all_in_one():
    """A function that plots all five previous graphs in one figure"""
    y0 = np.arange(0, 11) ** 3

    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
    y1 += 180

    x2 = np.arange(0, 28651, 5730)
    r2 = np.log(0.5)
    t2 = 5730
    y2 = np.exp((r2 / t2) * x2)

    x3 = np.arange(0, 21000, 1000)
    r3 = np.log(0.5)
    t31 = 5730
    t32 = 1600
    y31 = np.exp((r3 / t31) * x3)
    y32 = np.exp((r3 / t32) * x3)

    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)

    plt.figure(figsize=(12, 8))
    plt.suptitle("All in One")

    # Plot 1
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(y0, color="red")
    ax1.set_xlim(0, 10)

    # Plot 2
    ax2 = plt.subplot(3, 2, 2)
    ax2.scatter(x1, y1, color="magenta")
    ax2.set_title("Men's Height vs Weight", fontsize="x-small")
    ax2.set_xlabel("Height (in)", fontsize="x-small")
    ax2.set_ylabel("Weight (lbs)", fontsize="x-small")

    # Plot 3
    ax3 = plt.subplot(3, 2, 3)
    ax3.plot(x2, y2, color="blue")
    ax3.set_xlim((0, 28650))
    ax3.set_yscale("log")
    ax3.set_xlabel("Time (years)", fontsize="x-small")
    ax3.set_ylabel("Fraction Remaining", fontsize="x-small")
    ax3.set_title("Exponential Decay of C-14", fontsize="x-small")

    # Plot 4
    ax4 = plt.subplot(3, 2, 4)
    ax4.plot(x3, y32, color="green", label="Ra-336")
    ax4.plot(x3, y31, color="red", label="C-14", linestyle="dashed")
    ax4.set_xlim(0, 20000)
    ax4.set_ylim(0, 1)
    ax4.set_xlabel("Time (years)", fontsize="x-small")
    ax4.set_ylabel("Fraction Remaining", fontsize="x-small")
    ax4.set_title("Exponential Decay of Radioactive Elements",
                  fontsize="x-small")
    ax4.legend(fontsize="x-small")

    # Plot 5
    ax5 = plt.subplot(3, 1, 3)
    ax5.hist(student_grades, bins=np.arange(0, 101, 10), edgecolor="black")
    ax5.set_xlim(0, 100)
    ax5.set_ylim(0, 30)
    ax5.set_xticks(np.arange(0, 101, 10))
    ax5.set_xlabel("Grades", fontsize="x-small")
    ax5.set_ylabel("Number of Students", fontsize="x-small")
    ax5.set_title("Project A", fontsize="x-small")

    plt.tight_layout()
    plt.show()
