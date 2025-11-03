#!/usr/bin/env python3
"""A script that plots a cubic function as a line graph"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def line():

    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    plt.plot(y, color="red")
    plt.xlim(0, 10)

    plt.show()
