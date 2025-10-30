#!/usr/bin/env python3
"""A script that creates a pd.DataFrame from a np.ndarray"""


import pandas as pd
import numpy as np


def from_numpy(array):
    """A function that creates a pd.DataFrame from a np.ndarray"""
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    num_cols = array.shape[1]
    col_names = list(letters[:num_cols])
    df = pd.DataFrame(array, columns=col_names)
    return df
