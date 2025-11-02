#!/usr/bin/env python3
"""
A script that computes descriptive statistics
for all columns except the Timestamp column
"""


def analyze(df):
    """
    A script that computes descriptive statistics
    for all columns except the Timestamp column
    """
    return df.loc[:, df.columns != 'Timestamp'].describe()
