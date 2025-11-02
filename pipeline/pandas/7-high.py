#!/usr/bin/env python3
"""
A script that sorts the DataFrame by the
High price in descending order
"""


def high(df):
    """
    A function that sorts the DataFrame by the
    High price in descending order
    """
    return df.sort_values(by="High", ascending=False)
