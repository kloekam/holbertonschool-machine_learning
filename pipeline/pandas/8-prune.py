#!/usr/bin/env python3
"""A script that removes entries where Close has NaN values"""


def prune(df):
    """A function that removes entries where Close has NaN values"""
    return df.dropna(subset=["Close"])
