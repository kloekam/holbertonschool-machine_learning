#!/usr/bin/env python3
"""A script that converts selected values into a numpy.ndarray"""


def array(df):
    """A function that converts selected values into a numpy.ndarray"""
    df = df[["High", "Close"]].tail(10).to_numpy()
    return df
