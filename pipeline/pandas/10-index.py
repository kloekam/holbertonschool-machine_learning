#!/usr/bin/env python
"""
A script that sets the Timestamp column
as the index of the DataFrame
"""


def index(df):
    """
    A function that sets the Timestamp column
    as the index of the DataFrame
    """
    return df.set_index(['Timestamp'])
