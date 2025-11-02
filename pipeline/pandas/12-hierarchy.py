#!/usr/bin/env python3
"""
A script that concatenates bitstamp and coinbase tables
with Timestamp as first MultiIndex level
"""


import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    """
    A function that concatenates bitstamp and coinbase tables
    with Timestamp as first MultiIndex level
    """
    df1 = index(df1)
    df2 = index(df2)
    df1 = df1[(df1.index >= 1417411980) & (df1.index <= 1417417980)]
    df2 = df2[(df2.index >= 1417411980) & (df2.index <= 1417417980)]
    df = pd.concat([df2, df1], keys=['bitstamp', 'coinbase'])
    df = df.swaplevel(0, 1)
    df = df.sort_index(level=0)
    return df
