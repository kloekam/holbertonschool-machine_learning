#!/usr/bin/env python3
"""A script that slices pd.DataFrame"""


def slice(df):
    """A function that slices pd.DataFrame"""
    df = df[["High", "Low", "Close", "Volume_(BTC)"]].iloc[::60]
    return df
