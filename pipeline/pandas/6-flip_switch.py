#!/usr/bin/env python3
"""A script that sorts and transposes the sorted dataframe"""


def flip_switch(df):
    """A function that sorts and transposes the sorted dataframe"""
    df = df.sort_values(by="Timestamp", ascending=False).T
    return df
