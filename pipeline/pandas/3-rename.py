#!/usr/bin/env python3
"""
A script that renames columns and converts
timestamp values to datatime values
"""


import pandas as pd


def rename(df):
    """
    A function that renames columns and converts
    timestamp values to datatime values
    """
    df = df.rename(columns={"Timestamp": "Datetime"})
    df["Datetime"] = pd.to_datetime(df["Datetime"], unit="s")
    df = df[["Datetime", "Close"]]
    return df
