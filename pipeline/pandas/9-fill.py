#!/usr/bin/env python3
"""A script that removes a column and fills missing values"""


def fill(df):
    """A function that removes a column and fills missing values"""
    df = df.drop(['Weighted_Price'], axis=1)
    df["Close"] = df["Close"].ffill()
    df["High"] = df["High"].fillna(df["Close"])
    df["Low"] = df["Low"].fillna(df["Close"])
    df["Open"] = df["Open"].fillna(df["Close"])
    df["Volume_(BTC)"] = df["Volume_(BTC)"].fillna(0)
    df["Volume_(Currency)"] = df["Volume_(Currency)"].fillna(0)

    return df
