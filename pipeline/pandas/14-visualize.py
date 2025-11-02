#!/usr/bin/env python3
"""
A script that aggregates and visualizes daily
Bitcoin data form Coinbase
"""


import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Remove Weighted_Price column
df = df.drop(['Weighted_Price'], axis=1)

# Rename the Timestamp column & convert to date values
df = df.rename(columns={'Timestamp': 'Date'})
df['Date'] = pd.to_datetime(df['Date'], unit='s')

# Set index on Date
df = df.set_index(['Date'])

# Fill missing values
df['Close'] = df['Close'].ffill()
df['High'] = df['High'].fillna(df['Close'])
df['Low'] = df['Low'].fillna(df['Close'])
df['Open'] = df['Open'].fillna(df['Close'])

# Set missing values to 0
df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

# Filter from 2017 onward
df = df['2017':]

# Resample daily and aggregate
df = df.resample('D').agg({
    'High': 'max',
    'Low': 'min',
    'Open': 'mean',
    'Close': 'mean',
    'Volume_(BTC)': 'sum',
    'Volume_(Currency)': 'sum'
})

# Print the transfomed dataframe
print(df)

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

df[['High', 'Low', 'Open', 'Close']].plot(ax=ax1)
ax1.set_ylabel('Price (USD)')
ax1.set_title('BTC/USD Daily Prices (2017-2019)')

df[['Volume_(BTC)', 'Volume_(Currency)']].plot(ax=ax2)
ax2.set_ylabel('Volume')
ax2.set_xlabel('Date')
ax2.set_title('BTC/USD Daily Volume (BTC & Currency)')

plt.show()
