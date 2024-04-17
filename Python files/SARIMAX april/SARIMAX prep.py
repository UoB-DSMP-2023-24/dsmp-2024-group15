#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 14:41:01 2024

@author: theorogers
"""


#%% Cell 1 Read in Data (30s)

import pandas as pd
import matplotlib.pyplot as plt
import time

time_start_1 = time.time()


# Replace 'your_file.csv' with the path to your CSV file
file_path = r"/Users/theorogers/Desktop/LOB filt/filtered_data_0506.csv"


# Reading the CSV file into a DataFrame
lobs = pd.read_csv(file_path)


max(lobs['adjusted_time_seconds'])

time_taken = round(time.time() - time_start_1,2)
print("Cell 1 time taken: " + str(time_taken))

#%% Cell 2 hourly mean of data (2s)

time_start_2 = time.time()

# Assuming 'lobs' is your DataFrame and it has the 'adjusted_time_seconds', 'Type', and 'Price' columns.
# You'll first want to convert the 'adjusted_time_seconds' to hours.

# Convert 'adjusted_time_seconds' to hours
lobs['adjusted_time_hours'] = lobs['adjusted_time_seconds'] // 3600

# Now, group by 'Type' and 'adjusted_time_hours' and calculate the mean 'Price' for each hour.
mean_prices = lobs.groupby(['Type', 'adjusted_time_hours'])['Price'].mean().reset_index()

# If you need to separate the bid and ask prices, you can do this:
mean_bid_prices = mean_prices[mean_prices['Type'] == 'bid']
mean_ask_prices = mean_prices[mean_prices['Type'] == 'ask']


time_taken = round(time.time() - time_start_2,2)
print("Cell 2 time taken: " + str(time_taken))

#%% Cell 3 plots (2s)


# Plotting the scatter plot
plt.figure(figsize=(10, 6))

# Plot bids in blue
plt.scatter(mean_bid_prices['adjusted_time_hours'], mean_bid_prices['Price'], color='blue', label='Bids')

# Plot asks in orange
plt.scatter(mean_ask_prices['adjusted_time_hours'], mean_ask_prices['Price'], color='orange', label='Asks')

# Labeling the plot
plt.title('Mean Bid/Ask Prices per Hour')
plt.xlabel('Adjusted Time (hours)')
plt.ylabel('Mean Price')
plt.legend()

# Show the plot
plt.show()

mean_ask_prices.to_csv('SARIMAX_asks.csv', index=False)
mean_bid_prices.to_csv('SARIMAX_bids.csv', index=False)