#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 14:23:13 2024

@author: theorogers
"""

#%% Cell 1 Read in Data (30s)
import os
import pandas as pd
import matplotlib.pyplot as plt
import time

time_start_1 = time.time()

# Replace 'your_file.csv' with the path to your CSV file
file_path = 'C:/Users/fearh/Desktop/theo pt2/combined_df.csv'

# Reading the CSV file into a DataFrame
combined_df = pd.read_csv(file_path)


time_taken = round(time.time() - time_start_1,2)
print("Cell 1 time taken: " + str(time_taken))


#%% Cell 2.1 filtering (v3)

# Start time measurement
time_start_2 = time.time()
# Convert timedelta strings to datetime objects
combined_df['adjusted_time'] = pd.to_timedelta(combined_df['adjusted_time'])

# Define the function to filter bids and asks
def filter_bids_asks(group, n_bid, n_ask):
    if 'bid' in group['Type'].values:
        max_bid_price = group.loc[group['Type'] == 'bid', 'Price'].max()
        min_bid_price = group.loc[group['Type'] == 'bid', 'Price'].min()
        bid_std = group.loc[group['Type'] == 'bid', 'Price'].std()
        filtered_bids = group[(group['Type'] == 'bid') & 
                              (group['Price'] >= max_bid_price - n_bid * bid_std) & 
                              (group['Price'] <= min_bid_price + n_bid * bid_std)]
    else:
        filtered_bids = pd.DataFrame(columns=group.columns)  # Empty DataFrame if no bids
    
    if 'ask' in group['Type'].values:
        min_ask_price = group.loc[group['Type'] == 'ask', 'Price'].min()
        max_ask_price = group.loc[group['Type'] == 'ask', 'Price'].max()
        ask_std = group.loc[group['Type'] == 'ask', 'Price'].std()
        filtered_asks = group[(group['Type'] == 'ask') & 
                              (group['Price'] >= min_ask_price - n_ask * ask_std) & 
                              (group['Price'] <= max_ask_price + n_ask * ask_std)]
    else:
        filtered_asks = pd.DataFrame(columns=group.columns)  # Empty DataFrame if no asks
    
    return pd.concat([filtered_bids, filtered_asks])



# Define the value of 'n' for bids and asks separately
n_bid = 10
n_ask = 1/10

# Group by one-hour intervals and 'Type'
grouped = combined_df.groupby([pd.Grouper(key='adjusted_time', freq='H')])

# Apply the filtering function to each group
filtered_df = grouped.apply(filter_bids_asks, n_bid=n_bid, n_ask=n_ask).reset_index(drop=True)

# Measure time taken
time_taken = round(time.time() - time_start_2, 2)
print("Cell 2.1 time taken: " + str(time_taken))


#%% Cell 2.2, plotting the filtering

# Start time measurement
time_start_2_2 = time.time()

# Convert timedelta strings to datetime objects
combined_df['adjusted_time'] = pd.to_timedelta(combined_df['adjusted_time'])
bid_filt_h_df = filtered_df[filtered_df['Type'] == 'bid']
ask_filt_h_df = filtered_df[filtered_df['Type'] == 'ask']
# Plotting
plt.figure(figsize=(10, 6))

# Plot bids
plt.scatter(bid_filt_h_df['adjusted_time'], bid_filt_h_df['Price'], s=1, alpha=0.02, color='blue', label='Bids')

# Plot asks
plt.scatter(ask_filt_h_df['adjusted_time'], ask_filt_h_df['Price'], s=1, alpha=0.02, color='orange', label='Asks')

# Add legend
plt.legend()

# Add labels and title
plt.xlabel('Adjusted Time (seconds since start)')
plt.ylabel('Price')
plt.title(f'Scatter Plot of Adjusted Time vs Price for Bids and Asks (n_bid={n_bid}, n_ask={n_ask})')

# Show plot

plt.show()
time_taken = round(time.time() - time_start_2_2, 2)
print("Cell 2.2 time taken: " + str(time_taken))


#%% Cell 2.3, zooming in on x axis
# Plotting
plt.figure(figsize=(10, 6))

# Plot bids
plt.scatter(bid_filt_h_df['adjusted_time'], bid_filt_h_df['Price'], s=2, alpha=0.5, color='green', label='Bids')

# Plot asks
plt.scatter(ask_filt_h_df['adjusted_time'], ask_filt_h_df['Price'], s=2, alpha=0.05, color='red', label='Asks')

# Add legend
plt.legend()

lim = 4*(10**15)

plt.xlim(0,lim/1000)

# Add labels and title
plt.xlabel('Adjusted Time (seconds since start)')
plt.ylabel('Price')
plt.title(f'Scatter Plot of Adjusted Time vs Price for Bids and Asks (n_bid={n_bid}, n_ask={n_ask})')

# Show plot
plt.show()

