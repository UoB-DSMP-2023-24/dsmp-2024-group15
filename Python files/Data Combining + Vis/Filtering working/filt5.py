#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 14:23:13 2024

@author: theorogers
"""

#%% Cell 1 Read in Data (40s)

import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np

time_start_1 = time.time()

# Replace 'your_file.csv' with the path to your CSV file
file_path = 'C:/Users/fearh/Desktop/theo pt2/combined_df.csv'

# Reading the CSV file into a DataFrame
combined_df = pd.read_csv(file_path)

#combined_df =combined_df.head(round(len(combined_df)/100))




time_taken = round(time.time() - time_start_1,2)
print("Cell 1 time taken: " + str(time_taken))


#%% Cell 2.1 filtering (v3) (1.5 mins)

# Start time measurement
time_start_2 = time.time()
# Convert the 'adjusted_time' column to timedelta objects first
combined_df['adjusted_time_2'] = pd.to_timedelta(combined_df['adjusted_time'])

#Hyperparameters, (0.5,0.6 3600) chosen as optimal
n_bid = 0.5
n_ask = 0.6
time_int = (3600)

# Convert the 'adjusted_time_2' column, from timedelta to seconds
combined_df['adjusted_time_seconds'] = combined_df['adjusted_time_2'].dt.total_seconds()

# Define the function to filter bids and asks
def filter_bids_asks(group, n_bid, n_ask):
    if 'bid' in group['Type'].values:
        max_bid_price = group.loc[group['Type'] == 'bid', 'Price'].max()
       
        bid_std = group.loc[group['Type'] == 'bid', 'Price'].std()
        filtered_bids = group[(group['Type'] == 'bid') & 
                              (group['Price'] >= max_bid_price - n_bid * bid_std)]
    else:
        filtered_bids = pd.DataFrame(columns=group.columns)  # Empty DataFrame if no bids
    
    if 'ask' in group['Type'].values:
        min_ask_price = group.loc[group['Type'] == 'ask', 'Price'].min()
       
        ask_std = group.loc[group['Type'] == 'ask', 'Price'].std()
        filtered_asks = group[(group['Type'] == 'ask') & 
                              (group['Price'] <= min_ask_price + n_ask * ask_std)]
    else:
        filtered_asks = pd.DataFrame(columns=group.columns)  # Empty DataFrame if no asks
    
    return pd.concat([filtered_bids, filtered_asks])





# Group based on the time interval and bid/ask
grouped = combined_df.groupby(combined_df['adjusted_time_seconds'] // (time_int))

# Apply the filtering function to each group
filtered_df = grouped.apply(filter_bids_asks, n_bid=n_bid, n_ask=n_ask).reset_index(drop=True)
filtered_df.drop(columns=['new_day_start', 'adjusted_time_2'], inplace=True)


time_taken = round(time.time() - time_start_2, 2)
print("Cell 2.1 time taken: " + str(time_taken))

#%% Cell 2.2 Statistics about the data Given

#The percent of the data that has been filtered out
len_original = len(combined_df)
len_new = len(filtered_df)
percent_lost = round((100*(len_original - len_new)/len_original),2)
print(f"{percent_lost} % of the data was lost")

# Group by adjusted_time_seconds and bid/ask and count entries
grouped_counts = filtered_df.groupby([filtered_df['adjusted_time_seconds'] // time_int, 'Type']).size().unstack(fill_value=0)

# Find the index of the group number with the minimum number of bids and asks
min_bids_group = grouped_counts['bid'].idxmin()
min_asks_group = grouped_counts['ask'].idxmin()

# Print the minimum counts along with the corresponding group
print(f"Smallest number of bids: {grouped_counts['bid'].min()}")
print(f"Smallest number of asks: {grouped_counts['ask'].min()}")


# Filter the DataFrame to get only the rows corresponding to the smallest groups
min_bids_data = filtered_df[(filtered_df['adjusted_time_seconds'] // time_int == min_bids_group) & (filtered_df['Type'] == 'bid')]
min_asks_data = filtered_df[(filtered_df['adjusted_time_seconds'] // time_int == min_asks_group) & (filtered_df['Type'] == 'ask')]

# Plot histograms of price distributions for these specific groups
plt.figure(figsize=(12, 6))

# Histogram for prices in the group with the smallest number of bids
plt.subplot(1, 2, 1)
plt.hist(min_bids_data['Price'], bins=20, color='blue', alpha=0.7)
plt.title(f'Price Distribution for Group {min_bids_group} (Bids) (n_bid = {n_bid})')
plt.xlabel('Price')
plt.ylabel('Frequency')

# Histogram for prices in the group with the smallest number of asks
plt.subplot(1, 2, 2)
plt.hist(min_asks_data['Price'], bins=20, color='orange', alpha=0.7)
plt.title(f'Price Distribution for Group {min_asks_group} (Asks) (n_ask = {n_ask})')
plt.xlabel('Price')

plt.tight_layout()
plt.show()



filtered_df.to_csv('filtered_data_0506.csv', index=False)


#%% Cell 3, plotting the filtering (2.5 mins)

# Start time measurement
time_start = time.time()

# Convert timedelta strings to datetime objects

bid_filt_h_df = filtered_df[filtered_df['Type'] == 'bid']
ask_filt_h_df = filtered_df[filtered_df['Type'] == 'ask']

# Plotting
plt.figure(figsize=(10, 6))

# Bids
plt.scatter(bid_filt_h_df['adjusted_time_seconds'], bid_filt_h_df['Price'], s=1, alpha=0.02, color='blue', label='Bids')

# Asks
plt.scatter(ask_filt_h_df['adjusted_time_seconds'], ask_filt_h_df['Price'], s=1, alpha=0.02, color='orange', label='Asks')

plt.legend()

plt.xlabel('Adjusted Time (seconds since start)')
plt.ylabel('Price')
plt.title(f'Scatter Plot of Adjusted Time vs Price for Bids and Asks (n_bid={n_bid}, n_ask={n_ask}) time = {time_int}')


plt.show()
time_taken = round(time.time() - time_start, 2)
print("Cell 3 time taken: " + str(time_taken))


#%% Cell 3.1, zooming in on x axis (unimportant)
# Plotting
plt.figure(figsize=(10, 6))

# Bids
plt.scatter(bid_filt_h_df['adjusted_time_seconds'], bid_filt_h_df['Price'], s=1, alpha=0.5, color='green', label='Bids')

# Asks
plt.scatter(ask_filt_h_df['adjusted_time_seconds'], ask_filt_h_df['Price'], s=1, alpha=0.5, color='red', label='Asks')

plt.legend()

lim = 60*60*8.5


plt.xlim(0,lim)

plt.xlabel('Adjusted Time (seconds since start)')
plt.ylabel('Price')
plt.title(f'Scatter Plot of Adjusted Time vs Price for Bids and Asks (n_bid={n_bid}, n_ask={n_ask})')

plt.show()



#%% Cell 4.1, histogram - Bids

# Prepare the bid DataFrame
bid_filt_h_df = bid_filt_h_df.copy()
bid_filt_h_df['adjusted_time'] = pd.to_timedelta(bid_filt_h_df['adjusted_time'])
adjusted_time_seconds = bid_filt_h_df['adjusted_time'].dt.total_seconds()
bid_filt_h_df.sort_values('adjusted_time', inplace=True)
num_histograms = 10
time_bins = np.linspace(adjusted_time_seconds.min(), adjusted_time_seconds.max(), num_histograms + 1)

# Plotting 10 histograms for the data split into 10 time intervals
fig, axs = plt.subplots(nrows=num_histograms, figsize=(10, 20), sharex=True)
for i in range(num_histograms):
    mask = (adjusted_time_seconds >= time_bins[i]) & (adjusted_time_seconds < time_bins[i+1])
    bin_data = bid_filt_h_df.loc[mask, 'Price']
    axs[i].hist(bin_data, bins=20, color='blue', edgecolor='black')
    axs[i].set_title(f'Time Interval {i+1}: {time_bins[i]:.0f} to {time_bins[i+1]:.0f}')
    axs[i].set_ylabel('Frequency')
plt.xlabel('Price')
plt.title('Bids distribution in 10 increments')
plt.tight_layout()
plt.show()


#%% Cell 4.2, histogram - Asks
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Prepare the ask DataFrame
ask_filt_h_df = ask_filt_h_df.copy()
ask_filt_h_df['adjusted_time'] = pd.to_timedelta(ask_filt_h_df['adjusted_time'])
adjusted_time_seconds = ask_filt_h_df['adjusted_time'].dt.total_seconds()
ask_filt_h_df.sort_values('adjusted_time', inplace=True)
num_histograms = 10
time_bins = np.linspace(adjusted_time_seconds.min(), adjusted_time_seconds.max(), num_histograms + 1)

# Plotting 10 histograms for the data split into 10 time intervals
fig, axs = plt.subplots(nrows=num_histograms, figsize=(10, 20), sharex=True)
for i in range(num_histograms):
    mask = (adjusted_time_seconds >= time_bins[i]) & (adjusted_time_seconds < time_bins[i+1])
    bin_data = ask_filt_h_df.loc[mask, 'Price']
    axs[i].hist(bin_data, bins=20, color='orange', edgecolor='black')
    axs[i].set_title(f'Time Interval {i+1}: {time_bins[i]:.0f} to {time_bins[i+1]:.0f}')
    axs[i].set_ylabel('Frequency')
plt.xlabel('Price')
plt.title('Asks distribution in 10 increments')
plt.tight_layout()
plt.show()



#%% Cell 5, histogram.3


# Convert 'adjusted_time' to total seconds and filter data to the first 10 hours for both DataFrames
def prepare_data(df):
    df = df.copy()
    df['adjusted_time'] = pd.to_timedelta(df['adjusted_time'])
    df['hours'] = df['adjusted_time'].dt.total_seconds() / 3600  # convert seconds to hours
    df = df[df['hours'] <= 10]  # filter to first 10 hours
    df.sort_values('adjusted_time', inplace=True)
    return df

bid_filt_h_df = prepare_data(bid_filt_h_df)
ask_filt_h_df = prepare_data(ask_filt_h_df)

# Define the number of histograms (one per hour)
num_histograms = 10
time_bins = np.linspace(0, 10, num_histograms + 1)  # 10 hour time bins

# Create the histograms
fig, axs = plt.subplots(nrows=num_histograms, figsize=(10, 20), sharex=True)

for i in range(num_histograms):
    # Bids
    mask_bids = (bid_filt_h_df['hours'] >= time_bins[i]) & (bid_filt_h_df['hours'] < time_bins[i+1])
    bin_data_bids = bid_filt_h_df.loc[mask_bids, 'Price']
    
    # Asks
    mask_asks = (ask_filt_h_df['hours'] >= time_bins[i]) & (ask_filt_h_df['hours'] < time_bins[i+1])
    bin_data_asks = ask_filt_h_df.loc[mask_asks, 'Price']

    # Plot histograms
    axs[i].hist(bin_data_bids, bins=20, color='blue', edgecolor='black', alpha=0.5, label='Bids')
    axs[i].hist(bin_data_asks, bins=20, color='orange', edgecolor='black', alpha=0.5, label='Asks')
    axs[i].set_title(f'Time Interval from {time_bins[i]:.1f} to {time_bins[i+1]:.1f} Hours')
    axs[i].set_ylabel('Frequency')
    axs[i].legend()

# Set common labels
plt.xlabel('Price')
plt.tight_layout()
plt.show()





