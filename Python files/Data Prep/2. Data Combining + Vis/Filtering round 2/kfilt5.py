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
file_path = 'C:/Users/fearh/Desktop/theo pt2/17.4/combined_df.csv'

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

#Hyperparameters
n_bid = 0.1
n_ask = 0.1
time_int = (60)

# Now, convert the 'adjusted_time_2' column, which is a timedelta, to seconds
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





# Group by one-hour intervals and 'Type'
grouped = combined_df.groupby(combined_df['adjusted_time_seconds'] // (time_int))

# Apply the filtering function to each group
filtered_df = grouped.apply(filter_bids_asks, n_bid=n_bid, n_ask=n_ask).reset_index(drop=True)
filtered_df.drop(columns=['new_day_start', 'adjusted_time_2'], inplace=True)


lobs = filtered_df

# Convert 'adjusted_time_seconds' to mins
lobs['adjusted_time'] = pd.to_timedelta(lobs['adjusted_time'])
lobs['adjusted_time'] = lobs['adjusted_time'].dt.floor('T')
lobs['adjusted_time_ints'] = lobs['adjusted_time_seconds'] // time_int

# Now, group by 'Type' and 'adjusted_time_hours' and calculate the mean 'Price' for each min.
# Ensure the 'date' column is carried over in the groupby operation
mean_prices_volumes = lobs.groupby(['Type', 'adjusted_time_ints', 'adjusted_time'])[['Price', 'Volume']].mean().reset_index()


# Measure time taken
time_taken = round(time.time() - time_start_2, 2)
print("Cell 2.1 time taken: " + str(time_taken))

#%% cell 2.2

len_original = len(combined_df)
len_new = len(filtered_df)
percent_lost = round((100*(len_original - len_new)/len_original),2)

print(f"{percent_lost} % of the data was lost")

# Group by adjusted_time_seconds and Type and count entries
grouped_counts = filtered_df.groupby([filtered_df['adjusted_time_seconds'] // time_int, 'Type']).size().unstack(fill_value=0)
# Find the index (group number) with the minimum number of bids and asks
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


# Convert the DataFrame to an Excel file named 'hyperparamtable.xlsx'
file_path = r'C:/Users/fearh/Desktop/theo pt2/18.4/new_ints/filtered_data_mimed0101.csv'

# Save the DataFrame to an CSV file
mean_prices_volumes.to_csv(file_path, index=False)



#%% Cell 3, plotting the filtering (2.5 mins)

# Start time measurement
time_start = time.time()

# Convert timedelta strings to datetime objects

bid_filt_h_df = filtered_df[filtered_df['Type'] == 'bid']
ask_filt_h_df = filtered_df[filtered_df['Type'] == 'ask']
# Plotting
plt.figure(figsize=(10, 6))

# Plot bids
plt.scatter(bid_filt_h_df['adjusted_time_seconds'], bid_filt_h_df['Price'], s=1, alpha=0.02, color='blue', label='Bids')

# Plot asks
plt.scatter(ask_filt_h_df['adjusted_time_seconds'], ask_filt_h_df['Price'], s=1, alpha=0.02, color='orange', label='Asks')

# Add legend
plt.legend()

# Add labels and title
plt.xlabel('Adjusted Time (seconds since start)')
plt.ylabel('Price')
plt.title(f'Scatter Plot of Adjusted Time vs Price for Bids and Asks (n_bid={n_bid}, n_ask={n_ask}) time = {time_int}')

# Show plot

plt.show()
time_taken = round(time.time() - time_start, 2)
print("Cell 3 time taken: " + str(time_taken))


#%% Cell 3.1, zooming in on x axis
# Plotting
plt.figure(figsize=(10, 6))

# Plot bids
plt.scatter(bid_filt_h_df['adjusted_time_seconds'], bid_filt_h_df['Price'], s=1, alpha=0.5, color='green', label='Bids')

# Plot asks
plt.scatter(ask_filt_h_df['adjusted_time_seconds'], ask_filt_h_df['Price'], s=1, alpha=0.5, color='red', label='Asks')

# Add legend
plt.legend()

lim = 60*60*8.5


plt.xlim(0,lim)

# Add labels and title
plt.xlabel('Adjusted Time (seconds since start)')
plt.ylabel('Price')
plt.title(f'Scatter Plot of Adjusted Time vs Price for Bids and Asks (n_bid={n_bid}, n_ask={n_ask})')

# Show plot
plt.show()



#%% Cell 4.1, histogram - Bids
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Prepare the bid DataFrame
bid_filt_h_df = bid_filt_h_df.copy()
bid_filt_h_df['adjusted_time'] = pd.to_timedelta(bid_filt_h_df['adjusted_time'])
adjusted_time_seconds = bid_filt_h_df['adjusted_time'].dt.total_seconds()
bid_filt_h_df.sort_values('adjusted_time', inplace=True)
num_histograms = 10
time_bins = np.linspace(adjusted_time_seconds.min(), adjusted_time_seconds.max(), num_histograms + 1)

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

#%% Cell 4.1.2, histogram.2 - Bids for first 10 hours
adjusted_time_hours = bid_filt_h_df['adjusted_time'].dt.total_seconds() / 3600
filtered_df = bid_filt_h_df[adjusted_time_hours <= 10]
filtered_df.sort_values('adjusted_time', inplace=True)
time_bins = np.linspace(0, 10, num_histograms + 1)

fig, axs = plt.subplots(nrows=num_histograms, figsize=(10, 20), sharex=True)
for i in range(num_histograms):
    mask = (adjusted_time_hours >= time_bins[i]) & (adjusted_time_hours < time_bins[i+1])
    bin_data = filtered_df.loc[mask, 'Price']
    axs[i].hist(bin_data, bins=20, color='blue', edgecolor='black')
    axs[i].set_title(f'Time Interval from {time_bins[i]:.1f} to {time_bins[i+1]:.1f} Hours')
    axs[i].set_ylabel('Frequency')
plt.xlabel('Price')
plt.title('Bids distribution in 10 hours')
plt.tight_layout()
plt.show()

# Repeat the same for ask_filt_h_df with orange histograms

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

#%% Cell 4.2.1, histogram.2 - Asks for first 10 hours
adjusted_time_hours = ask_filt_h_df['adjusted_time'].dt.total_seconds() / 3600
filtered_df = ask_filt_h_df[adjusted_time_hours <= 10]
filtered_df.sort_values('adjusted_time', inplace=True)

fig, axs = plt.subplots(nrows=num_histograms, figsize=(10, 20), sharex=True)
for i in range(num_histograms):
    mask = (adjusted_time_hours >= time_bins[i]) & (adjusted_time_hours < time_bins[i+1])
    bin_data = filtered_df.loc[mask, 'Price']
    axs[i].hist(bin_data, bins=20, color='orange', edgecolor='black')
    axs[i].set_title(f'Time Interval from {time_bins[i]:.1f} to {time_bins[i+1]:.1f} Hours')
    axs[i].set_ylabel('Frequency')
plt.xlabel('Price')
plt.title('Asks distribution in 10 hours')
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
    axs[i].hist(bin_data_bids, bins=5, color='blue', edgecolor='black', alpha=0.5, label='Bids')
    axs[i].hist(bin_data_asks, bins=15, color='orange', edgecolor='black', alpha=0.5, label='Asks')
    axs[i].set_title(f'Time Interval from {time_bins[i]:.1f} to {time_bins[i+1]:.1f} Hours')
    axs[i].set_ylabel('Frequency')
    axs[i].legend()

# Set common labels
plt.xlabel('Price')
plt.tight_layout()
plt.show()

#%% Cell 6
# Sample DataFrame setup (you should replace this with your actual DataFrame loading code)
# bid_filt_h_df = pd.read_csv('path_to_your_data.csv')
# For demonstration, I'm creating a dummy DataFrame
# bid_filt_h_df = pd.DataFrame({
#     'adjusted_time_seconds': pd.to_timedelta(['00:00:00', '00:01:00', '00:04:00', '00:09:00', '00:15:00']).dt.total_seconds()
# })
bid_filt_h_df = filtered_df[filtered_df['Type'] == 'bid']
# Calculate differences between consecutive timestamps in seconds
bid_filt_h_df['time_diff'] = bid_filt_h_df['adjusted_time_seconds'].diff()

# Remove the first entry as it will be NaN after diff calculation
time_differences = bid_filt_h_df['time_diff'].dropna()

# Plotting the histogram of time differences
plt.figure(figsize=(10, 6))
plt.hist(bid_filt_h_df['time_diff'], bins=40, color='blue', alpha=0.7)
plt.title('Histogram of Time Differences Between Consecutive Bids')
plt.xlabel('Time Difference (seconds)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


print(bid_filt_h_df['adjusted_time_seconds'].describe())
print(bid_filt_h_df['adjusted_time_seconds'].isna().sum())

#%% Cell 6.2

# Assume 'time_differences' is a Pandas Series of time differences you've calculated
time_differences = bid_filt_h_df['adjusted_time_seconds'].diff().dropna()

# Determine your cutoff point (this is just an example, adjust it to your needs)
cutoff_point = time_differences.quantile(0.99)  # for example, you might use the 90th percentile

# Filter the data into two parts
dense_part = time_differences[time_differences <= cutoff_point]
sparse_part = time_differences[time_differences > cutoff_point]

# Plot histogram for the dense part
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(dense_part, bins=30, color='blue', alpha=0.7)
plt.title('Dense Part of Time Differences')
plt.xlabel('Time Difference (seconds)')
plt.ylabel('Frequency')

# Plot histogram for the sparse part
plt.subplot(1, 2, 2)
plt.hist(sparse_part, bins=30, color='orange', alpha=0.7)
plt.title('Sparse Part of Time Differences')
plt.xlabel('Time Difference (seconds)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Continuing from the previous snippet, with `dense_part` and `sparse_part` already defined

# Plot histogram for the dense part with logarithmic scale
plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
plt.hist(dense_part, bins=30, color='blue', alpha=0.7, log=True)
plt.title('Dense Part of Time Differences (Log Scale)')
plt.xlabel('Time Difference (seconds)')
plt.ylabel('Log(Frequency)')

# Plot histogram for the sparse part with logarithmic scale
plt.subplot(1, 2, 2)
plt.hist(sparse_part, bins=30, color='orange', alpha=0.7, log=True)
plt.title('Sparse Part of Time Differences (Log Scale)')
plt.xlabel('Time Difference (seconds)')
plt.ylabel('Log(Frequency)')

plt.tight_layout()
plt.show()


