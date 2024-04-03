#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 22:40:45 2024

@author: theorogers
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import time

time_start = time.time()


dataframes = []

# Define the directory to iterate over
directory = "/Users/theorogers/Desktop/LOB filt/Data filtered"

# Get a sorted list of file names
sorted_filenames = sorted(os.listdir(directory))

# Iterate over each file in the directory in alphabetical order
for filename in sorted(os.listdir(directory)):
    file_path = os.path.join(directory, filename)
    if os.path.isfile(file_path) and file_path.endswith('.csv'):
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        # Add a new column to identify the file source (optional)
        df['source_file'] = filename
        # Append the DataFrame to the list
        dataframes.append(df)
        


combined_df = pd.concat(dataframes, ignore_index=True)
combined_df = combined_df.drop(columns=['Index'])

# Assuming 'Time' is in seconds and 'source_file' uniquely identifies each day
combined_df['Time'] = pd.to_timedelta(combined_df['Time'], unit='s')

# Sort by source_file and Time (if not already sorted)
combined_df = combined_df.sort_values(by=['source_file', 'Time'])




# Calculate the offset for each day
# First, map each day to a unique number (0 for the first day, 1 for the second, etc.)
day_mapping = {day: idx for idx, day in enumerate(combined_df['source_file'].unique())}

# Apply the mapping to create a new column 'day_number'
combined_df['day_number'] = combined_df['source_file'].map(day_mapping)

# Calculate the time offset (in seconds) for each day and apply it
offset_per_day = 8.5 * 60 * 60  # 8.5 hours in seconds
combined_df['adjusted_time'] = combined_df['Time'] + pd.to_timedelta(combined_df['day_number'] * offset_per_day, unit='s')


# Create a binary column to indicate the start of a new day
combined_df['new_day_start'] = combined_df['day_number'].diff().fillna(1).astype(bool).astype(int)

# Drop the 'day_number' column if not needed
#combined_df = combined_df.drop(columns=['day_number'])

# Now combined_df has 'adjusted_time' and 'new_day_start' columns as required
time_taken = time.time() - time_start
print("Time taken: " + str(time_taken))



# Filter the DataFrame
bid_df = combined_df[combined_df['Type'] == 'bid']

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(bid_df['adjusted_time'], bid_df['Price'], s=10, alpha=0.2)
plt.xlabel('Adjusted Time (seconds since start)')
plt.ylabel('Price')
plt.title('Scatter Plot of Adjusted Time vs Price for Bids')
plt.show()

# Filter the DataFrame
ask_df = combined_df[combined_df['Type'] == 'ask']

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(ask_df['adjusted_time'], ask_df['Price'], s=10, alpha=0.2)
plt.xlabel('Adjusted Time (seconds since start)')
plt.ylabel('Price')
plt.title('Scatter Plot of Adjusted Time vs Price for Asks')
plt.show()

combined_df.to_csv('combined_data.csv', index=False)
bid_df.to_csv('combined_data_bids.csv', index=False)
ask_df.to_csv('combined_data_asks.csv', index=False)
