#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 22:40:45 2024

@author: theorogers
"""
#%% Cell 1, adjusting time. Takes less than a minute
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

#%% Cell 2, plottings

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


#%% Cell 3.1, filtering further (on a minutely basis) (can take about 10 mins)

cell_3_start = time.time()

from tqdm import tqdm

combined_df.set_index('adjusted_time', inplace=True)
# Group by one-minute intervals and 'Type'
grouped = combined_df.groupby([pd.Grouper(freq='T'), 'Type'])

# Function to filter within one std from the max bid and min ask
def filter_bids_asks(group):
    if group.name[1] == 'bid':
        max_price = group['Price'].max()
        return group[group['Price'] >= max_price - group['Price'].std()]
    elif group.name[1] == 'ask':
        min_price = group['Price'].min()
        return group[group['Price'] <= min_price + group['Price'].std()]
    return group  # In case there are types other than 'bid' or 'ask'

# Apply the filtering function to each group
filtered_df = grouped.apply(filter_bids_asks).reset_index(drop=True)

time_taken = time.time() - cell_3_start
print("Cell 3 time taken: " + str(time_taken))

#%% Cell 3.2, plotting the filtering

# Filter the DataFrame
bid_filt_df = filtered_df[filtered_df['Type'] == 'bid']

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(bid_filt_df['adjusted_time'], bid_filt_df['Price'], s=10, alpha=0.2)
plt.xlabel('Adjusted Time (seconds since start)')
plt.ylabel('Price')
plt.title('Scatter Plot of Adjusted Time vs Price for Bids')
plt.show()

# Filter the DataFrame
ask_filt_df = filtered_df[filtered_df['Type'] == 'ask']

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(ask_filt_df['adjusted_time'], ask_filt_df['Price'], s=10, alpha=0.2)
plt.xlabel('Adjusted Time (seconds since start)')
plt.ylabel('Price')
plt.title('Scatter Plot of Adjusted Time vs Price for Asks')
plt.show()

#%% Cell 4.1, filtering further (on a hourly basis) (run cell 1 before running this cell)
#Takes 10s or so


cell_4_start = time.time()

n = 10

combined_df.set_index('adjusted_time', inplace=True)
# Group by one-minute intervals and 'Type'
grouped = combined_df.groupby([pd.Grouper(freq='H'), 'Type'])

# Function to filter within one std from the max bid and min ask
def filter_bids_asks(group):
    if group.name[1] == 'bid':
        max_price = group['Price'].max()
        return group[group['Price'] >= max_price - n*group['Price'].std()]
    elif group.name[1] == 'ask':
        min_price = group['Price'].min()
        return group[group['Price'] <= min_price + n*group['Price'].std()]
    return group  # In case there are types other than 'bid' or 'ask'

# Apply the filtering function to each group
filtered_df = grouped.apply(filter_bids_asks).reset_index(drop=True)

time_taken = time.time() - cell_4_start
print("Cell 4 time taken: " + str(time_taken))



#%% Cell 4.2, plotting the filtering

# Filter the DataFrame
bid_filt_h_df = filtered_df[filtered_df['Type'] == 'bid']

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(bid_filt_h_df['adjusted_time'], bid_filt_h_df['Price'], s=10, alpha=0.2)
plt.xlabel('Adjusted Time (seconds since start)')
plt.ylabel('Price')
plt.title('Scatter Plot of Adjusted Time vs Price for Bids')
plt.show()

# Filter the DataFrame
ask_filt_h_df = filtered_df[filtered_df['Type'] == 'ask']

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(ask_filt_h_df['adjusted_time'], ask_filt_h_df['Price'], s=10, alpha=0.2)
plt.xlabel('Adjusted Time (seconds since start)')
plt.ylabel('Price')
plt.title('Scatter Plot of Adjusted Time vs Price for Asks')
plt.show()

#%% Cell 5 saving df
import pandas as pd
from tqdm import tqdm

# Assuming combined_df is a pandas DataFrame
output_file = 'combined_df_2.csv'  # Name of the output CSV file

chunk_size = 1000  # Size of each chunk for processing
number_of_chunks = len(combined_df) // chunk_size + (1 if len(combined_df) % chunk_size else 0)  # Total number of chunks

# Open a file in write mode
with open(output_file, 'w', newline='', encoding='utf-8') as file:
    # If you want to include headers
    combined_df.iloc[:0].to_csv(file, header=True, index=False)

    # Iterate through the DataFrame in chunks
    for i in tqdm(range(number_of_chunks), desc='Saving CSV'):
        start_index = i * chunk_size
        end_index = start_index + chunk_size

        # Get the chunk of the DataFrame
        chunk = combined_df.iloc[start_index:end_index]

        # Write the chunk to the CSV file, without headers
        chunk.to_csv(file, mode='a', header=False, index=False)

print("CSV file has been created.")
#%% Cell 5.2 saving df
filtered_df.to_csv('/Users/theorogers/Desktop/filtered_df.csv', index=False)

