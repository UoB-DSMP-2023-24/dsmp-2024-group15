#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 00:48:00 2024

@author: theorogers
"""
import pandas as pd
path_file = "/Users/theorogers/Desktop/filtered_data_mime0101 1.csv" 
df = pd.read_csv(path_file)


# Calculate the price range for 'bids'
bids_range = df[df['Type'] == 'bid']['Price'].max() - df[df['Type'] == 'bid']['Price'].min()

# Calculate the price range for 'asks'
asks_range = df[df['Type'] == 'ask']['Price'].max() - df[df['Type'] == 'ask']['Price'].min()

# Calculate the overall price range
overall_range = df['Price'].max() - df['Price'].min()

# Print the ranges
print("Price range for 'bids':", bids_range)
print("Price range for 'asks':", asks_range)
print("Overall price range:", overall_range)