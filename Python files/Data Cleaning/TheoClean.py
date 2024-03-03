#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 11:33:39 2024

@author: theorogers
"""
import os

#Current working directory
script_dir = os.getcwd()

#Data file path, relative to CWD
file_path = os.path.join(script_dir, '..', '..', 'Prompt', 'Data', 'UoB_Set01_2025-01-07LOBs.txt')



import pandas as pd
import ast





#Function to process a line of the LOB
def process_line(line):
    #This allows the string to pass through as literal
    line = line.replace("Exch0", "'Exch0'") 
    #Converting the string as literal
    data = ast.literal_eval(line)
    
    #Splitting up the parts of the string
    time_stamp = data[0]
    bids_asks = data[2]
    
    #Empty list to append to a table
    temp_row = []
    
    #Iterate through the bids and asks string, apart of the string
    for bid_ask in bids_asks:
        type_ = bid_ask[0]  #'bid' or 'ask'
        for entry in bid_ask[1]:
            price = entry[0]
            volume = entry[1]
            #Creating a the string to add to the dataframe
            temp_row.append([time_stamp, type_, price, volume])
    
    # Convert the list to a DataFrame
    return pd.DataFrame(temp_row, columns=['Time', 'Type', 'Price', 'Volume'])



#Open the file and process each line
with open(file_path, 'r') as file:
    lines = file.read().splitlines()
    
def process_file(lines):
    #Creating an empty dataframe
    df = pd.DataFrame() 
    #Initialising a counter which counts how many lines have been processed
    count = 0 
    
    #Process each line and append to the DataFrame
    for line in lines:
        
        count += 1
        if count%1000 == 0: #Every 1000 lines,it reports progress
            print(str(round(count*100/len(lines),2)) + "%") #Prints percent of lines finished 
    
        #Process the line and return a row in the df
        line_df = process_line(line.strip())
        
        #Append the row to the main df
        df = pd.concat([df, line_df], ignore_index=True)
    return(df)

#Process and display the first file
df = process_file(lines)
df.head(10)
df.to_csv("LOB_sorted.csv")

