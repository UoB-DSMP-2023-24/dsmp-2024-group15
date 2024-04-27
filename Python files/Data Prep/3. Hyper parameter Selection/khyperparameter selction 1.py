# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 16:58:19 2024

@author: fearh
"""

import pandas as pd
import time
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm

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

def fit_arima(df):
    # Fit the ARIMA model (with p=1, d=0, q=0)
    model = sm.tsa.ARIMA(df, order=(1,0,0))
    results = model.fit()
    return results.aic

#%% Cell 1 Read in Data (40s)
# Replace 'your_file.csv' with the path to your CSV file
file_path = 'C:/Users/fearh/Desktop/theo pt2/17.4/combined_df.csv'
folder_path = r'C:/Users/fearh/Desktop/theo pt2/18.4/new_ints'


# Reading the CSV file into a DataFrame
combined_df = pd.read_csv(file_path)




# List of hyperparameters to test
hyperparameters = [
    (0.1, 0.1, 3600),
    (0.5, 0.5, 3600),
    (1, 1, 3600),
    (0.55, 0.45, 3600),
    (0.4, 0.6, 3600),
    (0.4, 0.5, 3600),
    (0.55, 0.55, 3600),
    (0.45, 0.6, 3600),
    (0.5, 0.6, 3600),
]

hyperparameters_1 = [
    # 1 minute (60 seconds)
    (0.1, 0.1, 60),
    (0.5, 0.5, 60),
    (1, 1, 60),
    # 15 minutes (900 seconds)
    (0.1, 0.1, 900),
    (0.5, 0.5, 900),
    (1, 1, 900),
    # 30 minutes (1800 seconds)
    (0.1, 0.1, 1800),
    (0.5, 0.5, 1800),
    (1, 1, 1800)
]

hyperparameters_2 = [
    # For the 60-second interval, try to increase n_bid and n_ask slightly
    # to see if we can retain more data without losing too many bids/asks.
    (0.6, 0.6, 60),
    (0.7, 0.7, 60),
    (0.8, 0.8, 60),
    
    # For the 900-second interval, since 0.5 was somewhat balanced, you might
    # want to try a slight increase to see the effect.
    (0.55, 0.55, 900),
    (0.6, 0.6, 900),
    
    # For the 1800-second interval, since 0.5 seemed to perform well, we can
    # experiment with values around it.
    (0.55, 0.55, 1800),
    (0.6, 0.6, 1800),
]

new_hyperparameters = [
    (x, x, duration) 
    for duration in [60, 900, 3600]
    for x in [0.1 + (i * 0.1) for i in range(10)]
]

i =1

results = []

#%% Cell 2 iteration

for n_bid, n_ask, time_int in new_hyperparameters:
    # Start time measurement
    time_start = time.time()

    # Convert 'adjusted_time' column to timedelta objects
    combined_df['adjusted_time_2'] = pd.to_timedelta(combined_df['adjusted_time'])
    # Convert the 'adjusted_time_2' column, which is a timedelta, to seconds
    combined_df['adjusted_time_seconds'] = combined_df['adjusted_time_2'].dt.total_seconds()

    # Group by one-hour intervals and 'Type'
    grouped = combined_df.groupby(combined_df['adjusted_time_seconds'] // time_int)
    
    

    # Apply the filtering function to each group
    filtered_df = grouped.apply(filter_bids_asks, n_bid=n_bid, n_ask=n_ask).reset_index(drop=True)

    # Group the filtered dataframe by hour and calculate min bids and asks
    int_grouped = filtered_df.groupby(filtered_df['adjusted_time_seconds'] // time_int)
    int_grouped_counts = int_grouped['Type'].value_counts().unstack(fill_value=0)
    int_min_bids = int_grouped_counts.get('bid', pd.Series()).min()
    int_min_asks = int_grouped_counts.get('ask', pd.Series()).min()
    int_mean_bids = int_grouped_counts.get('bid', pd.Series()).mean()
    int_mean_asks = int_grouped_counts.get('ask', pd.Series()).mean()


    

    # Data loss calculation
    len_original = len(combined_df)
    len_filtered = len(filtered_df)
    percent_lost = round((100 * (len_original - len_filtered) / len_original), 2)
    
    
    
    
    lobs = filtered_df
    # Convert 'adjusted_time_seconds' to hours
    lobs['adjusted_time_ints'] = lobs['adjusted_time_seconds'] // time_int

    # Now, group by 'Type' and 'adjusted_time_hours' and calculate the mean 'Price' for each hour.
    mean_prices_volumes = lobs.groupby(['Type', 'adjusted_time_ints'])[['Price', 'Volume']].mean().reset_index()

    mean_bid_prices_volumes = mean_prices_volumes[mean_prices_volumes['Type'] == 'bid']
    mean_ask_prices_volumes = mean_prices_volumes[mean_prices_volumes['Type'] == 'ask']
    
    bid_prices = mean_bid_prices_volumes['Price']
    ask_prices = mean_ask_prices_volumes['Price']
    
    # Calculate AIC for both bid and ask prices
    bid_aic = fit_arima(bid_prices)
    ask_aic = fit_arima(ask_prices)
    
    # Store results
    results.append({
        'n_bid': n_bid,
        'n_ask': n_ask,
        'time interval': time_int,
        'percent_lost': percent_lost,
        'min_bids': int_min_bids,
        'min_asks': int_min_asks,
        'mean_bids/int': int_mean_bids,
        'min_asks/int': int_mean_asks,
        'Bid_AIC': bid_aic,
        'Ask_AIC': ask_aic
        
    })
    
    
    time_taken = round(time.time() - time_start, 2)
    print(f'Time taken round {i}: {time_taken}')
    i += 1

# Print all results
for result in results:
    print(result)
    # Convert the dictionary to a pandas DataFrame
df = pd.DataFrame(results)


#%% Cell 2.2
# Convert the DataFrame to an Excel file named 'hyperparamtable.xlsx'
file_path = r'C:/Users/fearh/Desktop/theo pt2/18.4/new_ints/hyperparamtable 3x10.xlsx'

# Save the DataFrame to an Excel file
df.to_excel(file_path, index=False)


#%% Cell 3 ARIMA prep

time_start_2 = time.time()




#%% Cell 2.1 unimportant
    min_bids_group = int(hourly_grouped_counts['bid'].idxmin())
    min_asks_group = int(hourly_grouped_counts['ask'].idxmin())

    min_bids_data = filtered_df[(filtered_df['adjusted_time_seconds'] // time_int == min_bids_group) & (filtered_df['Type'] == 'bid')]
    min_asks_data = filtered_df[(filtered_df['adjusted_time_seconds'] // time_int == min_asks_group) & (filtered_df['Type'] == 'ask')]

    # Plotting and saving the plots
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(min_bids_data['Price'], bins=20, color='blue', alpha=0.7)
    plt.title(f'Price Distribution for Group {min_bids_group} (Bids)')
    plt.xlabel('Price')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.hist(min_asks_data['Price'], bins=20, color='orange', alpha=0.7)
    plt.title(f'Price Distribution for Group {min_asks_group} (Asks)')
    plt.xlabel('Price')

    plt.tight_layout()

    # Generate a unique filename for each plot
    plot_filename = f'plot_n_bid_{n_bid}_n_ask_{n_ask}_time_{time_int}s.png'
    full_path = os.path.join(folder_path, plot_filename)

    # Save the figure to the specified path
    plt.savefig(full_path)
    plt.close()  # Close the plot to free up memory
    
    time_taken = round(time.time() - time_start, 2)
    print(f'Time taken round {i}: {time_taken}')
    i += 1

# Print all results
for result in results:
    print(result)
    # Convert the dictionary to a pandas DataFrame
df = pd.DataFrame(results)

