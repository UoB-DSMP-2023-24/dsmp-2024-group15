# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 16:58:19 2024

@author: theo-rogers
"""



#%% Cell 1 Read in Data, create functions and import packages (40s)

import pandas as pd
import time
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm

#Defining the function to filter bids and asks
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
    #Fit an ARIMA(1) model
    model = sm.tsa.ARIMA(df, order=(1,0,0))
    results = model.fit()
    #Returning AIC
    return results.aic



file_path = 'C:/Users/fearh/Desktop/theo pt2/17.4/combined_df.csv'
folder_path = r'C:/Users/fearh/Desktop/theo pt2/18.4/new_ints'


#Reading the CSV file into a DataFrame
combined_df = pd.read_csv(file_path)


# List of hyperparameters that could be test
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

    (0.1, 0.1, 60),
    (0.5, 0.5, 60),
    (1, 1, 60),
    (0.1, 0.1, 900),
    (0.5, 0.5, 900),
    (1, 1, 900),
    (0.1, 0.1, 1800),
    (0.5, 0.5, 1800),
    (1, 1, 1800)
]

hyperparameters_2 = [
    (0.6, 0.6, 60),
    (0.7, 0.7, 60),
    (0.8, 0.8, 60),
    (0.55, 0.55, 900),
    (0.6, 0.6, 900),
    (0.55, 0.55, 1800),
    (0.6, 0.6, 1800),
]

new_hyperparameters = [
    (x, x, duration) 
    for duration in [60, 900, 3600]
    for x in [0.1 + (i * 0.1) for i in range(10)]
]

results = []

#%% Cell 2 Testing hyper parameter values (takes 3minutes per iteration)

i =1
for n_bid, n_ask, time_int in new_hyperparameters:
    
    
    time_start = time.time()

    #Converting 'adjusted_time' column to timedelta objects
    combined_df['adjusted_time_2'] = pd.to_timedelta(combined_df['adjusted_time'])
    #Converting the 'adjusted_time_2' column, which is a timedelta, to seconds
    combined_df['adjusted_time_seconds'] = combined_df['adjusted_time_2'].dt.total_seconds()

    #Grouping by  time interval
    grouped = combined_df.groupby(combined_df['adjusted_time_seconds'] // time_int)
    
    #Applying the filtering function to each group
    filtered_df = grouped.apply(filter_bids_asks, n_bid=n_bid, n_ask=n_ask).reset_index(drop=True)

    #Grouping the filtered dataframe by interval and calculate min bids and asks
    int_grouped = filtered_df.groupby(filtered_df['adjusted_time_seconds'] // time_int)
    int_grouped_counts = int_grouped['Type'].value_counts().unstack(fill_value=0)
    int_min_bids = int_grouped_counts.get('bid', pd.Series()).min()
    int_min_asks = int_grouped_counts.get('ask', pd.Series()).min()
    int_mean_bids = int_grouped_counts.get('bid', pd.Series()).mean()
    int_mean_asks = int_grouped_counts.get('ask', pd.Series()).mean()

    #Data loss calculation
    len_original = len(combined_df)
    len_filtered = len(filtered_df)
    percent_lost = round((100 * (len_original - len_filtered) / len_original), 2)

    #Taking the mean to get one data point per time interval (so ARIMA and other models work)
    lobs = filtered_df
    #Converting 'adjusted_time_seconds' to hours
    lobs['adjusted_time_ints'] = lobs['adjusted_time_seconds'] // time_int

    #Same as before
    mean_prices_volumes = lobs.groupby(['Type', 'adjusted_time_ints'])[['Price', 'Volume']].mean().reset_index()

    mean_bid_prices_volumes = mean_prices_volumes[mean_prices_volumes['Type'] == 'bid']
    mean_ask_prices_volumes = mean_prices_volumes[mean_prices_volumes['Type'] == 'ask']
    
    bid_prices = mean_bid_prices_volumes['Price']
    ask_prices = mean_ask_prices_volumes['Price']
    
    #Calculate AIC for both bid and ask prices for an ARIMA(1) model
    bid_aic = fit_arima(bid_prices)
    ask_aic = fit_arima(ask_prices)
    
    #Saving results
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


#Converting the dictionary to a pandas DataFrame
df = pd.DataFrame(results)


#%% Cell 2.2
#Convert the DataFrame to an Excel file named 'hyperparamtable.xlsx'
file_path = r'C:/Users/fearh/Desktop/theo pt2/18.4/new_ints/hyperparamtable 3x10.xlsx'

#Save the DataFrame to an Excel file
df.to_excel(file_path, index=False)



#%% Cell 3, unimportant

#This cell is more an archive of the code to visualise the distribution of the smallest group, 
    min_bids_group = int(hourly_grouped_counts['bid'].idxmin())
    min_asks_group = int(hourly_grouped_counts['ask'].idxmin())

    min_bids_data = filtered_df[(filtered_df['adjusted_time_seconds'] // time_int == min_bids_group) & (filtered_df['Type'] == 'bid')]
    min_asks_data = filtered_df[(filtered_df['adjusted_time_seconds'] // time_int == min_asks_group) & (filtered_df['Type'] == 'ask')]

    #Plotting and saving the plots
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

    #Generating a unique filename for each plot
    plot_filename = f'plot_n_bid_{n_bid}_n_ask_{n_ask}_time_{time_int}s.png'
    full_path = os.path.join(folder_path, plot_filename)

    #Saving the figure to the specified path
    #plt.savefig(full_path)
    plt.close()
    
    time_taken = round(time.time() - time_start, 2)
    print(f'Time taken round {i}: {time_taken}')
    i += 1

#Printing all results
for result in results:
    print(result)
    # Convert the dictionary to a pandas DataFrame
df = pd.DataFrame(results)

