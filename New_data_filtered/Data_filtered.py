import os
import pandas as pd
import ast
import time

start_time = time.time()

# Function to process a line of the LOB
def process_line(line):
    line = line.replace("Exch0", "'Exch0'")
    data = ast.literal_eval(line)
    time_stamp = data[0]
    bids_asks = data[2]
    temp_rows = [
        [time_stamp, bid_ask[0], entry[0], entry[1]]
        for bid_ask in bids_asks
        for entry in bid_ask[1]
    ]
    return temp_rows


# Function to process a file and filter bids/asks
def process_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.read().splitlines()

    all_rows = [row for line in lines for row in process_line(line.strip())]
    df = pd.DataFrame(all_rows, columns=['Time', 'Type', 'Price', 'Volume'])

    df['Original_Order'] = df.index
    df['Time_Segment'] = df['Time'].astype(int)

    def filter_bids_asks(group):
        top_bids = group[group['Type'] == 'bid'].nlargest(2, 'Price')
        bottom_asks = group[group['Type'] == 'ask'].nsmallest(2, 'Price')
        return pd.concat([top_bids, bottom_asks])

    filtered_df = df.groupby('Time_Segment').apply(filter_bids_asks).reset_index(drop=True)
    filtered_sorted_df = filtered_df.sort_values(by='Original_Order').reset_index(drop=True)
    filtered_sorted_df.drop(['Time_Segment', 'Original_Order'], axis=1, inplace=True)

    return filtered_sorted_df


# Directory processing
script_dir = os.getcwd()
for file_name in os.listdir(script_dir):
    if file_name.endswith('.txt'):
        print(f"Processing {file_name}...")
        file_path = os.path.join(script_dir, file_name)
        df_filtered_sorted = process_file(file_path)

        # Add an index column starting from 1 at the beginning
        df_filtered_sorted.insert(0, 'Index', range(1, len(df_filtered_sorted) + 1))

        output_file_name_filtered = os.path.splitext(file_name)[0] + "_LOB_sorted_filtered.csv"
        df_filtered_sorted.to_csv(os.path.join(script_dir, output_file_name_filtered), index=False)
        print(f"Filtered and sorted data saved to {output_file_name_filtered}")

end_time = time.time()

print("Total time:", end_time - start_time, "seconds")
