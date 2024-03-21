import os
import pandas as pd
import ast
import time


def process_line_optimized(line):
    line = line.replace("Exch0", "'Exch0'")
    data = ast.literal_eval(line)
    time_stamp, bids_asks = data[0], data[2]
    for bid_ask in bids_asks:
        type_ = bid_ask[0]
        for price, volume in bid_ask[1]:
            yield time_stamp, type_, price, volume


def process_file_optimized(file_path):
    lines_gen = (line.strip() for line in open(file_path, 'r'))
    rows_gen = (row for line in lines_gen for row in process_line_optimized(line))
    return pd.DataFrame(rows_gen, columns=['Time', 'Type', 'Price', 'Volume'])


def process_data_by_second(df):
    df.sort_values(by='Time', inplace=True)
    df['TimeGroup'] = df['Time'].astype(int)
    result_df = pd.DataFrame(columns=df.columns)

    grouped = df.groupby('TimeGroup')
    for _, group in grouped:
        ask_row = group[group['Type'] == 'ask'].head(1)
        bid_row = group[group['Type'] == 'bid'].head(1)
        result_df = pd.concat([result_df, ask_row, bid_row], ignore_index=True)

    result_df.drop(columns=['TimeGroup'], inplace=True)
    return result_df


def main_process(directory):
    start_time = time.time()  # Start time

    for file_name in os.listdir(directory):
        if file_name.endswith('.txt'):
            print(f"Processing {file_name}...")
            file_path = os.path.join(directory, file_name)
            df = process_file_optimized(file_path)

            # Save the original data converted to CSV
            base_name = os.path.splitext(file_name)[0]
            raw_csv_file_name = f"{base_name}_raw.csv"
            df.to_csv(os.path.join(directory, raw_csv_file_name), index=False)
            print(f"Raw data saved to {raw_csv_file_name}")

            # Filter data by second and save
            processed_df = process_data_by_second(df)
            filtered_csv_file_name = f"{base_name}_filtered.csv"
            processed_df.to_csv(os.path.join(directory, filtered_csv_file_name), index=False)
            print(f"Filtered data saved to {filtered_csv_file_name}")

    end_time = time.time()
    print(f"Total processing time: {end_time - start_time} seconds")  # Display the total processing time


directory = "/Users/Desktop/01-08LOBs"
main_process(directory)

