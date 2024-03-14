import os
import pandas as pd

# indicate a folder path
folder_path = 'C:\\Users\\s2420485\\Desktop\\dsmp-2024-group15-main\\Prompt\\Data\\Chunyu merged file new'

# Initializes an empty DataFrame to store the merged data
merged_data = pd.DataFrame()

# Cycle through each CSV file
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        # Extract date information from a file name (assuming that the file name contains date information)
        date = filename[10:20]  # find the position of date information in the filename, like '2025-01-08'
        # Reading CSV file
        data = pd.read_csv(file_path)
        # Add date column
        data['Date'] = date
        # Merge the data into a DataFrame
        merged_data = pd.concat([merged_data, data], ignore_index=True)

# Save the merged data to a new CSV file in the specified path
output_file_path = os.path.join(folder_path, 'merged_with_date.csv')
merged_data.to_csv(output_file_path, index=False)

# In fact, the csv file can only store no more than 1048572 lines of data, 
# so the txt file is used to display all the data
output_txt_file_path = os.path.join(folder_path, 'merged_data.txt')
merged_data.to_csv(output_txt_file_path, sep='\t', index=False)

print("The merge is complete and saved to the {} file.".format(output_file_path))
print("The CSV file has been converted to a TXT file and saved to the {} file.".format(output_txt_file_path))







