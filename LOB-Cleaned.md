```python
import pandas as pd

# Path to your CSV file
file_path = "/Users/tanmaysagarhuria/Downloads/lobcsvformat.csv"

# Read the CSV file into a pandas DataFrame
lob = pd.read_csv(file_path)

# Now you can work with your DataFrame (df)
# For example, you can print the first few rows
print(lob.head())
```

                                                 Column1
    0         [0.000, Exch0, [['bid', []], ['ask', []]]]
    1   [0.279, Exch0, [['bid', [[1, 6]]], ['ask', []]]]
    2  [1.333, Exch0, [['bid', [[1, 6]]], ['ask', [[8...
    3  [1.581, Exch0, [['bid', [[1, 6]]], ['ask', [[7...
    4  [1.643, Exch0, [['bid', [[1, 6]]], ['ask', [[7...



```python

```


```python
lob
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[0.000, Exch0, [['bid', []], ['ask', []]]]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[0.279, Exch0, [['bid', [[1, 6]]], ['ask', []]]]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[1.333, Exch0, [['bid', [[1, 6]]], ['ask', [[8...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[1.581, Exch0, [['bid', [[1, 6]]], ['ask', [[7...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[1.643, Exch0, [['bid', [[1, 6]]], ['ask', [[7...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>352965</th>
      <td>[30599.542, Exch0, [['bid', [[290, 1], [288, 2...</td>
    </tr>
    <tr>
      <th>352966</th>
      <td>[30599.573, Exch0, [['bid', [[290, 1], [288, 2...</td>
    </tr>
    <tr>
      <th>352967</th>
      <td>[30599.635, Exch0, [['bid', [[290, 1], [288, 2...</td>
    </tr>
    <tr>
      <th>352968</th>
      <td>[30599.759, Exch0, [['bid', [[288, 1], [281, 4...</td>
    </tr>
    <tr>
      <th>352969</th>
      <td>[30599.945, Exch0, [['bid', [[288, 1], [281, 4...</td>
    </tr>
  </tbody>
</table>
<p>352970 rows × 1 columns</p>
</div>




```python
import pandas as pd

# Function to split the complex column into structured data
def split_columns(row):
    try:
        # Strip leading/trailing brackets and split by commas and brackets where applicable
        parts = row.strip('[]').split(', ', 2)  # Split into 3 parts: timestamp, exchange, and the bid/ask data
        timestamp = parts[0]
        exchange = parts[1].strip('\'\"')  # Remove quotes
        bid_ask_data = parts[2]
        return pd.Series([timestamp, exchange, bid_ask_data])
    except Exception as e:
        # In case of any error, return None or some error indicator
        return pd.Series([None, None, None])
```


```python
lob[['Timestamp', 'Exchange', 'BidAskData']] = lob['Column1'].apply(split_columns)
```


```python
print(lob.head())
```

                                                 Column1 Timestamp Exchange  \
    0         [0.000, Exch0, [['bid', []], ['ask', []]]]     0.000    Exch0   
    1   [0.279, Exch0, [['bid', [[1, 6]]], ['ask', []]]]     0.279    Exch0   
    2  [1.333, Exch0, [['bid', [[1, 6]]], ['ask', [[8...     1.333    Exch0   
    3  [1.581, Exch0, [['bid', [[1, 6]]], ['ask', [[7...     1.581    Exch0   
    4  [1.643, Exch0, [['bid', [[1, 6]]], ['ask', [[7...     1.643    Exch0   
    
                                 BidAskData  
    0                [['bid', []], ['ask',   
    1          [['bid', [[1, 6]]], ['ask',   
    2  [['bid', [[1, 6]]], ['ask', [[800, 1  
    3  [['bid', [[1, 6]]], ['ask', [[799, 1  
    4  [['bid', [[1, 6]]], ['ask', [[798, 1  



```python
lob
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column1</th>
      <th>Timestamp</th>
      <th>Exchange</th>
      <th>BidAskData</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[0.000, Exch0, [['bid', []], ['ask', []]]]</td>
      <td>0.000</td>
      <td>Exch0</td>
      <td>[['bid', []], ['ask',</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[0.279, Exch0, [['bid', [[1, 6]]], ['ask', []]]]</td>
      <td>0.279</td>
      <td>Exch0</td>
      <td>[['bid', [[1, 6]]], ['ask',</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[1.333, Exch0, [['bid', [[1, 6]]], ['ask', [[8...</td>
      <td>1.333</td>
      <td>Exch0</td>
      <td>[['bid', [[1, 6]]], ['ask', [[800, 1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[1.581, Exch0, [['bid', [[1, 6]]], ['ask', [[7...</td>
      <td>1.581</td>
      <td>Exch0</td>
      <td>[['bid', [[1, 6]]], ['ask', [[799, 1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[1.643, Exch0, [['bid', [[1, 6]]], ['ask', [[7...</td>
      <td>1.643</td>
      <td>Exch0</td>
      <td>[['bid', [[1, 6]]], ['ask', [[798, 1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>352965</th>
      <td>[30599.542, Exch0, [['bid', [[290, 1], [288, 2...</td>
      <td>30599.542</td>
      <td>Exch0</td>
      <td>[['bid', [[290, 1], [288, 2], [281, 4], [278, ...</td>
    </tr>
    <tr>
      <th>352966</th>
      <td>[30599.573, Exch0, [['bid', [[290, 1], [288, 2...</td>
      <td>30599.573</td>
      <td>Exch0</td>
      <td>[['bid', [[290, 1], [288, 2], [281, 4], [278, ...</td>
    </tr>
    <tr>
      <th>352967</th>
      <td>[30599.635, Exch0, [['bid', [[290, 1], [288, 2...</td>
      <td>30599.635</td>
      <td>Exch0</td>
      <td>[['bid', [[290, 1], [288, 2], [281, 4], [278, ...</td>
    </tr>
    <tr>
      <th>352968</th>
      <td>[30599.759, Exch0, [['bid', [[288, 1], [281, 4...</td>
      <td>30599.759</td>
      <td>Exch0</td>
      <td>[['bid', [[288, 1], [281, 4], [278, 1], [270, ...</td>
    </tr>
    <tr>
      <th>352969</th>
      <td>[30599.945, Exch0, [['bid', [[288, 1], [281, 4...</td>
      <td>30599.945</td>
      <td>Exch0</td>
      <td>[['bid', [[288, 1], [281, 4], [278, 1], [270, ...</td>
    </tr>
  </tbody>
</table>
<p>352970 rows × 4 columns</p>
</div>




```python
lob.drop('Column1', axis=1, inplace=True)
```


```python
lob
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Timestamp</th>
      <th>Exchange</th>
      <th>BidAskData</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000</td>
      <td>Exch0</td>
      <td>[['bid', []], ['ask',</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.279</td>
      <td>Exch0</td>
      <td>[['bid', [[1, 6]]], ['ask',</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.333</td>
      <td>Exch0</td>
      <td>[['bid', [[1, 6]]], ['ask', [[800, 1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.581</td>
      <td>Exch0</td>
      <td>[['bid', [[1, 6]]], ['ask', [[799, 1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.643</td>
      <td>Exch0</td>
      <td>[['bid', [[1, 6]]], ['ask', [[798, 1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>352965</th>
      <td>30599.542</td>
      <td>Exch0</td>
      <td>[['bid', [[290, 1], [288, 2], [281, 4], [278, ...</td>
    </tr>
    <tr>
      <th>352966</th>
      <td>30599.573</td>
      <td>Exch0</td>
      <td>[['bid', [[290, 1], [288, 2], [281, 4], [278, ...</td>
    </tr>
    <tr>
      <th>352967</th>
      <td>30599.635</td>
      <td>Exch0</td>
      <td>[['bid', [[290, 1], [288, 2], [281, 4], [278, ...</td>
    </tr>
    <tr>
      <th>352968</th>
      <td>30599.759</td>
      <td>Exch0</td>
      <td>[['bid', [[288, 1], [281, 4], [278, 1], [270, ...</td>
    </tr>
    <tr>
      <th>352969</th>
      <td>30599.945</td>
      <td>Exch0</td>
      <td>[['bid', [[288, 1], [281, 4], [278, 1], [270, ...</td>
    </tr>
  </tbody>
</table>
<p>352970 rows × 3 columns</p>
</div>




```python
import ast

# Function to safely evaluate the string representation of a list
def safe_eval_list(data_str):
    try:
        return ast.literal_eval(data_str)
    except ValueError:
        return None

# Function to extract bid and ask data
def extract_bid_ask(bid_ask_data_str):
    bid_ask_data = safe_eval_list(bid_ask_data_str)
    if bid_ask_data:
        bid_data = bid_ask_data[0][1] if len(bid_ask_data) > 0 else []
        ask_data = bid_ask_data[1][1] if len(bid_ask_data) > 1 else []
        return bid_data, ask_data
    else:
        return [], []
```


```python
lob['BidData'], lob['AskData'] = zip(*lob['BidAskData'].apply(extract_bid_ask))
```


    Traceback (most recent call last):


      File ~/anaconda3/lib/python3.11/site-packages/IPython/core/interactiveshell.py:3505 in run_code
        exec(code_obj, self.user_global_ns, self.user_ns)


      Cell In[18], line 1
        lob['BidData'], lob['AskData'] = zip(*lob['BidAskData'].apply(extract_bid_ask))


      File ~/anaconda3/lib/python3.11/site-packages/pandas/core/series.py:4771 in apply
        return SeriesApply(self, func, convert_dtype, args, kwargs).apply()


      File ~/anaconda3/lib/python3.11/site-packages/pandas/core/apply.py:1123 in apply
        return self.apply_standard()


      File ~/anaconda3/lib/python3.11/site-packages/pandas/core/apply.py:1174 in apply_standard
        mapped = lib.map_infer(


      File pandas/_libs/lib.pyx:2924 in pandas._libs.lib.map_infer


      Cell In[15], line 12 in extract_bid_ask
        bid_ask_data = safe_eval_list(bid_ask_data_str)


      Cell In[15], line 6 in safe_eval_list
        return ast.literal_eval(data_str)


      File ~/anaconda3/lib/python3.11/ast.py:64 in literal_eval
        node_or_string = parse(node_or_string.lstrip(" \t"), mode='eval')


      File ~/anaconda3/lib/python3.11/ast.py:50 in parse
        return compile(source, filename, mode, flags,


      File <unknown>:1
        [['bid', []], ['ask',
                      ^
    SyntaxError: '[' was never closed




```python
# Function to manually parse the bid and ask data
def parse_bid_ask_data(row):
    # Remove the brackets at the beginning and end, and then split the string
    parts = row.strip('][').split('], [')
    bid_part = parts[0] if len(parts) > 0 else ''
    ask_part = parts[1] if len(parts) > 1 else ''
    
    # Further processing to extract only the numeric values if necessary
    bid_data = bid_part.replace('[', '').replace(']', '').replace('\'', '').replace('bid', '').replace(',', '').split()
    ask_data = ask_part.replace('[', '').replace(']', '').replace('\'', '').replace('ask', '').replace(',', '').split()
    
    # Convert to list of lists [[price, size], [price, size], ...]
    bid_data = [bid_data[i:i+2] for i in range(0, len(bid_data), 2)]
    ask_data = [ask_data[i:i+2] for i in range(0, len(ask_data), 2)]
    
    return bid_data, ask_data

# Apply the parsing function to each row and create new columns for bid and ask data
lob[['BidData', 'AskData']] = lob.apply(lambda row: pd.Series(parse_bid_ask_data(row['BidAskData'])), axis=1)

# Now, display the updated dataframe to confirm the changes
print(lob.head())

```

      Timestamp Exchange                            BidAskData   BidData  \
    0     0.000    Exch0                [['bid', []], ['ask',         []   
    1     0.279    Exch0          [['bid', [[1, 6]]], ['ask',   [[1, 6]]   
    2     1.333    Exch0  [['bid', [[1, 6]]], ['ask', [[800, 1  [[1, 6]]   
    3     1.581    Exch0  [['bid', [[1, 6]]], ['ask', [[799, 1  [[1, 6]]   
    4     1.643    Exch0  [['bid', [[1, 6]]], ['ask', [[798, 1  [[1, 6]]   
    
          AskData  
    0          []  
    1          []  
    2  [[800, 1]]  
    3  [[799, 1]]  
    4  [[798, 1]]  



```python
lob
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Timestamp</th>
      <th>Exchange</th>
      <th>BidAskData</th>
      <th>BidData</th>
      <th>AskData</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000</td>
      <td>Exch0</td>
      <td>[['bid', []], ['ask',</td>
      <td>[]</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.279</td>
      <td>Exch0</td>
      <td>[['bid', [[1, 6]]], ['ask',</td>
      <td>[[1, 6]]</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.333</td>
      <td>Exch0</td>
      <td>[['bid', [[1, 6]]], ['ask', [[800, 1</td>
      <td>[[1, 6]]</td>
      <td>[[800, 1]]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.581</td>
      <td>Exch0</td>
      <td>[['bid', [[1, 6]]], ['ask', [[799, 1</td>
      <td>[[1, 6]]</td>
      <td>[[799, 1]]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.643</td>
      <td>Exch0</td>
      <td>[['bid', [[1, 6]]], ['ask', [[798, 1</td>
      <td>[[1, 6]]</td>
      <td>[[798, 1]]</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>352965</th>
      <td>30599.542</td>
      <td>Exch0</td>
      <td>[['bid', [[290, 1], [288, 2], [281, 4], [278, ...</td>
      <td>[[290, 1]]</td>
      <td>[[288, 2]]</td>
    </tr>
    <tr>
      <th>352966</th>
      <td>30599.573</td>
      <td>Exch0</td>
      <td>[['bid', [[290, 1], [288, 2], [281, 4], [278, ...</td>
      <td>[[290, 1]]</td>
      <td>[[288, 2]]</td>
    </tr>
    <tr>
      <th>352967</th>
      <td>30599.635</td>
      <td>Exch0</td>
      <td>[['bid', [[290, 1], [288, 2], [281, 4], [278, ...</td>
      <td>[[290, 1]]</td>
      <td>[[288, 2]]</td>
    </tr>
    <tr>
      <th>352968</th>
      <td>30599.759</td>
      <td>Exch0</td>
      <td>[['bid', [[288, 1], [281, 4], [278, 1], [270, ...</td>
      <td>[[288, 1]]</td>
      <td>[[281, 4]]</td>
    </tr>
    <tr>
      <th>352969</th>
      <td>30599.945</td>
      <td>Exch0</td>
      <td>[['bid', [[288, 1], [281, 4], [278, 1], [270, ...</td>
      <td>[[288, 1]]</td>
      <td>[[281, 4]]</td>
    </tr>
  </tbody>
</table>
<p>352970 rows × 5 columns</p>
</div>




```python
print(lob)
```

            Timestamp Exchange                                         BidAskData  \
    0           0.000    Exch0                             [['bid', []], ['ask',    
    1           0.279    Exch0                       [['bid', [[1, 6]]], ['ask',    
    2           1.333    Exch0               [['bid', [[1, 6]]], ['ask', [[800, 1   
    3           1.581    Exch0               [['bid', [[1, 6]]], ['ask', [[799, 1   
    4           1.643    Exch0               [['bid', [[1, 6]]], ['ask', [[798, 1   
    ...           ...      ...                                                ...   
    352965  30599.542    Exch0  [['bid', [[290, 1], [288, 2], [281, 4], [278, ...   
    352966  30599.573    Exch0  [['bid', [[290, 1], [288, 2], [281, 4], [278, ...   
    352967  30599.635    Exch0  [['bid', [[290, 1], [288, 2], [281, 4], [278, ...   
    352968  30599.759    Exch0  [['bid', [[288, 1], [281, 4], [278, 1], [270, ...   
    352969  30599.945    Exch0  [['bid', [[288, 1], [281, 4], [278, 1], [270, ...   
    
               BidData     AskData  
    0               []          []  
    1         [[1, 6]]          []  
    2         [[1, 6]]  [[800, 1]]  
    3         [[1, 6]]  [[799, 1]]  
    4         [[1, 6]]  [[798, 1]]  
    ...            ...         ...  
    352965  [[290, 1]]  [[288, 2]]  
    352966  [[290, 1]]  [[288, 2]]  
    352967  [[290, 1]]  [[288, 2]]  
    352968  [[288, 1]]  [[281, 4]]  
    352969  [[288, 1]]  [[281, 4]]  
    
    [352970 rows x 5 columns]



```python
# First, let's explode the 'BidData' list into separate rows
bids_exploded = lob.explode('BidData')

# Now, let's create separate 'BidPrice' and 'BidQuantity' columns
# We'll handle cases where 'BidData' might be empty or None
def get_bid_price(bid_data):
    return bid_data[0] if isinstance(bid_data, list) and len(bid_data) > 0 else None

def get_bid_quantity(bid_data):
    return bid_data[1] if isinstance(bid_data, list) and len(bid_data) > 1 else None

bids_exploded['BidPrice'] = bids_exploded['BidData'].apply(get_bid_price)
bids_exploded['BidQuantity'] = bids_exploded['BidData'].apply(get_bid_quantity)

# Now, we no longer need the 'BidData' column
bids_exploded.drop('BidData', axis=1, inplace=True)

# Display the updated DataFrame
print(bids_exploded[['Timestamp', 'Exchange', 'BidPrice', 'BidQuantity']].head())


```

      Timestamp Exchange BidPrice BidQuantity
    0     0.000    Exch0     None        None
    1     0.279    Exch0        1           6
    2     1.333    Exch0        1           6
    3     1.581    Exch0        1           6
    4     1.643    Exch0        1           6



```python
lob
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Timestamp</th>
      <th>Exchange</th>
      <th>BidAskData</th>
      <th>BidData</th>
      <th>AskData</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000</td>
      <td>Exch0</td>
      <td>[['bid', []], ['ask',</td>
      <td>[]</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.279</td>
      <td>Exch0</td>
      <td>[['bid', [[1, 6]]], ['ask',</td>
      <td>[[1, 6]]</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.333</td>
      <td>Exch0</td>
      <td>[['bid', [[1, 6]]], ['ask', [[800, 1</td>
      <td>[[1, 6]]</td>
      <td>[[800, 1]]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.581</td>
      <td>Exch0</td>
      <td>[['bid', [[1, 6]]], ['ask', [[799, 1</td>
      <td>[[1, 6]]</td>
      <td>[[799, 1]]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.643</td>
      <td>Exch0</td>
      <td>[['bid', [[1, 6]]], ['ask', [[798, 1</td>
      <td>[[1, 6]]</td>
      <td>[[798, 1]]</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>352965</th>
      <td>30599.542</td>
      <td>Exch0</td>
      <td>[['bid', [[290, 1], [288, 2], [281, 4], [278, ...</td>
      <td>[[290, 1]]</td>
      <td>[[288, 2]]</td>
    </tr>
    <tr>
      <th>352966</th>
      <td>30599.573</td>
      <td>Exch0</td>
      <td>[['bid', [[290, 1], [288, 2], [281, 4], [278, ...</td>
      <td>[[290, 1]]</td>
      <td>[[288, 2]]</td>
    </tr>
    <tr>
      <th>352967</th>
      <td>30599.635</td>
      <td>Exch0</td>
      <td>[['bid', [[290, 1], [288, 2], [281, 4], [278, ...</td>
      <td>[[290, 1]]</td>
      <td>[[288, 2]]</td>
    </tr>
    <tr>
      <th>352968</th>
      <td>30599.759</td>
      <td>Exch0</td>
      <td>[['bid', [[288, 1], [281, 4], [278, 1], [270, ...</td>
      <td>[[288, 1]]</td>
      <td>[[281, 4]]</td>
    </tr>
    <tr>
      <th>352969</th>
      <td>30599.945</td>
      <td>Exch0</td>
      <td>[['bid', [[288, 1], [281, 4], [278, 1], [270, ...</td>
      <td>[[288, 1]]</td>
      <td>[[281, 4]]</td>
    </tr>
  </tbody>
</table>
<p>352970 rows × 5 columns</p>
</div>




```python
# Define a function to unnest bid or ask data
def unnest_order_data(order_data, max_len, order_type='Bid'):
    # Create a DataFrame to hold the unnested data
    order_df = pd.DataFrame(index=lob.index)
    
    # Iterate over each possible position in the order data
    for i in range(max_len):
        # Safely extract the price and quantity from each order
        order_df[f'{order_type}Price_{i+1}'] = order_data.apply(lambda x: x[i][0] if isinstance(x, list) and len(x) > i else None)
        order_df[f'{order_type}Quantity_{i+1}'] = order_data.apply(lambda x: x[i][1] if isinstance(x, list) and len(x) > i else None)
    
    return order_df

# Find the maximum number of bids and asks in any row
max_bids = lob['BidData'].apply(lambda x: len(x) if isinstance(x, list) else 0).max()
max_asks = lob['AskData'].apply(lambda x: len(x) if isinstance(x, list) else 0).max()

# Unnest the bid and ask data
bid_df = unnest_order_data(lob['BidData'], max_bids, 'Bid')
ask_df = unnest_order_data(lob['AskData'], max_asks, 'Ask')

# Concatenate the bid and ask DataFrames with the original DataFrame
lob = pd.concat([lob, bid_df, ask_df], axis=1)

# Drop the original 'BidData' and 'AskData' columns as they have been unnested
lob.drop(['BidData', 'AskData'], axis=1, inplace=True)

# Display the first few rows of the updated DataFrame to verify the structure
print(lob.head())

```

      Timestamp Exchange                            BidAskData BidPrice_1  \
    0     0.000    Exch0                [['bid', []], ['ask',        None   
    1     0.279    Exch0          [['bid', [[1, 6]]], ['ask',           1   
    2     1.333    Exch0  [['bid', [[1, 6]]], ['ask', [[800, 1          1   
    3     1.581    Exch0  [['bid', [[1, 6]]], ['ask', [[799, 1          1   
    4     1.643    Exch0  [['bid', [[1, 6]]], ['ask', [[798, 1          1   
    
      BidQuantity_1 AskPrice_1 AskQuantity_1  
    0          None       None          None  
    1             6       None          None  
    2             6        800             1  
    3             6        799             1  
    4             6        798             1  



```python
lob
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Timestamp</th>
      <th>Exchange</th>
      <th>BidAskData</th>
      <th>BidPrice_1</th>
      <th>BidQuantity_1</th>
      <th>AskPrice_1</th>
      <th>AskQuantity_1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000</td>
      <td>Exch0</td>
      <td>[['bid', []], ['ask',</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.279</td>
      <td>Exch0</td>
      <td>[['bid', [[1, 6]]], ['ask',</td>
      <td>1</td>
      <td>6</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.333</td>
      <td>Exch0</td>
      <td>[['bid', [[1, 6]]], ['ask', [[800, 1</td>
      <td>1</td>
      <td>6</td>
      <td>800</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.581</td>
      <td>Exch0</td>
      <td>[['bid', [[1, 6]]], ['ask', [[799, 1</td>
      <td>1</td>
      <td>6</td>
      <td>799</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.643</td>
      <td>Exch0</td>
      <td>[['bid', [[1, 6]]], ['ask', [[798, 1</td>
      <td>1</td>
      <td>6</td>
      <td>798</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>352965</th>
      <td>30599.542</td>
      <td>Exch0</td>
      <td>[['bid', [[290, 1], [288, 2], [281, 4], [278, ...</td>
      <td>290</td>
      <td>1</td>
      <td>288</td>
      <td>2</td>
    </tr>
    <tr>
      <th>352966</th>
      <td>30599.573</td>
      <td>Exch0</td>
      <td>[['bid', [[290, 1], [288, 2], [281, 4], [278, ...</td>
      <td>290</td>
      <td>1</td>
      <td>288</td>
      <td>2</td>
    </tr>
    <tr>
      <th>352967</th>
      <td>30599.635</td>
      <td>Exch0</td>
      <td>[['bid', [[290, 1], [288, 2], [281, 4], [278, ...</td>
      <td>290</td>
      <td>1</td>
      <td>288</td>
      <td>2</td>
    </tr>
    <tr>
      <th>352968</th>
      <td>30599.759</td>
      <td>Exch0</td>
      <td>[['bid', [[288, 1], [281, 4], [278, 1], [270, ...</td>
      <td>288</td>
      <td>1</td>
      <td>281</td>
      <td>4</td>
    </tr>
    <tr>
      <th>352969</th>
      <td>30599.945</td>
      <td>Exch0</td>
      <td>[['bid', [[288, 1], [281, 4], [278, 1], [270, ...</td>
      <td>288</td>
      <td>1</td>
      <td>281</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
<p>352970 rows × 7 columns</p>
</div>




```python
def get_order_list_length(order_list_str):
    try:
        # Try to parse the string into actual lists using ast.literal_eval
        bid_ask_lists = ast.literal_eval(order_list_str)
        # Extract bids and asks
        bids = bid_ask_lists[0][1]  # This will be the list of bids
        asks = bid_ask_lists[1][1]  # This will be the list of asks
        return len(bids), len(asks)
    except (ValueError, SyntaxError, IndexError):
        # If there's any error parsing the string, return (0,0) to ignore this row
        return 0, 0

# Apply the function to each row and take the maximum
max_bid_ask_lengths = lob['BidAskData'].apply(get_order_list_length)
max_bids = max(max_bid_ask_lengths.apply(lambda x: x[0]))
max_asks = max(max_bid_ask_lengths.apply(lambda x: x[1]))

# Print the maximum lengths
print(f'Maximum number of bid pairs: {max_bids}')
print(f'Maximum number of ask pairs: {max_asks}')
```

    Maximum number of bid pairs: 0
    Maximum number of ask pairs: 0



```python
import pandas as pd
import re

# Assuming the lob DataFrame exists and has a column 'BidAskData' with the bid and ask data as strings
# The following code will split the 'BidAskData' into separate columns for bid1, bid2, ask1, and ask2

# A function to safely extract bid and ask data from the string without using literal_eval
def extract_bids_asks(bid_ask_str):
    # Using regular expressions to extract numbers from the strings
    bids_asks = re.findall(r'\[\[(.*?)\]\]', bid_ask_str)
    bids = re.findall(r'(\d+), (\d+)', bids_asks[0]) if len(bids_asks) > 0 else []
    asks = re.findall(r'(\d+), (\d+)', bids_asks[1]) if len(bids_asks) > 1 else []
    
    # Preparing the data with placeholders if there are less than two bids or asks
    bid1 = (int(bids[0][0]), int(bids[0][1])) if len(bids) > 0 else (None, None)
    bid2 = (int(bids[1][0]), int(bids[1][1])) if len(bids) > 1 else (None, None)
    ask1 = (int(asks[0][0]), int(asks[0][1])) if len(asks) > 0 else (None, None)
    ask2 = (int(asks[1][0]), int(asks[1][1])) if len(asks) > 1 else (None, None)
    
    return bid1 + bid2 + ask1 + ask2

# Create new columns in the DataFrame for each bid and ask price and quantity
lob[['BidPrice1', 'BidQuantity1', 'BidPrice2', 'BidQuantity2', 'AskPrice1', 'AskQuantity1', 'AskPrice2', 'AskQuantity2']] = pd.DataFrame(
    lob['BidAskData'].apply(extract_bids_asks).tolist(), index=lob.index
)

# Display the updated DataFrame
lob.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Timestamp</th>
      <th>Exchange</th>
      <th>BidAskData</th>
      <th>AskPrice_1</th>
      <th>AskQuantity_1</th>
      <th>BidPrice1</th>
      <th>BidQuantity1</th>
      <th>BidPrice2</th>
      <th>BidQuantity2</th>
      <th>AskPrice1</th>
      <th>AskQuantity1</th>
      <th>AskPrice2</th>
      <th>AskQuantity2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000</td>
      <td>Exch0</td>
      <td>[['bid', []], ['ask',</td>
      <td>None</td>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.279</td>
      <td>Exch0</td>
      <td>[['bid', [[1, 6]]], ['ask',</td>
      <td>None</td>
      <td>None</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.333</td>
      <td>Exch0</td>
      <td>[['bid', [[1, 6]]], ['ask', [[800, 1</td>
      <td>800</td>
      <td>1</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.581</td>
      <td>Exch0</td>
      <td>[['bid', [[1, 6]]], ['ask', [[799, 1</td>
      <td>799</td>
      <td>1</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.643</td>
      <td>Exch0</td>
      <td>[['bid', [[1, 6]]], ['ask', [[798, 1</td>
      <td>798</td>
      <td>1</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Assuming 'lob' is your pandas DataFrame and you want to remove columns named 'Column1', 'Column2', 'Column3'
columns_to_drop = ['AskPrice_1', 'AskQuantity_1']  # Replace with your actual column names
lob.drop(columns=columns_to_drop, axis=1, inplace=True)

# Now, 'lob' no longer contains the columns listed in 'columns_to_drop'.

```


```python
lob
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Timestamp</th>
      <th>Exchange</th>
      <th>BidAskData</th>
      <th>BidPrice1</th>
      <th>BidQuantity1</th>
      <th>BidPrice2</th>
      <th>BidQuantity2</th>
      <th>AskPrice1</th>
      <th>AskQuantity1</th>
      <th>AskPrice2</th>
      <th>AskQuantity2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000</td>
      <td>Exch0</td>
      <td>[['bid', []], ['ask',</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.279</td>
      <td>Exch0</td>
      <td>[['bid', [[1, 6]]], ['ask',</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.333</td>
      <td>Exch0</td>
      <td>[['bid', [[1, 6]]], ['ask', [[800, 1</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.581</td>
      <td>Exch0</td>
      <td>[['bid', [[1, 6]]], ['ask', [[799, 1</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.643</td>
      <td>Exch0</td>
      <td>[['bid', [[1, 6]]], ['ask', [[798, 1</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>352965</th>
      <td>30599.542</td>
      <td>Exch0</td>
      <td>[['bid', [[290, 1], [288, 2], [281, 4], [278, ...</td>
      <td>290.0</td>
      <td>1.0</td>
      <td>288.0</td>
      <td>2.0</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>352966</th>
      <td>30599.573</td>
      <td>Exch0</td>
      <td>[['bid', [[290, 1], [288, 2], [281, 4], [278, ...</td>
      <td>290.0</td>
      <td>1.0</td>
      <td>288.0</td>
      <td>2.0</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>352967</th>
      <td>30599.635</td>
      <td>Exch0</td>
      <td>[['bid', [[290, 1], [288, 2], [281, 4], [278, ...</td>
      <td>290.0</td>
      <td>1.0</td>
      <td>288.0</td>
      <td>2.0</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>352968</th>
      <td>30599.759</td>
      <td>Exch0</td>
      <td>[['bid', [[288, 1], [281, 4], [278, 1], [270, ...</td>
      <td>288.0</td>
      <td>1.0</td>
      <td>281.0</td>
      <td>4.0</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>352969</th>
      <td>30599.945</td>
      <td>Exch0</td>
      <td>[['bid', [[288, 1], [281, 4], [278, 1], [270, ...</td>
      <td>288.0</td>
      <td>1.0</td>
      <td>281.0</td>
      <td>4.0</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
<p>352970 rows × 11 columns</p>
</div>




```python

```
