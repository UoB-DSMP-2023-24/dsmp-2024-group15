import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('/Users/leichunyu/Desktop/UoB_Set01_2025-01-08LOBs_LOB_sorted.csv')


# Set drawing style
sns.set(style="whitegrid")

# Creates a graph object with two subgraphs that show the change in price and volume over time, respectively
fig, ax = plt.subplots(2, 1, figsize=(12, 10))

# Scatter plot: The effect of time on price
sns.scatterplot(x='Time', y='Price', hue='Type', style='Type', data=df, ax=ax[0])
ax[0].set_title('Effect of time on "price" (by type)')
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Price')

# Scatter plot: The effect of time on volume
sns.scatterplot(x='Time', y='Volume', hue='Type', style='Type', data=df, ax=ax[1])
ax[1].set_title('Impact of time on "volume" (by type)')
ax[1].set_xlabel('Time')
ax[1].set_ylabel('Volume')


plt.tight_layout()  # Automatically adjusts subgraph parameters to fill the entire image area
plt.show()


