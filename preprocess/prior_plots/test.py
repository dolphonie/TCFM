import pandas as pd

# Example dataframe
data = {'lat': [1, 1, 2, 2, 3, 3, 3, 4, 4, 5],
        'long': [10, 10, 20, 20, 30, 30, 30, 40, 40, 50]}
df = pd.DataFrame(data)

# Create a mask to filter rows where both 'lat' and 'long' are different from the previous row
mask = ~((df['lat'] == df['lat'].shift()) & (df['long'] == df['long'].shift()))

# Apply the mask to filter the dataframe
df_filtered = df[mask]

print(df_filtered)