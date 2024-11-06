import pandas as pd
import matplotlib.pyplot as plt
# df = pd.read_csv(case)

df = pd.read_csv('/home/sean/Flight Data/flight_track_data_N172CK_2018.csv')
df = df.dropna(subset=['latitude', 'longitude', 'altitude'])

# Set any altitude less than 0 to 0
df.loc[df['altitude'] < 0, 'altitude'] = 0

# Get the first 1000 data points
subset_df = df.head(360)
print(subset_df)

# # Save to a new CSV file
# subset_df.to_csv('/coc/data/prisoner_datasets/flight_track_data_N172CK_2018_subset.csv', index=False)

# lon_min = df['longitude'].min()
# lon_max = df['longitude'].max()

# lat_min = df['latitude'].min()
# lat_max = df['latitude'].max()

factor = 2

lon_min = df['longitude'].mean() + factor*df['longitude'].std()
lon_max = df['longitude'].mean() - factor*df['longitude'].std()

lat_min = df['latitude'].mean() + factor*df['latitude'].std()
lat_max = df['latitude'].mean() - factor*df['latitude'].std()

alt_min = df['altitude'].mean() - factor*df['altitude'].std()
alt_max = df['altitude'].mean() + factor*df['altitude'].std()

print(lon_min, lon_max, lat_min, lat_max)

print("altitude")
print(alt_min, alt_max)

plt.figure(figsize=(10, 6))
plt.hist(df['altitude'], bins=50, color='skyblue', edgecolor='black')
plt.title('Altitude Distribution')
plt.xlabel('Altitude')
plt.ylabel('Frequency')
plt.grid(axis='y')
plt.savefig('altitude_histogram.png', dpi=300)


# get the first few rows of the data and turn to numpy array
obs = df.to_numpy()

# print(obs)
import folium

# Create a base map
m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=10)

bounds = [(lat_min, lon_min), (lat_max, lon_max)]
m.fit_bounds(bounds)

# Plot the flight path
folium.PolyLine(subset_df[['latitude', 'longitude']].values, color="blue", weight=2.5, opacity=1).add_to(m)

m.save('test.html')

# from mpl_toolkits.basemap import Basemap
# import matplotlib.pyplot as plt
# import numpy as np

# plt.clf()
# fig = plt.gcf()
# fig.set_size_inches(10, 10)

# path_length = len(obs)

# show_type = 'flat'
# if show_type == 'flat':
# # create map using BASEMAP
#     m = Basemap(
#                 llcrnrlon=lon_min,
#                 llcrnrlat=lat_min,
#                 urcrnrlon=lon_max,
#                 urcrnrlat=lat_max,
#                 lat_0=(lat_max - lat_min)/2,
#                 lon_0=(lon_max-lon_min)/2,
#                 projection='merc',
#                 # projection='aeqd',
#                 resolution = None,
#                 area_thresh=10000.,
#                 )
# else:
#     width = 28000000
#     m = Basemap(
#                 lat_0=(lat_max - lat_min)/2,
#                 lon_0=(lon_max-lon_min)/2,
#                 projection='aeqd',
#                 width = width,
#                 height = width, 
#                 )
# m.bluemarble()

# size = 1
# colors = plt.cm.jet(np.linspace(0,1,path_length))

# # lon, lat
# x, y = m(obs[:, 0], obs[:, 1])
# # m.plot(x, y, marker='.', linestyle='', s=1, markersize=size, c='grey', label='Path')
# m.scatter(x, y, s=size, c=colors, marker='o', label='Path')

# #tight layout
# plt.tight_layout()

# plt.axis('off')
# plt.title('test')