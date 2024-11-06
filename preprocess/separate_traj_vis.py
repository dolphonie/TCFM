import os
import pandas as pd
import folium

path = '/home/sean/FlightData/processed/N172CK/train'

# Load the data
# Folder contains multiple csv files for each trajectory
files = [f for f in os.listdir(path) if f.endswith('.csv')]

lat_mean = 40.86974775781146
lon_mean = -75.0979672941728

# m = folium.Map(location=[lat_mean, lon_mean], zoom_start=10)
m = folium.Map(location=[lat_mean, lon_mean], zoom_start=10, tiles="Cartodb Positron")

colors = ['blue', 'red', 'purple', 'orange', 'darkred', 'lightred', 'darkblue', 'cadetblue', 'darkpurple', 'pink', 'black']
total_colors = len(colors)

import random
# select 10 random files from files
random.shuffle(files)

for i, file in enumerate(files[:15]):
    flight_data = pd.read_csv(os.path.join(path, file))

    # Plot the flight path
    # choose different color for each path
    folium.PolyLine(flight_data[['latitude', 'longitude']].values, color=colors[i % total_colors], weight=2.5, opacity=1).add_to(m)

    # folium.PolyLine(flight_data[['latitude', 'longitude']].values, color="blue", weight=2.5, opacity=1).add_to(m)

# save the map to image
m.save('flight_traj.html')
