import pandas as pd
import folium

# data_path = "data/N202GA_resampled.csv"
data_path = "data/N843E_resampled.csv"

# Load the data
flight_data = pd.read_csv(data_path)
flight_data.head()

flight_data.loc[flight_data['altitude'].isna(), 'onground'] = False
# if there any missing values in the latitude or longitude, remove those timestep from the dataset.
flight_data.dropna(subset=['latitude', 'longitude'], inplace=True)

flight = flight_data.head(10000)

# # Create a trajectory identifier based on changes in the "onground" column
# flight_data['trajectory_id'] = flight_data['onground'].diff().ne(0).cumsum()

# # Now, we'll split the data into separate trajectories based on the trajectory_id
# trajectories_dict = {trajectory_id: data for trajectory_id, data in flight_data.groupby('trajectory_id')}

# print(trajectories_dict)

# # filter out trajectories that are too short
# # trajectories = [data for trajectory_id, data in trajectories_dict.items() if len(data) > 500]
# trajectories = [data for trajectory_id, data in trajectories_dict.items() if len(data)]
# print([len(trajectory) for trajectory in trajectories])

# idx = 7
# flight = trajectories[idx]
# Create a base map

print(flight['latitude'].mean())
print(flight['longitude'].mean())

m = folium.Map(location=[flight['latitude'].mean(), flight['longitude'].mean()], zoom_start=10)

# Plot the flight path
folium.PolyLine(flight[['latitude', 'longitude']].values, color="blue", weight=2.5, opacity=1).add_to(m)

# save the map to image
m.save('flight_path.html')