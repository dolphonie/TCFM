import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import timedelta


def split_trajectories_max_duration(df, max_duration=timedelta(hours=24), min_duration=timedelta(minutes=30), max_time_gap=timedelta(minutes=30)):
    """
    Split trajectories based on maximizing the time between first and last points,
    while ensuring this time doesn't exceed the specified maximum duration.
    Trajectories shorter than the minimum duration are discarded.
    Any two points more than max_time_gap apart are treated as separate trajectories.

    :param df: pandas DataFrame with a 'timestamp' column
    :param max_duration: maximum allowed duration for a trajectory (default 24 hours)
    :param min_duration: minimum duration for a trajectory to be kept (default 30 minutes)
    :param max_time_gap: maximum allowed time gap between consecutive points (default 30 minutes)
    :return: tuple of (list of split indices, processed DataFrame)
    """
    # Ensure timestamp is in datetime format and sort
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # df = df.sort_values('timestamp').reset_index(drop=True)

    split_indices = [0]  # Start index of the first trajectory
    trajectory_start = 0
    valid_trajectories = []

    for i in range(1, len(df)):
        current_duration = abs(df.iloc[i]['timestamp'] - df.iloc[trajectory_start]['timestamp'])
        time_gap = df.iloc[i]['timestamp'] - df.iloc[i-1]['timestamp']
        
        print(current_duration)
        if current_duration > max_duration or time_gap > max_time_gap or i == len(df) - 1:
            print("Trajectory Start: ", trajectory_start, "Trajectory End: ", i)
            # End the current trajectory
            trajectory_end = i - 1 if time_gap > max_time_gap else i
            
            # Check if the trajectory meets the minimum duration
            if df.iloc[trajectory_end]['timestamp'] - df.iloc[trajectory_start]['timestamp'] >= min_duration:
                valid_trajectories.append((trajectory_start, trajectory_end + 1))
            
            # Start a new trajectory
            trajectory_start = i

    # # Create new split indices based on valid trajectories
    # new_split_indices = [start for start, _ in valid_trajectories]
    # if new_split_indices[-1] != len(df):
    #     new_split_indices.append(len(df) - 1)

    return valid_trajectories, df


def process_df(filepath, split="onground"):
    # df = pd.read_csv(case)

    df = pd.read_csv(filepath)
    df = df.dropna(subset=['latitude', 'longitude', 'altitude'])

    mask = ~((df['latitude'] == df['latitude'].shift()) & (df['longitude'] == df['longitude'].shift()) & (df['altitude'] == df['altitude'].shift()))

    df = df[mask].reset_index(drop=True)

    # Set any altitude less than 0 to 0
    df.loc[df['altitude'] < 0, 'altitude'] = 0

    # Get the first 1000 data points
    # subset_df = df.head(360)
    # print(subset_df)

    if split == "onground":
        # Get the points where the onground changes from 0 to 1
        onground = df['onground']
        cumsum_onground = onground.cumsum()
        diff = cumsum_onground.shift(1) != cumsum_onground
        # print indices where diff is True
        indices = diff[diff].index
    elif split=="altitude":
        # Get where the altitude is below 500
        alt_thresh = df[df['altitude'] < 2000].index
        diff = alt_thresh[1:] - alt_thresh[:-1]
        # indices = diff[diff > 1].index
        indices = alt_thresh[1:][diff > 1].tolist()
        indices = [0] + indices
    elif split == "time":
        # Get the points where the time changes
        # time = df['timestamp']
        time = pd.to_datetime(df['timestamp']).apply(lambda x: x.timestamp())
        diff = time - time.shift(1)
        # print indices where diff is True
        # indices = diff[diff > 18e3].index
        indices = diff[diff > 18e2].index # 30 minutes
        indices = diff[diff > 36e2].index # 60 minutes

        # indices = [0] + indices

    # print(lon_min, lon_max, lat_min, lat_max)

    # print("altitude")
    # print(alt_min, alt_max)

    # plt.figure(figsize=(10, 6))
    # plt.hist(df['altitude'], bins=50, color='skyblue', edgecolor='black')
    # plt.title('Altitude Distribution')
    # plt.xlabel('Altitude')
    # plt.ylabel('Frequency')
    # plt.grid(axis='y')
    # plt.savefig('altitude_histogram.png', dpi=300)

    selected_columns = df[['longitude', 'latitude', 'timestamp', 'altitude']]
    obs = selected_columns.to_numpy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    b = df.head(3500)

    # plot the altitude
    plt.figure()
    plt.scatter(b['timestamp'], b['altitude'], s=1)
    # plt.ylim(0, 6e3)
    # plt.title('Altitude', fontsize=20)
    plt.xlabel('Timestamp', fontsize=20)
    plt.ylabel('Altitude', fontsize=20)
    # make all the axis labels large
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.show()

    # datetime = pd.to_datetime(df['timestamp'])

    # print the IQR for altitude, latitude, and longitude
    # print(df['altitude'].quantile([0.25, 0.75]))
    # print(df['latitude'].quantile([0.25, 0.75]))
    # print(df['longitude'].quantile([0.25, 0.75]))

    # n = 10000
    # dt_subset = datetime[:n]
    # alt_subset = df['altitude'][:n]

    # plt.figure()
    # plt.scatter(dt_subset, alt_subset, s=1)
    # plt.show()

    
    return indices, df

def plot_path_on_map(m, x, y, size, gradient_color=False):
    path_length = len(x)
    
    if gradient_color:
        colors = plt.cm.jet(np.linspace(0, 1, path_length))
        m.scatter(x, y, s=size, c=colors, marker='o', label='Path')
    else:
        m.scatter(x, y, s=size, marker='o', label='Path', c='black')
        
        # Add text for start point
        plt.text(x[0], y[0], 'Start', fontsize=12, fontweight='bold', 
                 ha='right', va='bottom', color='red')
        
        # Add text for end point
        plt.text(x[-1], y[-1], 'End', fontsize=12, fontweight='bold', 
                 ha='left', va='top', color='red')
        
        # Optionally, you can add arrows or larger markers for start/end points
        m.scatter(x[0], y[0], s=size*2, marker='^', c='red', label='Start')
        m.scatter(x[-1], y[-1], s=size*2, marker='v', c='red', label='End')

def plot(df, lon_min, lon_max, lat_min, lat_max, gradient_color=False):
    plt.clf()
    fig = plt.gcf()
    fig.set_size_inches(10, 10)

    path_length = len(df)
    # path_length = len(obs)

    show_type = 'flat'
    if show_type == 'flat':
    # create map using BASEMAP
        m = Basemap(
                    llcrnrlon=lon_min,
                    llcrnrlat=lat_min,
                    urcrnrlon=lon_max,
                    urcrnrlat=lat_max,
                    lat_0=(lat_max - lat_min)/2,
                    lon_0=(lon_max-lon_min)/2,
                    projection='merc',
                    # projection='aeqd',
                    resolution = None,
                    area_thresh=10000.,
                    )
    else:
        width = 28000000
        m = Basemap(
                    lat_0=(lat_max - lat_min)/2,
                    lon_0=(lon_max-lon_min)/2,
                    projection='aeqd',
                    width = width,
                    height = width, 
                    )
    # m.bluemarble()
    m.shadedrelief()

    size = 3

    lons = df['longitude'].to_numpy()
    lats = df['latitude'].to_numpy()

    x, y = m(lons, lats)
    plot_path_on_map(m, x, y, size, gradient_color)

    # Draw parallels and meridians with labels in feet
    # m.drawparallels(range(int(lat_min), int(lat_max), 1), labels=[True, False, False, True], labelstyle='+/-', fmt='%.0f feet')
    # m.drawmeridians(range(int(lon_min), int(lon_max), 1), labels=[False, True, False, True], labelstyle='+/-', fmt='%.0f feet')

    #tight layout
    plt.tight_layout()
    plt.axis('off')

    if "timestamp" in df.columns:
        datetime = df["timestamp"]
    else:
        datetime = df.index
    # datetime = pd.to_datetime(df['timestamp'])
    altitude = df["altitude"].to_numpy()

    # display the altitude
    plt.figure()
    # time_x = list(range(len(altitude)))
    plt.scatter(datetime, altitude, s=1)
    plt.title('Altitude')
    plt.xlabel('Time')
    plt.ylabel('Altitude')
    # plt.ylim(0, 10e3)

    plt.show(block=False)

def plot_start_end(lons, lats, lon_min, lon_max, lat_min, lat_max):
    plt.clf()
    fig = plt.gcf()
    fig.set_size_inches(10, 10)

    # path_length = len(obs)

    show_type = 'flat'
    if show_type == 'flat':
    # create map using BASEMAP
        m = Basemap(
                    llcrnrlon=lon_min,
                    llcrnrlat=lat_min,
                    urcrnrlon=lon_max,
                    urcrnrlat=lat_max,
                    lat_0=(lat_max - lat_min)/2,
                    lon_0=(lon_max-lon_min)/2,
                    projection='merc',
                    # projection='aeqd',
                    resolution = None,
                    area_thresh=10000.,
                    )
    else:
        width = 28000000
        m = Basemap(
                    lat_0=(lat_max - lat_min)/2,
                    lon_0=(lon_max-lon_min)/2,
                    projection='aeqd',
                    width = width,
                    height = width, 
                    )
    m.bluemarble()

    size = 1
    # colors = plt.cm.jet(np.linspace(0,1,path_length))



    x, y = m(lons.tolist(), lats.tolist())
    # lon, lat
    # x, y = m(obs[:, 0], obs[:, 1])
    # m.plot(x, y, marker='.', linestyle='', s=1, markersize=size, c='grey', label='Path')
    m.scatter(x, y, s=size, marker='o', label='Path')

def interpolate_remove_outliers(traj_df):
    """ First remove all outliers from the altitude data
    Then linearly interpolate all the missing datapoints
    """
    # Remove altitude outliers
    traj_df = traj_df[traj_df['altitude'] < 10000]

    traj_df = traj_df[['longitude', 'latitude', 'timestamp', 'altitude']]
    # traj_df['timestamp'] = pd.to_datetime(traj_df['timestamp'])

    # Interpolate the missing data
    # traj_df = traj_df.interpolate(method='linear')

    # Resample the dataframe to have a measurement every minute
    df_resampled = traj_df.resample('30S', on='timestamp').mean().interpolate(method='linear')

    # df_resampled = df_resampled.reset_index(drop=True)

    # df_resampled['datetime']

    return df_resampled

def calculate_bounds(df):
    """
    Calculate the longitude and latitude bounds from a DataFrame.
    
    :param df: DataFrame containing 'longitude' and 'latitude' columns
    :return: tuple of (lon_min, lon_max, lat_min, lat_max)
    """
    lon_min = df['longitude'].min()
    lon_max = df['longitude'].max()
    lat_min = df['latitude'].min()
    lat_max = df['latitude'].max()
    
    # Add a small buffer (e.g., 1% of the range) to the bounds
    lon_buffer = (lon_max - lon_min) * 0.01
    lat_buffer = (lat_max - lat_min) * 0.01
    
    return (lon_min - lon_buffer, lon_max + lon_buffer, 
            lat_min - lat_buffer, lat_max + lat_buffer)

def main():

    # filepath = '/home/sean/Flight Data/flight_track_data_N172CK_2019.csv'
    filepath = 'data/march_flight_coamps.csv'
    # filepath = "data/N202GA_resampled.csv"
    folder_path = '/home/sean/TCFM/data/split'

    # check if the folder path exists, if not create it
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    factor = 2
    df = pd.read_csv(filepath)
    valid_trajs, df= split_trajectories_max_duration(df, max_duration=timedelta(hours=7), min_duration=timedelta(minutes=30), max_time_gap=timedelta(minutes=30))

    # lon_min = df['longitude'].mean() + factor*df['longitude'].std()
    # lon_max = df['longitude'].mean() - factor*df['longitude'].std()

    # lat_min = df['latitude'].mean() + factor*df['latitude'].std()
    # lat_max = df['latitude'].mean() - factor*df['latitude'].std()

    # alt_min = df['altitude'].mean() - factor*df['altitude'].std()
    # alt_max = df['altitude'].mean() + factor*df['altitude'].std()
    
    print("Number of segments: ", len(valid_trajs)-1)

    # plot all the start and end points scattered
    # end_lons = df['longitude'][indices]
    # end_lats = df['latitude'][indices]
    # plot_start_end(end_lons, end_lats, lon_min, lon_max, lat_min, lat_max)
    # plt.show()

    # for i in range(1, len(indices)-1):
    for i, (start, end) in enumerate(valid_trajs):
        # start = indices[i]
        # end = indices[i+1] - 1

        print(start, end)
        # df_resampled = interpolate_remove_outliers(df[start:end])
        df_resampled = df[start:end]
        lon_min, lon_max, lat_min, lat_max = calculate_bounds(df_resampled)
        plot(df_resampled, lon_min, lon_max, lat_min, lat_max)
        res = input("y/n accept or not: ")
        plt.close()

        if res != 'n':
            print("Saving segment ", i)
            df_resampled.to_csv(f'{folder_path}/segment_{i}.csv')

if __name__ == "__main__":
    main()

