import numpy as np
import os
import matplotlib.pyplot as plt

timesteps = list(range(1, 10, 1))

def get_stats_timestep(path):
    f = np.load(path, allow_pickle=True)
    dist_min = f['dist_min']
    dist_averages = f['dist_averages'] / 2428
    times = f['times']

    avg = np.mean(dist_averages)
    std = np.std(dist_averages)
    ste = std / np.sqrt(len(dist_averages))

    t_avg = np.mean(times)
    t_std = np.std(times)

    return avg, std, t_avg, t_std

def get_stats_folder(path):
    """ Return a list of averages for each timestep"""
    dist_averages = []
    dist_std = []
    for t in timesteps:
        p = os.path.join(path, f'{t}.npz')
        avg, std, t_avg, t_std = get_stats_timestep(p)
        dist_averages.append(avg)
        dist_std.append(std)

    return dist_averages


def get_stats_model(path):
    """ Return a list of averages for each model in the path."""
    # get all the folders in this path
    folders = os.listdir(path)

    dist_averages = []

    for folder in folders:
        avg  = get_stats_folder(os.path.join(path, folder, 'sample_time'))
        dist_averages.append(avg)

    dist = np.array(dist_averages)
    dist_averages_all = np.mean(dist, axis=0)
    dist_std = np.std(dist, axis=0)

    return dist_averages_all, dist_std



def main():

    diffusion_models = '/home/sean/prisoner_logs/MRS/4_detect/best'
    cfm_models = '/home/sean/prisoner_logs/IROS24/4_detect/cfm_wavelet_False/diffusion/H120_T100'

    dist_averages, dist_std = get_stats_model(diffusion_models)
    dist_cfm_averages, dist_cfm_std = get_stats_model(cfm_models)

    print(dist_cfm_std)

    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    # plt.plot(timesteps, dist_averages, '-o', label='Data')
    # plt.plot(timesteps, dist_cfm_averages, '-o', label='CFM Data')

    # Plot the line chart with error bars
    plt.errorbar(np.array(timesteps), dist_averages, yerr=dist_std, fmt='-o', label='CADENCE', ecolor='black')
    plt.errorbar(np.array(timesteps), dist_cfm_averages, yerr=dist_cfm_std, fmt='-o', label='T-CFM (Ours)', ecolor='black')

    # Add labels and title
    plt.xlabel('Number Sampling Steps')
    plt.ylabel('Average Displacement Error')
    # plt.title('')

    # plt.grid(axis='y')
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.legend()

    # xlim and ylim
    # plt.xlim(9.5, 0)
    plt.ylim(0, 0.25)

    # Show the plot
    # plt.show()
    plt.savefig('sampling_time_std_reversed.png')

if __name__ == '__main__':
    main()