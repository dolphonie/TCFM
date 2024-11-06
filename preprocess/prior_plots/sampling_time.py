import numpy as np
import os
import matplotlib.pyplot as plt

def get_stats(path):
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



def main():

    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    root_path = '/home/sean/prisoner_logs/MRS/4_detect/best/20230610-0141/sample_time'

    times = []
    times_std = []
    dist_averages = []
    dist_std = []

    timesteps = list(range(1, 10, 1))

    for t in timesteps:
        print(t)
        path = os.path.join(root_path, f'{t}.npz')
        avg, std, t_avg, t_std = get_stats(path)
        dist_averages.append(avg)
        dist_std.append(std)

    cfm_path = '/home/sean/prisoner_logs/IROS24/4_detect/cfm_wavelet_False/diffusion/H120_T100/20240213-1227/sample_time'

    dist_cfm_averages = []
    dist_cfm_std = []

    for t in timesteps:
        print(t)
        path = os.path.join(cfm_path, f'{t}.npz')
        avg, std, t_avg, t_std = get_stats(path)
        dist_cfm_averages.append(avg)
        dist_cfm_std.append(std)

    # plt.plot(timesteps, dist_averages, '-o', label='Data')
    # plt.plot(timesteps, dist_cfm_averages, '-o', label='CFM Data')

    # Plot the line chart with error bars
    plt.errorbar(np.array(timesteps) + 0.1, dist_averages, yerr=dist_std, fmt='-o', label='CADENCE')
    plt.errorbar(np.array(timesteps), dist_cfm_averages, yerr=dist_cfm_std, fmt='-o', label='T-CFM')

    # Add labels and title
    plt.xlabel('Number Sampling Steps')
    plt.ylabel('Average Displacement Error')
    # plt.title('')
    plt.legend()

    # xlim and ylim
    # plt.xlim(0, 1)
    plt.ylim(0, 0.35)

    # Show the plot
    # plt.show()
    plt.savefig('sampling_time.png')

if __name__ == '__main__':
    main()