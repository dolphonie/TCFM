import os
import numpy as np
import matplotlib.pyplot as plt

def get_diffusion_models():
    base_branches = [
        '/coc/data/sye40/prisoner_logs/MRS/base_branch/3_detects/best',
        '/coc/data/sye40/prisoner_logs/MRS/base_branch/4_detects/best',             
        '/coc/data/sye40/prisoner_logs/MRS/base_branch/7_detects/best']

    averages = []
    average_stds = []
    mins = []
    min_stds = []
    average_stes = []
    for base_branch in base_branches:
        folders = os.listdir(base_branch)
        folders = [os.path.join(base_branch, folder) for folder in folders]

        dist_mins = []
        dist_averages = []
        for folder in folders:
            if '3_detect' in base_branch:
                filepath = os.path.join(folder, 'distances_cond_original.npz')
            else:
                filepath = os.path.join(folder, 'distances_cond.npz')
            file = np.load(filepath, allow_pickle=True)
            dist_mins.append(file['dist_min'] / 2428)
            dist_averages.append(file['dist_averages'] / 2428)

        dist_averages = np.concatenate(dist_averages, axis=0)
        num_samples = dist_averages.shape[0]
        dist_avg_std = np.std(dist_averages, axis=0)
        d_averages_ste = dist_avg_std / np.sqrt(num_samples)

        dist_mins = np.concatenate(dist_mins, axis=0)
        dist_min_std = np.std(dist_mins, axis=0)

        mins.append(np.mean(dist_mins, axis=0))
        averages.append(np.mean(dist_averages, axis=0))
        average_stds.append(dist_avg_std)
        average_stes.append(d_averages_ste)
        min_stds.append(dist_min_std)
    return averages, average_stds, mins, min_stds, average_stes

def get_cfm_models():
    base_branches = ["/coc/data/sye40/prisoner_logs/IROS24/3_detect",
                "/coc/data/sye40/prisoner_logs/IROS24/4_detect",
                "/coc/data/sye40/prisoner_logs/IROS24/7_detect"]

    base_branches = [i + "/cfm_wavelet_False/" for i in base_branches]

    averages = []
    average_stds = []
    average_stes = []
    mins = []
    min_stds = []
    for base_branch in base_branches:
        # folders = os.listdir(base_branch)
        # folders = [os.path.join(base_branch, folder) for folder in folders]

        folders = []
        for root, dirs, files in os.walk(base_branch):
            for d in dirs:
                if not any(os.path.isdir(os.path.join(root, d, sub_d)) for sub_d in os.listdir(os.path.join(root, d))):
                    folders.append(os.path.join(root, d))

        dist_mins = []
        dist_averages = []
        for folder in folders:
            filepath = os.path.join(folder, 'distances_cond.npz')
            file = np.load(filepath, allow_pickle=True)
            dist_mins.append(file['dist_min'] / 2428)
            dist_averages.append(file['dist_averages'] / 2428)

        dist_averages = np.concatenate(dist_averages, axis=0)
        num_samples = dist_averages.shape[0]
        dist_avg_std = np.std(dist_averages, axis=0)
        d_averages_ste = dist_avg_std / np.sqrt(num_samples)

        dist_mins = np.concatenate(dist_mins, axis=0)
        dist_min_std = np.std(dist_mins, axis=0)

        mins.append(np.mean(dist_mins, axis=0))
        averages.append(np.mean(dist_averages, axis=0))
        average_stds.append(dist_avg_std)
        average_stes.append(d_averages_ste)
        min_stds.append(dist_min_std)
        
    return averages, average_stds, mins, min_stds, average_stes

def plot(ax, averages, average_stds, name, all_colors):
    # ax.plot(np.arange(0, 120), averages[0], linewidth=3, label=f'{name}-Low Detect Rate', color=all_colors[0])
    # ax.fill_between(np.arange(0, 120), averages[0] - average_stds[0], averages[0] + average_stds[0], color=all_colors[0], alpha=0.2)
    
    # ax.plot(np.arange(0, 120), averages[1], linewidth=3, label=f'{name} - Medium Detect Rate', color=all_colors[1])
    # ax.fill_between(np.arange(0, 120), averages[1] - average_stds[1], averages[1] + average_stds[1], color=all_colors[1], alpha=0.2)

    ax.plot(np.arange(0, 120), averages[2], linewidth=3, label=f'{name}-High Detect Rate', color=all_colors[2])
    ax.fill_between(np.arange(0, 120), averages[2] - average_stds[2], averages[2] + average_stds[2], color=all_colors[2], alpha=0.2)
    
    # ax.plot(np.arange(0, 120), averages[1], linewidth=3, label='Medium Rate')
    # ax.plot(np.arange(0, 120), averages[2], linewidth=3, label=f'{name}-High Detect Rate', color=all_colors[1])
    # ax.fill_between(np.arange(0, 120), averages[2] - average_stds[2], averages[2] + average_stds[2], color=all_colors[1], alpha=0.2)

def main():
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    colors = ['tab:blue', 'tab:orange']

    # multistream = '/coc/data/sye40/prisoner_logs/MRS/multiagent_branch/balance-map-random/multi/diffusion/H120_T100/20230627-1747'
    # singlestream = '/coc/data/sye40/prisoner_logs/MRS/multiagent_branch/balance-map-random/single-stream/20230627-1019'

    # dist_mins = []
    # dist_averages = []

    fig, ax = plt.subplots(figsize=(8, 6))
    # plt.title("Target Tracking", fontsize=16, fontweight="bold")

    all_colors = ["#78b24c", "#6581c6", "#edae65"]

    x = [0, 30, 60, 90, 120]
    three_detects = [0.060, 0.080, 0.109, 0.146, 0.163]
    four_detects = [0.047, 0.077, 0.110, 0.142, 0.167]
    seven_detects = [0.015, 0.056, 0.092, 0.117, 0.145]

    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    # fig, ax = plt.subplots(figsize=(8, 6))
    # plt.title("Single-Target Tracking", fontsize=16, fontweight="bold")
    # all_colors = ["black", "red", "red"]
    # all_colors = ['#e41a1c', '#e41a1c', '#e41a1c']
    # all_colors = ['#984ea3'] * 3
    all_colors = ["#377eb8"] * 3
    diff_averages, average_stds, mins, min_stds, d_averages_ste = get_diffusion_models()
    plot(ax, diff_averages, average_stds, "CADENCE", all_colors)
    
    
    # all_colors = ["#78b24c", "#6581c6", "#edae65"]
    all_colors = ['#e41a1c'] * 3
    cfm_averages, average_stds, mins, min_stds, d_averages_ste = get_cfm_models()
    plot(ax, cfm_averages, average_stds, "CFM", all_colors)

    print(cfm_averages[0][0], cfm_averages[0][29], cfm_averages[0][59], cfm_averages[0][89], cfm_averages[0][119])
    print(cfm_averages[1][0], cfm_averages[1][29], cfm_averages[1][59], cfm_averages[1][89], cfm_averages[1][119])
    print(cfm_averages[2][0], cfm_averages[2][29], cfm_averages[2][59], cfm_averages[2][89], cfm_averages[2][119])

    # compute the improvement over diffusion models
    improvement = (np.array(cfm_averages) - np.array(diff_averages)) / np.array(diff_averages)
    improvement = np.mean(improvement) * 100
    print("Improvement: ", improvement)

    # print(averages[1][0], averages[1][29], averages[1][59], averages[1][89], averages[1][119])
    # print(averages[2][0], averages[2][29], averages[2][59], averages[2][89], averages[2][119])

    xlabel = ax.set_xlabel("Prediction Horizon", fontsize=14)

    markersize = 10
    # ax.plot(x, three_detects, marker = '*', linestyle = 'dashed', label='GrAMMi-Low Detect Rate', markersize = markersize, color=all_colors[0])
    # # ax.scatter(x, four_detects, marker = '*', s=markersize, label='Medium Rate (Prev Best)')
    # ax.plot(x, seven_detects, marker = '*', linestyle = 'dashed', label='GrAMMI-High Detect Rate', markersize = markersize, color=all_colors[1])

    # plt.legend(ncols=1, fontsize=10)
    plt.ylim(0, 0.27)
    plt.legend()
    plt.legend(ncol=2, bbox_to_anchor=(0.5, -0.2), loc='lower center')
    plt.ylabel("Average Displacement Error (ADE)", fontsize=14)
    plt.xlabel("Prediction Horizon", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    plt.savefig('cfm_diffusion_ade.png')

if __name__ == "__main__":
    main()