import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

colors = ['tab:blue', 'tab:orange']

# multistream = '/coc/data/sye40/prisoner_logs/MRS/multiagent_branch/balance-map-random/multi/diffusion/H120_T100/20230627-1747'
# singlestream = '/coc/data/sye40/prisoner_logs/MRS/multiagent_branch/balance-map-random/single-stream/20230627-1019'

# dist_mins = []
# dist_averages = []

fig, ax = plt.subplots(figsize=(8, 6))
plt.title("Target Tracking", fontsize=16, fontweight="bold")

# i = 0
# for folder, folder_name, figure_name in zip([multistream, singlestream], ['Known Detection Origin', 'Unknown Detection Origin'], ['Known Detection Identity', 'Unknown Detection Identity']):
#     filepath = os.path.join(folder, 'distances_original.npz')
#     file = np.load(filepath, allow_pickle=True)


#     dist_min = file['dist_min']
#     dist_average = file['dist_averages']

#     if len(dist_min.shape) == 3:
#         num_samples = dist_min.shape[0] * dist_min.shape[2]

#         d_min = np.mean(dist_min, axis=0).mean(axis=-1) / 2428
#         d_min_std = np.std(dist_min, axis=0).mean(axis=-1) / 2428
#         d_min_ste = d_min_std / np.sqrt(num_samples)
#         d_averages = np.mean(dist_average, axis=0) / 2428

#         d_averages_std = np.std(dist_average, axis=0) / 2428
#         d_averages_ste = d_averages_std / np.sqrt(num_samples)

#     else:
#         num_samples = dist_min.shape[0]
#         d_min = np.mean(dist_min, axis=0) / 2428
#         d_min_std = np.std(dist_min, axis=0) / 2428
#         d_min_ste = d_min_std / np.sqrt(num_samples)
#         d_averages = np.mean(dist_average, axis=0) / 2428

#         d_averages_std = np.std(dist_average, axis=0) / 2428
#         d_averages_ste = d_averages_std / np.sqrt(num_samples)

#     print(folder_name)
#     print(np.mean(d_averages))
#     print(np.mean(d_min))

#     # print(d_min[0], d_min[29], d_min[59])
#     ax.plot(np.arange(0, 120), d_averages, linewidth=3, label=f'{figure_name} ADE', color = colors[i])
#     ax.plot(np.arange(0, 120), d_min, linewidth=3, label=f'{figure_name} minADE', linestyle='--', color = colors[i])

#     # Plot standard deviation
#     ax.fill_between(np.arange(0, 120), d_averages - 2*d_averages_ste, d_averages + 2*d_averages_ste, color=colors[i], alpha=0.2)
#     ax.fill_between(np.arange(0, 120), d_min - 2*d_min_ste, d_min + 2*d_min_ste, color=colors[i], alpha=0.2)

#     i += 1

def get_diffusion_models():
    base_branches = [
        '/coc/data/sye40/prisoner_logs/MRS/base_branch/4_detects/best']
        # '/coc/data/sye40/prisoner_logs/MRS/base_branch/4_detects/best',             
        # '/coc/data/sye40/prisoner_logs/MRS/base_branch/7_detects/best']

    averages = []
    average_stds = []
    mins = []
    min_stds = []
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
            dist_mins.append(file['dist_min'])
            dist_averages.append(file['dist_averages'])

        dist_averages = np.concatenate(dist_averages, axis=0)
        dist_avg_std = np.std(dist_averages, axis=0)

        dist_mins = np.concatenate(dist_mins, axis=0)
        dist_min_std = np.std(dist_mins, axis=0)

        mins.append(np.mean(dist_mins, axis=0)/ 2428)
        averages.append(np.mean(dist_averages, axis=0)/ 2428)
        average_stds.append(dist_avg_std / 2428)
        min_stds.append(dist_min_std / 2428)
    return averages, average_stds, mins, min_stds, dist_averages

def get_cfm_models():
    base_branches = ["/coc/data/sye40/prisoner_logs/IROS24/4_detect"]
                # "/coc/data/sye40/prisoner_logs/IROS24/4_detect",
                # "/coc/data/sye40/prisoner_logs/IROS24/7_detect"]

    base_branches = [i + "/cfm_wavelet_False/" for i in base_branches]

    averages = []
    average_stds = []
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
            # if "1227" in folder:
            filepath = os.path.join(folder, 'distances_cond.npz')
            file = np.load(filepath, allow_pickle=True)
            dist_mins.append(file['dist_min'])
            dist_averages.append(file['dist_averages'])

        dist_averages = np.concatenate(dist_averages, axis=0)
        dist_avg_std = np.std(dist_averages, axis=0)

        dist_mins = np.concatenate(dist_mins, axis=0)
        dist_min_std = np.std(dist_mins, axis=0)

        mins.append(np.mean(dist_mins, axis=0)/ 2428)
        averages.append(np.mean(dist_averages, axis=0)/ 2428)
        average_stds.append(dist_avg_std / 2428)
        min_stds.append(dist_min_std / 2428)
    return averages, average_stds, mins, min_stds, dist_averages

def plot(ax, averages, average_stds, name, all_colors):
    ax.plot(np.arange(0, 120), averages[0], linewidth=3, label=f'{name}-Low Detect Rate', color=all_colors[0])
    ax.fill_between(np.arange(0, 120), averages[0] - average_stds[0], averages[0] + average_stds[0], color=all_colors[0], alpha=0.2)
    # ax.plot(np.arange(0, 120), averages[1], linewidth=3, label='Medium Rate')
    # ax.plot(np.arange(0, 120), averages[2], linewidth=3, label=f'{name}-High Detect Rate', color=all_colors[1])
    # ax.fill_between(np.arange(0, 120), averages[2] - average_stds[2], averages[2] + average_stds[2], color=all_colors[1], alpha=0.2)

def t_test():
    from scipy import stats
    diff_averages, average_stds, mins, min_stds, dist_averages_diff = get_diffusion_models()
    cfm_averages, average_stds, mins, min_stds, dist_averages_cfm = get_cfm_models()

    idx = 119

    data1 = dist_averages_diff[:, idx]
    data2 = dist_averages_cfm[:, idx]

    t_statistic, p_value = stats.ttest_ind(data1, data2)

    # Calculate degrees of freedom
    df = len(data1) + len(data2) - 2

    # Define significance level
    alpha = 0.05

    # Print results
    print(f"t-statistic: {t_statistic}")
    print(f"p-value: {p_value}")
    print(f"Degrees of freedom: {df}")

    # Compare with critical value
    critical_value = stats.t.ppf(1 - alpha / 2, df)
    print(f"Critical value: {critical_value}")

    if np.abs(t_statistic) > critical_value:
        print("Reject null hypothesis: means are statistically significantly different")
    else:
        print("Fail to reject null hypothesis: means are not statistically significantly different")


def main():
    all_colors = ["#78b24c", "#6581c6", "#edae65"]

    x = [0, 30, 60, 90, 120]
    three_detects = [0.060, 0.080, 0.109, 0.146, 0.163]
    four_detects = [0.047, 0.077, 0.110, 0.142, 0.167]
    seven_detects = [0.015, 0.056, 0.092, 0.117, 0.145]

    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    # fig, ax = plt.subplots(figsize=(8, 6))
    # plt.title("Single-Target Tracking", fontsize=16, fontweight="bold")
    all_colors = ["#78b24c", "#6581c6", "#edae65"]
    diff_averages, average_stds, mins, min_stds, _ = get_diffusion_models()
    plot(ax, diff_averages, average_stds, "CADENCE", all_colors)
    
    all_colors = ["black", "#6581c6", "#edae65"]
    cfm_averages, average_stds, mins, min_stds, _ = get_cfm_models()
    plot(ax, cfm_averages, average_stds, "CFM", all_colors)

    print(cfm_averages[0][0], cfm_averages[0][29], cfm_averages[0][59], cfm_averages[0][89], cfm_averages[0][119])

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
    plt.ylim(0, 0.35)
    plt.legend(ncol=2, bbox_to_anchor=(0.5, -0.4), loc='lower center')
    plt.ylabel("Distance Error", fontsize=14)
    plt.xlabel("Prediction Horizon", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    plt.savefig('cfm_diffusion_ade.png')

if __name__ == "__main__":
    # main()
    t_test()