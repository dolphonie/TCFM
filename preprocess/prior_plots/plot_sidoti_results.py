import numpy as np
loadpath_diff = "/coc/data/sye40/prisoner_logs/aircraft_sidoti_separate/maze2d-large-v1/diffusion/H60_T100/20240221-1338/distances_cond.npz"
loadpath_cfm = "/coc/data/sye40/prisoner_logs/aircraft_sidoti_separate/cfm/diffusion/H60_T100/20240222-1318/distances_cond.npz"

data_diff = np.load(loadpath_diff)
data_cfm = np.load(loadpath_cfm)

# plot dist min and dist averages

dist_min_diff = data_diff['dist_min']
dist_min_cfm = data_cfm['dist_min']

dist_avg_diff = data_diff['dist_averages']
dist_avg_cfm = data_cfm['dist_averages']

import matplotlib.pyplot as plt

dist_min_diff_mean = dist_min_diff.mean(axis=0)
dist_min_diff_std = dist_min_diff.std(axis=0)

dist_min_cfm_mean = dist_min_cfm.mean(axis=0)
dist_min_cfm_std = dist_min_cfm.std(axis=0)

x_range = np.array(list(range(60))) / 2.

plt.plot(x_range, dist_min_diff_mean, label="diffusion")
plt.fill_between(x_range, dist_min_diff_mean - dist_min_diff_std, dist_min_diff_mean + dist_min_diff_std, alpha = 0.3)
# plt.plot(dist_min_cfm, label="cfm")

plt.plot(x_range, dist_min_cfm_mean, label="cfm")
plt.fill_between(x_range, dist_min_cfm_mean - dist_min_cfm_std, dist_min_cfm_mean + dist_min_cfm_std, alpha = 0.3)

plt.ylabel("Minimum Average Displacement Error (kms)")
plt.xlabel("Time Horizon (minutes)")
plt.ylim(0, 30)
plt.legend()
plt.savefig("sidoti_min.png")


gmm_dists = [
    0.547076068,
    7.865,
    11.398,
    13.23,
    12.1903]

gmm_x = np.array([0, 15, 30, 45, 60]) / 2

# plot dist averages
dist_avg_diff_mean = dist_avg_diff.mean(axis=0)
dist_avg_diff_std = dist_avg_diff.std(axis=0)

dist_avg_cfm_mean = dist_avg_cfm.mean(axis=0)
dist_avg_cfm_std = dist_avg_cfm.std(axis=0)

plt.figure()
plt.plot(x_range, dist_avg_diff_mean, label="diffusion")
plt.fill_between(x_range, dist_avg_diff_mean - dist_avg_diff_std, dist_avg_diff_mean + dist_avg_diff_std, alpha = 0.3)
# plt.plot(dist_avg_cfm, label="cfm")

plt.plot(x_range, dist_avg_cfm_mean, label="cfm")
plt.fill_between(x_range, dist_avg_cfm_mean - dist_avg_cfm_std, dist_avg_cfm_mean + dist_avg_cfm_std, alpha = 0.3)

plt.scatter(gmm_x, gmm_dists, label="gmm")

plt.ylabel("Average Displacement Error (kms)")
plt.xlabel("Time Horizon (min)")
plt.ylim(0, 35)
plt.legend()
plt.savefig("sidoti_avg.png")