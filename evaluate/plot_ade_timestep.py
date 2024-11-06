import numpy as np
import matplotlib.pyplot as plt

dist_t_sponsor = np.load("evaluate/dist_t_sponsor.npy")
dist_t_prisoner = np.load("evaluate/dist_t_prisoner.npy")

std_sponsor = np.std(dist_t_sponsor, axis=0)[:128]
std_prisoner = np.std(dist_t_prisoner, axis=0)[:128]

dist_t_sponsor = np.mean(dist_t_sponsor, axis=0)[:128]
dist_t_prisoner = np.mean(dist_t_prisoner, axis=0)[:128]
x_axis = np.arange(len(dist_t_sponsor))


plt.plot(x_axis, dist_t_sponsor, label="Sponsor Dataset")
plt.fill_between(x_axis, dist_t_sponsor + std_sponsor, dist_t_sponsor - std_sponsor, alpha=0.2)

plt.plot(x_axis, dist_t_prisoner, label="Prisoner Dataset")
plt.fill_between(x_axis, dist_t_prisoner + std_prisoner, dist_t_prisoner - std_prisoner, alpha=0.2)

plt.xlabel("Timesteps", fontsize=16, fontweight='bold')
plt.ylabel("Average Distance Error (ADE)", fontsize=16, fontweight='bold')
plt.legend()

plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')

plt.savefig("dist_t_both.png")