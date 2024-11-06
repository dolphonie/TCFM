import os
import numpy as np
import matplotlib.pyplot as plt

base_branches = [
    '/coc/data/sye40/prisoner_logs/IROS24/smuggler_paper_3_helo_40/cfm_wavelet_False/diffusion/H120_T100']
    # '/coc/data/sye40/prisoner_logs/IROS24/smuggler_paper_3_helo_40/cfm_wavelet_False/diffusion/H120_T100']

averages = []
mins = []
for base_branch in base_branches:
# base_branch = '/data/sye40/prisoner_logs/MRS/base_branch/smuggler_2_helo_40'
    folders = os.listdir(base_branch)
    folders = [os.path.join(base_branch, folder) for folder in folders]

dist_mins = []
dist_averages = []

for folder in folders:
    filepath = os.path.join(folder, 'distances.npz')
    file = np.load(filepath, allow_pickle=True)
    dist_mins.append(file['dist_min'])
    dist_averages.append(file['dist_averages'])

dist_averages = np.concatenate(dist_averages, axis=0)
dist_mins = np.concatenate(dist_mins, axis=0)

mins.append(np.mean(dist_mins, axis=0))
averages.append(np.mean(dist_averages, axis=0))

print(len(averages))

print(averages[0][0], averages[0][29], averages[0][59], averages[0][89], averages[0][119])
# print(averages[1][0], averages[1][29], averages[1][59], averages[1][89], averages[1][119])
# print(averages[2][0], averages[2][29], averages[2][59], averages[2][89], averages[2][119])