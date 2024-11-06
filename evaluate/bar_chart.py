import matplotlib.pyplot as plt
import numpy as np

# Define data as arrays
model_types = ['GMM-Sponsor', 'Diffusion-Sponsor', 'Diffusion-Prisoner']
datasets = ['Sponsor Train Set', 'Sponsor Test Set', 'Sponsor Train Set', 'Sponsor Test Set', 'Prisoner Train Set', 'Prisoner Test Set']
hideout_info = ['No', 'No', 'No', 'No', 'Yes', 'Yes']
ade = [947.56, 1047.93, 507.28, 732.88, 212.43, 434.53]
containment = [44, 37, 93, 35, 98, 92]

width = 0.4

# Create the figure and axis objects for ADE plot
fig1, ax1 = plt.subplots(figsize=(8, 6))

# Create arrays for the x-axis and bar positions
x1 = np.arange(0, len(model_types)*2, 2)

# Create the bar chart for ADE data
# colors = ['tab:blue' if hideout_info[i] == 'No' else 'tab:orange' for i in range(len(hideout_info))]
ax1.bar(x1 - width/2, ade[::2], width=0.4, label='Train')
ax1.bar(x1 + width/2, ade[1::2], width=0.4, label='Test')

# Add labels and legend for ADE plot
ax1.set_title('Average Displacement Error (ADE)', fontsize=16)
ax1.set_ylabel('Distance (km)', fontsize=14, weight='bold')
ax1.tick_params(axis='y', labelsize=12)
ax1.set_xticks(x1)
ax1.set_xticklabels(model_types, fontsize=14, weight='bold')
ax1.legend(fontsize=12)

plt.savefig('ade_comparison.png')

# Create the figure and axis objects for Containment plot
fig2, ax2 = plt.subplots(figsize=(8, 6))

# Create arrays for the x-axis and bar positions
x2= np.arange(0, len(model_types)*2, 2)

# Create the bar chart for containment data
# colors = ['tab:blue' if hideout_info[i] == 'No' else 'tab:orange' for i in range(len(hideout_info))]
ax2.bar(x2 - width/2, containment[::2], width=0.4, label='Train')
ax2.bar(x2 + width/2, containment[1::2], width=0.4,  label='Test')

# Add labels and legend for Containment plot
ax2.set_title('Containment Percentage', fontsize=16)
ax2.set_ylabel('Containment (%)', fontsize=14, weight='bold')
ax2.tick_params(axis='y', labelsize=12)
ax2.set_xticks(x2)
ax2.set_xticklabels(model_types, fontsize=14, weight='bold')
ax2.legend(fontsize=12)

# Show the plots
# plt.show()
plt.savefig('containment_comparison.png')