import numpy as np
import matplotlib.pyplot as plt

# Set the grid size and range
w = 7
points = 100
Y, X = np.mgrid[-w:w:points*1j, -w:w:points*1j]

# Generate dummy data for the vector field
U = np.sin(X) * np.cos(Y)
V = -np.cos(X) * np.sin(Y)

# Create a figure and axis
fig, ax = plt.subplots(figsize=(8, 8))

# Plot the vector field using quiver
ax.quiver(X, Y, U, V, np.sqrt(U**2 + V**2), cmap='coolwarm', scale=50.0, width=0.015, pivot='mid')

# Set the axis limits and remove ticks
ax.set_xlim(-w, w)
ax.set_ylim(-w, w)
ax.set_xticks([])
ax.set_yticks([])

# Add a title
ax.set_title("Flow Field Visualization", fontsize=16)

# Display the plot
# plt.show()

plt.savefig("quiver.png")