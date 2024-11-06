import matplotlib.pyplot as plt
import numpy as np

# Create some dummy reward data
t = np.arange(0, 100)
mean_reward = np.random.normal(size=100).cumsum()
std_reward = np.random.normal(scale=0.5, size=100)

# Plot the mean reward curve
plt.plot(t, mean_reward)

# Shade the area between the upper and lower bounds
plt.fill_between(t, mean_reward + std_reward, mean_reward - std_reward, alpha=0.2)

# Add labels and title
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Reward Curve with Upper and Lower Bounds')

# Show the plot
plt.show()
plt.savefig("test.png")