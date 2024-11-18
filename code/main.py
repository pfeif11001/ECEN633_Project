import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Create a figure and axes
fig, ax = plt.subplots()

# Create a circle
circle = Circle((0, 0), 1, fill=False)

# Add the circle to the axes
ax.add_patch(circle)

# Set equal aspect ratio
ax.set_aspect('equal')

# Set the plot limits
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)

# Show the plot
plt.show()