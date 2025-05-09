import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.animation as animation

s_start = -32*np.pi
s_end = 32*np.pi

fig, ax = plt.subplots()

# Read the first frame to initialize the plot and colorbar
phi0 = pd.read_csv("phi_0.csv", header=None).values
im = ax.imshow(phi0, extent=[s_start, s_end, s_start, s_end], cmap='inferno', vmin=-1, vmax=1)  # set fixed vmin/vmax if possible
cbar = fig.colorbar(im, ax=ax)
title = ax.set_title("Iteration = 0")

# List of frame indices
k_values = list(range(0, 1+100000, 10000))

# Animation update function
def update(frame):
    k = k_values[frame]
    phi = pd.read_csv(f"phi_{k}.csv", header=None).values
    im.set_array(phi)

    # vmin, vmax = phi.min(), phi.max()
    # im.set_clim(vmin, vmax)        # Update the image color scale
    # cbar.update_normal(im)         # Redraw colorbar to reflect new limits

    title.set_text(f"Iteration = {k}")
    return [im, title]

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=len(k_values), interval=200, blit=False)
ani.save("vid.gif", writer='pillow', fps=5)
plt.show()
