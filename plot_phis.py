import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

s_start = -32*np.pi
s_end = 32*np.pi

t_state, t_len, tau, eps_val, a, b, lam_val, zeta, D, M, s_start, s_end, s_N = np.loadtxt("parameters.csv", delimiter=",", skiprows=1)

fig, ax = plt.subplots()

# Read the first frame to initialize the plot and colorbar
phi0 = np.load("phi_0.npy")
im = ax.imshow(phi0, extent=[s_start, s_end, s_start, s_end], cmap='inferno', vmin=-1, vmax=1)  # set fixed vmin/vmax if possible
cbar = fig.colorbar(im, ax=ax)
title = ax.set_title("Iteration = 0")

# List of frame indices
k_values = list(range(0, 1+26000, 2000))

# Animation update function
def update(frame):
    k = k_values[frame]
    phi = np.load(f"phi_{k}.npy")
    im.set_array(phi)

    # for debugging purposes
    # vmin, vmax = phi.min(), phi.max()
    # im.set_clim(vmin, vmax)        # Update the image color scale
    # cbar.update_normal(im)         # Redraw colorbar to reflect new limits

    title.set_text(f"Iteration = {k}")
    return [im, title]

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=len(k_values), interval=200, blit=False)
ani.save("vid.gif", writer='pillow', fps=5)
plt.show()
