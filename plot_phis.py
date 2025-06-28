import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

s_start = -200
s_end = 200

parameter_data = np.loadtxt("parameters.csv", delimiter=",", skiprows=1)
c_0, t_state, t_len, tau, eps_val, a, b, lam_val, zeta, D, M, s_start, s_end, s_N = parameter_data

log_data = np.loadtxt("log.csv", delimiter=",", skiprows=1)
k_values = np.astype(np.unique(log_data[:, 0]), int) # List of frame indices
t_values = np.unique(np.round(log_data[:, 1], decimals=5))
t_values = t_values[k_values % 10000 == 0]
k_values = k_values[k_values % 10000 == 0]

fig, ax = plt.subplots()

# Read the first frame to initialize the plot and colorbar
phi0 = np.load("phi_0.npy")
im = ax.imshow(phi0, extent=[s_start, s_end, s_start, s_end], cmap='inferno', vmin=-1.5, vmax=1.5)  # set fixed vmin/vmax if possible
cbar = fig.colorbar(im, ax=ax)
title = ax.set_title("Iteration = 0")

# Animation update function
def update(frame):
    k = k_values[frame]
    phi = np.load(f"phi_{k}.npy")
    im.set_array(phi)

    # for debugging purposes
    # vmin, vmax = phi.min(), phi.max()
    # im.set_clim(vmin, vmax)        # Update the image color scale
    # cbar.update_normal(im)         # Redraw colorbar to reflect new limits

    title.set_text(f"Iteration = {k}, time = {t_values[frame]}")
    return [im, title]

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=len(k_values), interval=40, blit=False)
#FFwriter = animation.FFMpegWriter(fps=10)
ani.save("vid.gif", writer='pillow')
plt.show()
