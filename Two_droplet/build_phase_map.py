# Save this as build_phase_map.py

import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os

lambdas = np.linspace(-3, 3, 20)
zetas = np.linspace(-5, 5, 20)
region_map = np.full((len(zetas), len(lambdas)), -1)

region_codes = {'A': 0, 'B': 1, 'C': 2, 'Undetermined': 3}
region_names = ['Region A', 'Region B', 'Region C', 'Undetermined']
region_colors = ['lightblue', 'plum', 'orange', 'gray']

for i, lam in enumerate(lambdas):
    for j, zet in enumerate(zetas):
        lam_str = f"{lam:.2f}"
        zet_str = f"{zet:.2f}"
        run_dir = f"runs/lam{lam_str}_zeta{zet_str}"
        region_file = f"{run_dir}/region.txt"

        if not os.path.exists(region_file):
            print(f"Running λ={lam_str}, ζ={zet_str}")
            subprocess.run(["python", "simulate_two_droplets_worker.py", lam_str, zet_str])

        if os.path.exists(region_file):
            with open(region_file, "r") as f:
                label = f.read().strip()
                region_map[j, i] = region_codes.get(label, 3)
        else:
            print(f"Warning: No result for λ={lam_str}, ζ={zet_str}")

# Plotting
plt.figure(figsize=(8, 6))
cmap = plt.cm.get_cmap('Pastel1', 4)
cmap = plt.matplotlib.colors.ListedColormap(region_colors)
bounds = np.arange(-0.5, 4.5, 1)
norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)

plt.imshow(region_map, extent=[lambdas[0], lambdas[-1], zetas[0], zetas[-1]],
           origin='lower', cmap=cmap, norm=norm, aspect='auto')

plt.xlabel(r'$\lambda$', fontsize=14)
plt.ylabel(r'$\zeta$', fontsize=14)
plt.title('Numerical Figure 4a (Two-Droplet Dynamics)', fontsize=15)
cbar = plt.colorbar(ticks=[0, 1, 2, 3])
cbar.ax.set_yticklabels(region_names)
plt.tight_layout()
plt.savefig("figure4a_dynamic.png", dpi=300)
plt.show()
