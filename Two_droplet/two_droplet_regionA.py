import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os

# Grid settings
lambdas = np.linspace(-3, 3, 20)
zetas = np.linspace(-5, 5, 20)
region_map = np.full((len(zetas), len(lambdas)), -1)  # -1 means not yet classified

region_codes = {'A': 0, 'B': 1, 'C': 2, 'Undetermined': 3}
region_names = ['Region A', 'Region B', 'Region C', 'Undetermined']
region_colors = ['lightblue', 'plum', 'orange', 'gray']

for i, lam in enumerate(lambdas):
    for j, zet in enumerate(zetas):
        lam_str = f"{lam:.2f}"
        zet_str = f"{zet:.2f}"
        run_dir = f"runs/lam{lam_str}_zeta{zet_str}"
        region_file = f"{run_dir}/region.txt"

        # Run if results don't already exist
        if not os.path.exists(region_file):
            print(f"Running λ={lam:.2f}, ζ={zet:.2f}")
            subprocess.run(["python", "simulate_two_droplets_worker.py", lam_str, zet_str])

        # Parse result
        if os.path.exists(region_file):
            with open(region_file, "r") as f:
                label = f.read().strip()
                region_idx = region_codes.get(label, 3)
                region_map[j, i] = region_idx
        else:
            print(f"⚠ Missing result at λ={lam:.2f}, ζ={zet:.2f}")

# Plotting
plt.figure(figsize=(8,6))
cmap = plt.matplotlib.colors.ListedColormap(region_colors)
bounds = np.arange(-0.5, len(region_codes)+0.5, 1)
norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)

plt.imshow(region_map, extent=[lambdas[0], lambdas[-1], zetas[0], zetas[-1]],
           origin='lower', cmap=cmap, norm=norm, aspect='auto')

plt.xlabel(r'$\lambda$', fontsize=14)
plt.ylabel(r'$\zeta$', fontsize=14)
plt.title('Numerical Phase Diagram via Two-Droplet Simulation', fontsize=14)
cbar = plt.colorbar(ticks=range(len(region_names)))
cbar.ax.set_yticklabels(region_names)
plt.tight_layout()
plt.savefig("figure4a_dynamic.png", dpi=300)
plt.show()
