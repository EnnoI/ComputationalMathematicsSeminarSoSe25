import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os

# --- Parameter grid (same as Figure 4) ---
lambdas = np.linspace(-2.0, 2.0, 21)  # 21 points: -2.0, -1.8, ..., 2.0
zetas = np.linspace(-4.0, 4.0, 21)

region_codes = {'A': 1, 'B': 2, 'C': 3, 'Undetermined': 0}
code_labels = {1: 'A', 2: 'B', 3: 'C', 0: 'Undetermined'}
region_map = np.zeros((len(lambdas), len(zetas)), dtype=int)

# --- Output directory for results ---
os.makedirs("full_simulation", exist_ok=True)

# --- Loop over λ and ζ ---
for i, lam in enumerate(lambdas):
    for j, zeta in enumerate(zetas):
        tag = f"lam{lam}_zeta{zeta}"
        run_dir = f"runs/{tag}"
        region_file = os.path.join(run_dir, "region.txt")

        if os.path.exists(region_file):
            # Already simulated — just read
            with open(region_file) as f:
                region = f.read().strip()
        else:
            # Run the simulation
            print(f"Running λ={lam:.2f}, ζ={zeta:.2f}")
            try:
                subprocess.run(
                    ["python", "simulate_two_droplets_worker.py", str(lam), str(zeta)],
                    check=True,
                )
                with open(region_file) as f:
                    region = f.read().strip()
            except Exception as e:
                print(f"Simulation failed for λ={lam}, ζ={zeta}: {e}")
                region = "Undetermined"

        region_map[i, j] = region_codes.get(region, 0)

# --- Plotting ---
plt.figure(figsize=(8, 6))
extent = [zetas[0], zetas[-1], lambdas[0], lambdas[-1]]  # ζ horizontal, λ vertical
plt.imshow(region_map.T, origin='lower', extent=extent, aspect='auto', cmap='viridis')
plt.colorbar(
    ticks=[0,1,2,3],
    label="Region (A=1, B=2, C=3, Undetermined=0)"
)
plt.clim(-0.5, 3.5)
plt.xlabel("ζ (zeta)")
plt.ylabel("λ (lambda)")
plt.title("Simulation-Derived Region Map (Reproduction of Fig. 4c)")
plt.grid(False)
plt.tight_layout()
plt.show()
