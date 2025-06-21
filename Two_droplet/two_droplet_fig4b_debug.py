import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson

# --- Model Parameters ---
A = 1.0         # Strength of double-well potential
K = 1.0         # Interfacial energy
xi = np.sqrt(K / (2 * A))  # Interfacial width
y = np.linspace(-10*xi, 10*xi, 2000)

# --- Tanh Profile and Derivatives ---
phi = np.tanh(y / xi)
dphi_dy = (1 / xi) * (1 - phi**2)
weight = dphi_dy**2

# --- Grid Resolution ---
N = 200
lam_vals = np.linspace(-5, 5, N)
zeta_vals = np.linspace(-5, 5, N)
phase_map = np.zeros((N, N))

# --- Loop through (λ, ζ) ---
for i, lam in enumerate(lam_vals):
    for j, zet in enumerate(zeta_vals):
        exp_term = np.exp((zet - 2 * lam) * phi / K)
        S0 = simpson(weight, y)
        S1 = simpson(weight * exp_term, y)
        
        sigma_drop = K * (zet - 2 * lam) * (zet * S0 - 2 * lam * S1)
        sigma_bubble = K * (-zet - 2 * -lam) * (-zet * S0 - 2 * -lam * S1)  # Duality
        
        if sigma_drop < 0:
            region = 1  # Region B: Clusters
        elif sigma_bubble < 0:
            region = 2  # Region C: Bubbles
        else:
            region = 0  # Region A: Normal
        phase_map[j, i] = region

# --- Plot ---
cmap = plt.cm.get_cmap("Set2", 3)
plt.figure(figsize=(8, 6))
plt.contourf(lam_vals, zeta_vals, phase_map, levels=[-0.5, 0.5, 1.5, 2.5],
             colors=["lightblue", "plum", "orange"])
plt.colorbar(ticks=[0, 1, 2], label="Region")
plt.xlabel(r"$\lambda$", fontsize=14)
plt.ylabel(r"$\zeta$", fontsize=14)
plt.title("Figure 4a: Mean-Field Phase Diagram of Active Model B⁺", fontsize=15)
plt.grid(True)
plt.tight_layout()
plt.show()
