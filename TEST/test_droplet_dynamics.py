import numpy as np
from scipy.ndimage import label
import os, sys

# === Command-line arguments ===
if len(sys.argv) != 3:
    print("Usage: python simulate_two_droplets_worker.py <lambda> <zeta>")
    sys.exit(1)
lam = float(sys.argv[1])
zeta = float(sys.argv[2])

# === Simulation parameters ===
N = 128               # Grid points per dimension
L = 128.0             # Physical size of the domain
dx = L / N           # Spatial resolution
dt = 0.001           # Time step
steps = int(200 / dt) # Total steps to reach t=200
log_interval = 50     # Print/log every 50 steps

# AMB+ model parameters
A, B, K = 1.0, 1.0, 1.0

# Droplet initialization parameters
R_large = 10.0        # Initial large droplet radius
R_small = 6.0         # Initial small droplet radius
sep = 30.0            # Center separation on x-axis
width = 3.0           # Interface width for tanh

# Create output directory
out_dir = f"runs/lam{lam}_zeta{zeta}"
os.makedirs(out_dir, exist_ok=True)
log_csv = os.path.join(out_dir, "radii_log.csv")
with open(log_csv, 'w') as f:
    f.write("step,time,R_large,R_small\n")

# === Coordinate grid ===
x = np.linspace(-L/2, L/2, N, endpoint=False)
y = np.linspace(-L/2, L/2, N, endpoint=False)
X, Y = np.meshgrid(x, y, indexing='ij')

# === Helper functions ===

def laplacian(f):
    # 5-point finite-difference Laplacian
    return (
        np.roll(f, -1, axis=0) + np.roll(f, 1, axis=0) +
        np.roll(f, -1, axis=1) + np.roll(f, 1, axis=1) -
        4 * f
    ) / dx**2


def gradient(f):
    dfx = (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2 * dx)
    dfy = (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2 * dx)
    return dfx, dfy


def initialize_two_droplets(R1, R2, width):
    # Smooth tanh droplets and combine
    r1 = np.sqrt((X + sep)**2 + Y**2)
    r2 = np.sqrt((X - sep)**2 + Y**2)
    phi1 = np.tanh((R1 - r1) / width)
    phi2 = np.tanh((R2 - r2) / width)
    phi = np.maximum(phi1, phi2)
    return np.clip(phi, -1.0, 1.0)


def measure_radii(phi):
    # Detect two largest droplet radii
    binary = (phi > 0).astype(int)
    lw, num = label(binary)
    radii = []
    for lbl in range(1, num+1):
        area = np.sum(lw == lbl)
        if area > 20:
            radii.append(np.sqrt(area / np.pi))
    radii.sort(reverse=True)
    if len(radii) >= 2:
        return radii[0], radii[1]
    elif len(radii) == 1:
        return radii[0], 0.0
    else:
        return 0.0, 0.0

# === Initialize field ===
phi = initialize_two_droplets(R_large, R_small, width)
R_init_large, R_init_small = measure_radii(phi)
print(f"Initial radii: R_large={R_init_large:.2f}, R_small={R_init_small:.2f}")

# === Time evolution ===
for step in range(steps+1):
    # Compute gradients and laplacian of phi
    dphi_x, dphi_y = gradient(phi)
    grad_phi2 = dphi_x**2 + dphi_y**2
    lap_phi = laplacian(phi)

    # Chemical potential: μ_eq + λ |∇φ|²
    mu_eq = A * phi + B * phi**3 - K * lap_phi
    mu = mu_eq + lam * grad_phi2

    # Diffusive flux: ∇²μ
    lap_mu = laplacian(mu)

    # Active ζ term: divergence of (∇²φ ∇φ)
    Jz_x = lap_phi * dphi_x
    Jz_y = lap_phi * dphi_y
    divJz = (
        (np.roll(Jz_x, -1, axis=0) - np.roll(Jz_x, 1, axis=0)) +
        (np.roll(Jz_y, -1, axis=1) - np.roll(Jz_y, 1, axis=1))
    ) / (2 * dx)

    # Update φ conserving mass
    phi += dt * (lap_mu - zeta * divJz)

    # Stabilize
    phi = np.clip(phi, -2.0, 2.0)

    # Logging
    if step % log_interval == 0:
        t = step * dt
        R1, R2 = measure_radii(phi)
        with open(log_csv, 'a') as f:
            f.write(f"{step},{t:.4f},{R1:.4f},{R2:.4f}\n")
        print(f"[{t:.1f}] R_large={R1:.2f}, R_small={R2:.2f}")

# === Classify region ===
R_final_large, R_final_small = measure_radii(phi)
dR_large = R_final_large - R_init_large
dR_small = R_final_small - R_init_small

if dR_large > 0 and dR_small < 0:
    region = 'A'
elif dR_large < 0 and dR_small > 0:
    region = 'B'
elif dR_large > 0 and dR_small > 0:
    region = 'C'
else:
    region = 'Undetermined'

with open(os.path.join(out_dir, 'region.txt'), 'w') as f:
    f.write(region)

print(f"ΔR_small={dR_small:.3f}, ΔR_large={dR_large:.3f} → Region {region}")