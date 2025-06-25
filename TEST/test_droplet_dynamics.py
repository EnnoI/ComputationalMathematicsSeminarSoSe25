import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftfreq
import sys
import os

# --- Simulation parameters ---
L = 256               # Grid size
T = 2000              # Number of time steps
dt = 0.001            # Time step size
dx = 1.0              # Spatial resolution
A = 0.25
K = 1.0
K1 = 0.0             # Usually zero as per paper
D = 0.0               # No noise

# --- Get lambda and zeta from command-line ---
lambda_val = float(sys.argv[1])
zeta_val = float(sys.argv[2])

def laplacian(f):
    kx = fftfreq(L, d=dx) * 2 * np.pi
    ky = fftfreq(L, d=dx) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    k2 = KX**2 + KY**2
    return np.real(ifft2(-k2 * fft2(f)))

def grad_squared(f):
    dfx = np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)
    dfy = np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)
    return (dfx**2 + dfy**2) / (4 * dx**2)

def compute_mu(phi):
    mu_eq = A * phi + phi**3 - K * laplacian(phi)
    mu_act = lambda_val * grad_squared(phi)
    return mu_eq + mu_act

def compute_current(phi):
    mu = compute_mu(phi)
    dphi_x = (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) / (2 * dx)
    dphi_y = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / (2 * dx)
    lap_phi = laplacian(phi)
    Jx = - (np.roll(mu, -1, axis=0) - np.roll(mu, 1, axis=0)) / (2 * dx) + zeta_val * lap_phi * dphi_x
    Jy = - (np.roll(mu, -1, axis=1) - np.roll(mu, 1, axis=1)) / (2 * dx) + zeta_val * lap_phi * dphi_y
    return Jx, Jy

def divergence(Jx, Jy):
    dJx = (np.roll(Jx, -1, axis=0) - np.roll(Jx, 1, axis=0)) / (2 * dx)
    dJy = (np.roll(Jy, -1, axis=1) - np.roll(Jy, 1, axis=1)) / (2 * dx)
    return dJx + dJy

def init_two_droplets(R_large, R_small):
    x = np.arange(L)
    y = np.arange(L)
    X, Y = np.meshgrid(x, y, indexing='ij')
    xc1, yc1 = L // 3, L // 2
    xc2, yc2 = 2 * L // 3, L // 2
    dist1 = np.sqrt((X - xc1)**2 + (Y - yc1)**2)
    dist2 = np.sqrt((X - xc2)**2 + (Y - yc2)**2)
    phi = -1 * np.ones((L, L))
    phi[dist1 < R_large] = 1
    phi[dist2 < R_small] = 1
    return phi

def measure_radius(phi):
    labeled = (phi > 0).astype(int)
    from scipy.ndimage import label, center_of_mass
    from scipy.ndimage import measurements
    lw, num = label(labeled)
    if num < 2:
        return 0.0, 0.0
    radii = []
    for i in range(1, num+1):
        mask = lw == i
        area = np.sum(mask)
        r = np.sqrt(area / np.pi)
        radii.append(r)
    radii.sort(reverse=True)
    return radii[0], radii[1]

# --- Main Simulation ---
phi = init_two_droplets(45, 25)

log_file = f"run_lambda{lambda_val}_zeta{zeta_val}.log"
log_csv = "radii_log.csv"

with open(log_file, "w") as f:
    for t in range(T):
        Jx, Jy = compute_current(phi)
        divJ = divergence(Jx, Jy)
        phi += dt * (-divJ)
        if t % 50 == 0:
            R1, R2 = measure_radius(phi)
            mass = np.sum(phi)
            f.write(f"[{t * dt:.1f}] R1={R1:.2f}, R2={R2:.2f}, mass={mass:.4f}\n")
            print(f"[{t * dt:.1f}] R1={R1:.2f}, R2={R2:.2f}, mass={mass:.4f}")
            with open(log_csv, "a") as fc:
                fc.write(f"{t},{R1},{R2}\n")
            if np.isnan(phi).any():
                print("NaN detected, aborting.")
                break

# --- Classify Region ---
R1_final, R2_final = measure_radius(phi)
R1_init, R2_init = 45, 25

dR_large = R1_final - R1_init
dR_small = R2_final - R2_init

region = "Undetermined"
if dR_large > 0 and dR_small < 0:
    region = "A"
elif dR_large < 0 and dR_small > 0:
    region = "B"
elif dR_large > 0 and dR_small > 0:
    region = "C"

print(f"\u0394R_small = {dR_small:.3f}, \u0394R_large = {dR_large:.3f}")
print(f"Done: λ={lambda_val}, ζ={zeta_val} → Region {region}")

# Save classification
run_dir = f"runs/lam{lambda_val}_zeta{zeta_val}"
os.makedirs(run_dir, exist_ok=True)
with open(os.path.join(run_dir, "region.txt"), "w") as f:
    f.write(region)
