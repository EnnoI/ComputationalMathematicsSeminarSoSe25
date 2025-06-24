import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import label, center_of_mass
import sys
from tqdm import trange

# --- Parameters ---
L = 128        # Box size
N = 256        # Grid points
dx = L / N
x = np.linspace(-L/2, L/2, N, endpoint=False)
y = np.linspace(-L/2, L/2, N, endpoint=False)
X, Y = np.meshgrid(x, y, indexing="ij")

R1 = 10   # Radius of big droplet
R2 = 6    # Radius of small droplet

# --- Build 2-droplet initial condition ---
def build_two_droplets():
    phi = -1 * np.ones_like(X)
    d1 = (X+20)**2 + Y**2 < R1**2
    d2 = (X-20)**2 + Y**2 < R2**2
    phi[d1] = 1
    phi[d2] = 1
    return phi

# --- AMB+ Parameters ---
A, B, K = 1.0, 1.0, 1.0
D = 0.0     # Noise OFF
M = 1.0

def run_sim(phi0, lam, zeta, tau=0.01, T=100.0, save_every=100):
    kx = np.fft.fftfreq(N, d=dx) * 2*np.pi
    ky = np.fft.fftfreq(N, d=dx) * 2*np.pi
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    K2 = KX**2 + KY**2
    K4 = K2**2
    dt = tau
    steps = int(T / dt)
    
    phi = phi0.copy()
    f_phi = np.fft.fft2(phi)

    log = []
    
    for step in trange(steps):
        phi = np.real(np.fft.ifft2(f_phi))
        phi3 = phi**3
        
        dphi_dx = np.real(np.fft.ifft2(1j*KX*f_phi))
        dphi_dy = np.real(np.fft.ifft2(1j*KY*f_phi))
        grad_phi2 = dphi_dx**2 + dphi_dy**2
        lap_phi = np.real(np.fft.ifft2(-K2 * f_phi))
        
        Jzeta_x = np.fft.fft2(lap_phi * dphi_dx)
        Jzeta_y = np.fft.fft2(lap_phi * dphi_dy)
        Jzeta = 1j*KX*Jzeta_x + 1j*KY*Jzeta_y

        mu = -A*phi + B*phi3 - K*lap_phi + lam*grad_phi2
        f_mu = np.fft.fft2(mu)

        rhs = -K2*f_mu - zeta*Jzeta
        f_phi = (f_phi + dt*M*rhs) / (1 + dt*M*K4)

        # Track every few steps
        if step % save_every == 0:
            phi = np.real(np.fft.ifft2(f_phi))
            R_big, R_small = measure_radii(phi)
            log.append([step*dt, R_big, R_small])

    return np.array(log)

# --- Radius detection ---
def measure_radii(phi):
    mask = phi > 0.0
    labeled, n = label(mask)
    if n < 2:
        return [0, 0]

    sizes = [(labeled==i+1).sum() for i in range(n)]
    areas = np.array(sizes)*dx*dx
    radii = np.sqrt(areas/np.pi)
    radii.sort()  # ascending
    return [radii[-1], radii[-2]]

# --- Region classification ---
def classify(dlog):
    R1_0, R2_0 = dlog[0,1], dlog[0,2]
    R1_f, R2_f = dlog[-1,1], dlog[-1,2]
    if R1_f > R1_0 and R2_f < R2_0:
        return "A"
    elif R2_f > R2_0 and R1_f < R1_0:
        return "B"
    elif R1_f > R1_0 and R2_f > R2_0:
        return "C"
    else:
        return "Undetermined"

# --- Main script ---
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python simulate_fig4b_clean.py <zeta> <lambda>")
        sys.exit(1)
    
    zeta_val = float(sys.argv[1])
    lambda_val = float(sys.argv[2])

    phi0 = build_two_droplets()
    log = run_sim(phi0, lambda_val, zeta_val, T=1000.0)

    np.savetxt("radii_log.csv", log, delimiter=",", header="time,R1,R2")
    region = classify(log)

    with open("region.txt", "w") as f:
        f.write(region + "\n")

    print(f"✓ ζ={zeta_val:.2f}, λ={lambda_val:.2f} → Region {region}")
