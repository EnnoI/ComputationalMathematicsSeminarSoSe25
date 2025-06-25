import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, sum as ndi_sum
import os

# --- Parameters ---
N = 128
L = 128.0
dx = L / N
dt = 0.0025
steps = int(200 / dt)
snapshot_interval = 2000

A, Bc, K = 1.0, 1.0, 1.0
lam = -1.0   # Try region B: λ = -1, ζ = -4
zeta = -4.0

x = np.linspace(-L/2, L/2, N, endpoint=False)
y = np.linspace(-L/2, L/2, N, endpoint=False)
X, Y = np.meshgrid(x, y, indexing="ij")

kx = np.fft.fftfreq(N, d=dx) * 2*np.pi
ky = np.fft.fftfreq(N, d=dx) * 2*np.pi
KX, KY = np.meshgrid(kx, ky, indexing="ij")
K2 = KX**2 + KY**2
dealias = K2 < (0.67 * K2.max())

fft = np.fft.fft2
ifft = lambda f: np.fft.ifft2(f).real

# --- Initialize a single small droplet ---
phi = -0.9 * np.ones((N, N))
R = 8
phi[X**2 + Y**2 < R**2] = -1.0
phi += 0.01 * np.random.randn(*phi.shape)
f_phi = fft(phi)

radii = []

# --- Run ---
for step in range(steps + 1):
    phi = ifft(f_phi)
    phi = np.clip(phi, -2.0, 2.0)

    # Measure radius every few steps
    if step % snapshot_interval == 0:
        binary = (phi > 0.05).astype(int)
        labeled, num = label(binary)
        areas = ndi_sum(binary, labeled, index=range(1, num + 1))
        if len(areas) > 0:
            r = np.sqrt(areas[0] / np.pi)
        else:
            r = 0
        radii.append(r)

    # Dynamics
    dphi_dx = np.clip(ifft(1j * KX * f_phi), -10, 10)
    dphi_dy = np.clip(ifft(1j * KY * f_phi), -10, 10)
    grad_phi2 = np.clip(dphi_dx**2 + dphi_dy**2, 0, 100)

    lap_phi = ifft(-K2 * f_phi)
    mu = -A * phi + Bc * phi**3 - K * lap_phi + lam * grad_phi2
    f_mu = fft(mu)

    lap_phi_fx = fft(np.clip(lap_phi * dphi_dx, -1e2, 1e2))
    lap_phi_fy = fft(np.clip(lap_phi * dphi_dy, -1e2, 1e2))
    Jzeta = 1j * KX * lap_phi_fx + 1j * KY * lap_phi_fy

    dfdt = -K2 * f_mu - zeta * Jzeta
    dfdt = np.clip(dfdt, -1e4, 1e4)
    dfdt *= dealias
    f_phi += dt * dfdt

# --- Plotting radius over time ---
plt.plot(np.linspace(0, 200, len(radii)), radii, marker='o')
plt.xlabel("Time")
plt.ylabel("Droplet Radius")
plt.title(f"λ={lam}, ζ={zeta} — Radius Evolution")
plt.grid(True)
plt.tight_layout()
plt.show()
