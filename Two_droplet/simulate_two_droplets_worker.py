import numpy as np
from scipy.ndimage import label, sum as ndi_sum, center_of_mass
import os
import sys

# --- Parse command-line args ---
if len(sys.argv) != 3:
    print("Usage: python simulate_two_droplets_worker.py <lambda> <zeta>")
    sys.exit(1)

lam = float(sys.argv[1])
zeta = float(sys.argv[2])
tag = f"lam{lam}_zeta{zeta}"

# --- Simulation parameters ---
N = 128
L = 128.0
dx = L / N
dt = 0.0025
steps = int(2000 / dt)
save_interval = 100

A, B, K = 1.0, 1.0, 1.0
sep = 30  # x-offset of droplets

x = np.linspace(-L / 2, L / 2, N, endpoint=False)
y = np.linspace(-L / 2, L / 2, N, endpoint=False)
X, Y = np.meshgrid(x, y, indexing="ij")

kx = np.fft.fftfreq(N, d=dx) * 2 * np.pi
ky = np.fft.fftfreq(N, d=dx) * 2 * np.pi
KX, KY = np.meshgrid(kx, ky, indexing="ij")
K2 = KX**2 + KY**2
dealias = K2 < (0.67 * K2.max())

fft = np.fft.fft2
ifft = lambda f: np.fft.ifft2(f).real

# --- Initialize two droplets ---
def initialize_two_droplets(R1=10, R2=6):
    phi = -0.9 * np.ones((N, N))
    phi[(X + sep)**2 + Y**2 < R1**2] = 1.0
    phi[(X - sep)**2 + Y**2 < R2**2] = 1.0
    phi += 0.01 * np.random.randn(*phi.shape)
    return phi

phi = initialize_two_droplets()
f_phi = fft(phi)

# --- Setup output ---
out_dir = f"runs/{tag}"
os.makedirs(out_dir, exist_ok=True)
log_path = f"{out_dir}/radii_log.csv"
with open(log_path, "w") as log:
    log.write("time,radius_left,radius_right,num_blobs,max_phi,min_phi\n")

radii_history = []

# --- Time evolution loop ---
for step in range(steps + 1):
    phi = ifft(f_phi)
    phi = np.clip(phi, -2.0, 2.0)

    if not np.isfinite(phi).all():
        print(f"NaN detected at step {step}. Aborting.")
        break

    if step % save_interval == 0:
        binary = (phi > 0.05).astype(int)
        labeled, num = label(binary)
        areas = ndi_sum(binary, labeled, index=range(1, num + 1))
        COMs = center_of_mass(binary, labeled, index=range(1, num + 1))

        blob_data = [(a, com) for a, com in zip(areas, COMs) if a > 10]
        r_left = r_right = 0.0

        if len(blob_data) == 1:
            x_pos = x[int(round(blob_data[0][1][0]))]
            r = np.sqrt(blob_data[0][0] / np.pi)
            if x_pos < 0:
                r_left = r
            else:
                r_right = r
        elif len(blob_data) >= 2:
            blob_data.sort(key=lambda b: x[int(round(b[1][0]))])
            r_left = np.sqrt(blob_data[0][0] / np.pi)
            r_right = np.sqrt(blob_data[1][0] / np.pi)

        t_now = step * dt
        with open(log_path, "a") as log:
            log.write(f"{t_now:.2f},{r_left:.4f},{r_right:.4f},{len(blob_data)},{phi.max():.2e},{phi.min():.2e}\n")
        print(f"[{t_now:.1f}] RL={r_left:.2f}, RR={r_right:.2f}, n={len(blob_data)}, φmax={phi.max():.2e}")
        np.save(f"{out_dir}/phi_t{step:05d}.npy", phi)
        radii_history.append((r_left, r_right))

    dphi_dx = np.clip(ifft(1j * KX * f_phi), -10, 10)
    dphi_dy = np.clip(ifft(1j * KY * f_phi), -10, 10)
    grad_phi2 = np.clip(dphi_dx**2 + dphi_dy**2, 0, 100)

    lap_phi = ifft(-K2 * f_phi)
    phi3 = np.clip(phi, -2.0, 2.0) ** 3
    mu = -A * phi + B * phi3 - K * lap_phi + lam * grad_phi2
    f_mu = fft(mu)

    lap_phi_fx = fft(np.clip(lap_phi * dphi_dx, -1e2, 1e2))
    lap_phi_fy = fft(np.clip(lap_phi * dphi_dy, -1e2, 1e2))
    Jzeta = 1j * KX * lap_phi_fx + 1j * KY * lap_phi_fy

    dfdt = -K2 * f_mu - zeta * Jzeta
    dfdt = np.clip(dfdt, -1e4, 1e4)
    dfdt *= dealias
    f_phi += dt * dfdt

# --- Region classification ---
arr = np.array(radii_history)
if len(arr) == 0:
    region = "Undetermined"
else:
    growL = arr[-1, 0] - arr[0, 0]
    growR = arr[-1, 1] - arr[0, 1]
    if growL > 0.2 and growR < -0.2:
        region = "A"
    elif growR > 0.2 and growL < -0.2:
        region = "B"
    elif growL > 0.2 and growR > 0.2:
        region = "C"
    else:
        region = "Undetermined"

with open(f"{out_dir}/region.txt", "w") as f:
    f.write(region + "\n")
print(f"Done: λ={lam}, ζ={zeta} → Region {region}")
