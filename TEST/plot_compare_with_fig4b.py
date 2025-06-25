import numpy as np
import matplotlib.pyplot as plt

# Update this to match your actual file path
log_path = "C:/Users/anuhe/Documents/GitHub/ComputationalMathematicsSeminarSoSe25/Two_droplet/runs/lam0.5_zeta-1.0/radii_log.csv"


# Read the log file
time, r1, r2 = [], [], []
with open(log_path) as f:
    next(f)  # skip header
    for line in f:
        t, rl, rr, *_ = line.strip().split(",")
        time.append(float(t))
        r1.append(float(rl))
        r2.append(float(rr))

time = np.array(time)
r1 = np.array(r1)
r2 = np.array(r2)

# Plotting both droplet radii
plt.figure(figsize=(8, 5))
plt.plot(time, r1, 'r-', label="Droplet 1 (left)")
plt.plot(time, r2, 'b-', label="Droplet 2 (right)")
plt.xlabel("Time")
plt.ylabel("Droplet Radius")
plt.title("Two-Droplet Radius Evolution — λ = 0.5, ζ = –1.0")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
