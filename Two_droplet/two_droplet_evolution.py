#!/usr/bin/env python3
"""
two_droplet_demo.py

Demonstrate two‐droplet evolution (Fig. 4b) using a simple 1/R curvature
approximation for φ⁻(R), φ⁺(R). Run this as-is to get R₁(t), R₂(t).
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Model parameters
A = 0.25
K = 1.0
d = 3

# equilibrium interface tension
sigma_eq = np.sqrt(8*K*A/9)
fpp = A + 3      # f''(φ0) at φ0=1

# approximate coexistence densities
def phi_minus(R):
    return -1 + (d-1)*sigma_eq/(fpp * R)

def phi_plus(R):
    return +1 - (d-1)*sigma_eq/(fpp * R)

# f0 derivative
def f0(phi):
    return A*phi + phi**3

# right‐hand side of Eqs. E1–E2
def rhs(t, y):
    R1, R2 = y
    dp1 = phi_plus(R1) - phi_minus(R1)
    dp2 = phi_plus(R2) - phi_minus(R2)
    num1 = f0(phi_minus(R2)) - f0(phi_minus(R1))
    num2 = f0(phi_minus(R1)) - f0(phi_minus(R2))
    return [num1/(R1*dp1), num2/(R2*dp2)]

# initial radii at t=100
y0 = [15.0, 10.0]
# times from 10^2 to 10^5
t_eval = np.logspace(2, 5, 300)

# integrate
sol = solve_ivp(rhs,
                (t_eval[0], t_eval[-1]),
                y0,
                t_eval=t_eval,
                method='RK45',
                atol=1e-8, rtol=1e-6)

# plot
plt.figure(figsize=(6,4))
plt.loglog(sol.t, sol.y[0], 'r-', label=r'$R_1(t)$')
plt.loglog(sol.t, sol.y[1], 'b-', label=r'$R_2(t)$')
plt.xlabel('Time $t$')
plt.ylabel('Radius $R$')
plt.title('Two‐Droplet Evolution (Demo)')
plt.legend()
plt.tight_layout()
plt.show()
