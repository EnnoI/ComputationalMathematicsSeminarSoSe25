#!/usr/bin/env python3
"""
compute_coexistence.py

Compute coexistence densities φ⁻(R) and φ⁺(R) for Active Model B+,
including flat‐interface binodals and Laplace‐pressure curvature corrections.
Saves (R, φ_minus, φ_plus) into 'phi_values.npz'.
"""
import numpy as np
from scipy.optimize import fsolve

# ── Model parameters ─────────────────────────────────────────────────────────
A           = 0.25    # local free‐energy coefficient
K           = 1.0     # stiffness
zeta        = -4.0    # activity parameter ζ
lambda_val  = -1.0    # activity parameter λ
d           = 3       # spatial dimension

# ── Thermodynamics ──────────────────────────────────────────────────────────
def f0(phi):
    return A*phi + phi**3

def psi(phi):
    Δ = zeta - 2*lambda_val
    return (K/Δ) * (np.exp((Δ/K)*phi) - 1.0)

def g(phi):
    Δ = zeta - 2*lambda_val
    term1 = (6*A*K**4)/(Δ**4) - (A*K**2)/(Δ**2) - (A*K*np.exp(Δ*phi/K))/(Δ**4)
    term2 = -Δ**3*phi*(phi**2 - 1) + 6*K**3 - 6*zeta*K**2*phi + Δ**2*K*(3*phi**2 - 1)
    return term1 * term2

def pseudopressure(phi, mu):
    return psi(phi)*mu - g(phi)

# ── 1) Flat‐interface binodals ──────────────────────────────────────────────
def binodal_eqns(vars):
    phi_m, phi_p = vars
    mu = f0(phi_m)
    return [
        f0(phi_p) - mu,
        pseudopressure(phi_p, mu) - pseudopressure(phi_m, mu)
    ]

phi_minus_eq, phi_plus_eq = fsolve(binodal_eqns, [-1.0, 1.0])
print(f"Flat binodals: φ⁻={phi_minus_eq:.5f}, φ⁺={phi_plus_eq:.5f}")

# ── 2) Curvature corrections (Laplace pressure) ───────────────────────────
sigma_eq = np.sqrt(8*K*A/9)

def curved_eqns(vars, R):
    phi_m, phi_p = vars
    ΔLP = (d - 1)*sigma_eq / R
    return [
        f0(phi_p) - f0(phi_m) - ΔLP,
        pseudopressure(phi_p, f0(phi_p))
          - pseudopressure(phi_m, f0(phi_m)) - ΔLP
    ]

# Choose a grid of radii to precompute
R_values       = np.linspace(5.0, 100.0, 50)
phi_minus_vals = []
phi_plus_vals  = []

for R in R_values:
    sol = fsolve(curved_eqns, [phi_minus_eq, phi_plus_eq], args=(R,))
    phi_minus_vals.append(sol[0])
    phi_plus_vals.append(sol[1])

# ── Save to .npz ────────────────────────────────────────────────────────────
np.savez('phi_values.npz',
         R=R_values,
         phi_minus=np.array(phi_minus_vals),
         phi_plus =np.array(phi_plus_vals))
print("Saved phi_values.npz with coexistence densities.")
