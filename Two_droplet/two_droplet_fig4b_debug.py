#!/usr/bin/env python3
"""
two_droplet_fig4b_ambplus.py

Full AMB+ theory for Fig 4(b):
 for each (ζ,λ) we'll
   1) find flat‐interface binodals φ⁻,φ⁺ by solving (10,11)
   2) add Laplace curvature ΔP=(d−1)σ_eq/R (Eqs E = 4)
   3) build φ⁻(R),φ⁺(R) tables and interpolate
   4) integrate the two‐droplet ODEs (E1–E2)
   5) plot R₁(t), R₂(t) on log–log axes
"""
import numpy as np
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
from scipy.integrate   import solve_ivp
import matplotlib.pyplot as plt

# shared constants
A = 0.25       # double-well parameter
K = 1.0        # stiffness
d = 3          # dimension
sigma_eq = np.sqrt(8*K*A/9)

def make_coexistence_functions(zeta, lam):
    """Returns two callables phi_minus(R), phi_plus(R) for this zeta,lam."""
    Δ = zeta - 2*lam

    def f0(phi):
        return A*phi + phi**3

    def psi(phi):
        return (K/Δ)*(np.exp((Δ/K)*phi) - 1.0)

    def g(phi):
        t1 = (6*A*K**4)/(Δ**4) - (A*K**2)/(Δ**2) - (A*K*np.exp(Δ*phi/K))/(Δ**4)
        t2 = (-Δ**3*phi*(phi**2 - 1)
              + 6*K**3
              - 6*zeta*K**2*phi
              + Δ**2*K*(3*phi**2 - 1))
        return t1 * t2

    def pseudoP(phi, mu):
        return psi(phi)*mu - g(phi)

    # 1) Flat‐interface binodal eqns (Eqs 10–11)
    def binodal_eq(vars):
        phim, phip = vars
        mu = f0(phim)
        return [
            f0(phip) - mu,
            pseudoP(phip, mu) - pseudoP(phim, mu)
        ]

    phim0, phip0 = fsolve(binodal_eq, [-1.0, 1.0])

    # 2) Curvature‐corrected eqns (Laplace ΔP)
    def curved_eq(vars, R):
        phim, phip = vars
        ΔP = (d-1)*sigma_eq/R
        return [
            f0(phip) - f0(phim) - ΔP,
            pseudoP(phip, f0(phip)) - pseudoP(phim, f0(phim)) - ΔP
        ]

    # build a grid of R to tabulate φ±(R)
    R_grid = np.linspace(5.0, 100.0, 60)
    phim_grid = np.empty_like(R_grid)
    phip_grid = np.empty_like(R_grid)

    for i, R in enumerate(R_grid):
        sol = fsolve(curved_eq, [phim0, phip0], args=(R,))
        phim_grid[i], phip_grid[i] = sol

    # create interpolators
    phim_i = interp1d(R_grid, phim_grid, kind='cubic',
                      bounds_error=False, fill_value=(phim_grid[0], phim_grid[-1]))
    phip_i = interp1d(R_grid, phip_grid, kind='cubic',
                      bounds_error=False, fill_value=(phip_grid[0], phip_grid[-1]))

    def phi_minus(R): return phim_i(R)
    def phi_plus(R):  return phip_i(R)

    return phi_minus, phi_plus

def make_rhs(phi_minus, phi_plus):
    """Return the RHS(t,y) for the two‐droplet ODEs with these φ±(R)."""
    def f0(phi): return A*phi + phi**3

    def rhs(t, y):
        R1, R2 = y
        dp1 = phi_plus(R1) - phi_minus(R1)
        dp2 = phi_plus(R2) - phi_minus(R2)
        num1 = f0(phi_minus(R2)) - f0(phi_minus(R1))
        num2 = f0(phi_minus(R1)) - f0(phi_minus(R2))
        return [num1/(R1*dp1), num2/(R2*dp2)]
    return rhs

# The two AMB+ cases in Fig 4(b)
cases = [
    {"ζ": -1.0, "λ": +0.5, "color": "C1", "y0": [15.0, 10.0]},
    {"ζ": -4.0, "λ": -1.0, "color": "C0", "y0": [10.0,  7.0]},
]

t_eval = np.logspace(2, 5, 300)  # t from 10^2 → 10^5

plt.figure(figsize=(6,4))
for case in cases:
    ζ, lam, col, y0 = case["ζ"], case["λ"], case["color"], case["y0"]
    # Build coexistence curves and ODE
    phi_m, phi_p = make_coexistence_functions(ζ, lam)
    rhs          = make_rhs(phi_m, phi_p)

    # Integrate
    sol = solve_ivp(rhs, (t_eval[0], t_eval[-1]), y0,
                    t_eval=t_eval, method='RK45',
                    atol=1e-8, rtol=1e-6)

    # Plot: R1 dashed, R2 dotted
    plt.loglog(sol.t, sol.y[0], "--", color=col, lw=1.5)
    plt.loglog(sol.t, sol.y[1], ":",  color=col, lw=1.5,
               label=rf"$\zeta={ζ},\ \lambda={lam}$")

plt.xlabel("Time $t$")
plt.ylabel("Radius $R$")
plt.title("Fig. 4(b) — AMB+ Theory")
plt.legend()
plt.tight_layout()
plt.show()
# plt.savefig("fig4b_AMBplus_theory.png", dpi=150)
