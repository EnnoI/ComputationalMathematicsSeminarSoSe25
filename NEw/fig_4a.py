# fig4a_physical.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import solve_bvp

# --- 1) Thermodynamics (App B) ---
def free_energy(phi, a, b):
    return 0.5*a*phi**2 + 0.25*b*phi**4

def chem_potential(phi, a, b):
    return a*phi + b*phi**3

# Binodals φ₁, φ₂ (coexistence at flat interface)
def binodal_solver(a, b, K):
    def eqs(vars):
        p1, p2 = vars
        return [
            chem_potential(p1,a,b) - chem_potential(p2,a,b),
            (free_energy(p1,a,b)-p1*chem_potential(p1,a,b))
          - (free_energy(p2,a,b)-p2*chem_potential(p2,a,b))
        ]
    return fsolve(eqs, [-1.0, +1.0])

# --- 2) 1D droplet profile solver (App C) ---
def droplet_profile_solver(R, phi_out, a, b, K, lam, zeta, d=3):
    # Solve μ(φ)-K(∇²φ)+λs+ζz = μ_flat with BC φ'(0)=0, φ(∞)=φ_out
    r_max=5*R; Nr=200
    r = np.linspace(0, r_max, Nr)
    dr = r[1]-r[0]
    # initial guess: tanh
    phi0 = phi_out + ( -phi_out ) * np.tanh((r-R)/np.sqrt(2*K/(-a)))
    def ode(r, y):
        φ, φr = y
        μ_flat = chem_potential(phi_out,a,b)
        # leading terms
        lap = (chem_potential(φ,a,b)-μ_flat)/K
        φrr = lap - (d-1)/np.maximum(r,1e-8)*φr
        return np.vstack((φr, φrr))
    def bc(ya,yb):
        return np.array([ ya[1], yb[0]-phi_out ])
    sol = solve_bvp(ode, bc, r, np.vstack((phi0, np.gradient(phi0,r))), max_nodes=10000)
    φR = sol.sol(r)[0]
    return r, φR

# --- Precompute φ₋(R) and Δφ(R) over a radius grid ---
def build_radius_lookup(Rs, phi_out, a,b,K, lam, zeta):
    phi_minus = []
    delta_phi = []
    for R in Rs:
        r, φR = droplet_profile_solver(R, phi_out, a,b,K, lam, zeta)
        phi_minus.append(φR[-1])
        delta_phi.append(φR[0]-φR[-1])
    return np.array(phi_minus), np.array(delta_phi)

# --- 3) Two-droplet ODE derivative at t=0 (App E) ---
def dR1_dt(R1,R2, phi_minus_interp, delta_phi_interp, a,b):
    φm1 = phi_minus_interp(R1)
    φm2 = phi_minus_interp(R2)
    Δφ1 = delta_phi_interp(R1)
    return (chem_potential(φm2,a,b)-chem_potential(φm1,a,b)) / (R1 * Δφ1)

# --- 4) Classification loop over (ζ,λ) ---
a,b,K = -0.25, 0.25, 1.0
phi1, phi2 = binodal_solver(a,b,K)

# radius grid for lookup
Rs = np.linspace(5,25,20)

ζ_vals = np.linspace(-4,4,41)
λ_vals = np.linspace(-2,2,41)
regions = np.empty((len(ζ_vals),len(λ_vals)),dtype='<U1')

for i, ζ in enumerate(ζ_vals):
    for j, lam in enumerate(λ_vals):
        # build lookup for these parameters
        phi_minus, delta_phi = build_radius_lookup(Rs, phi1, a,b,K, lam, ζ)
        phi_minus_i = lambda R: np.interp(R, Rs, phi_minus)
        delta_phi_i = lambda R: np.interp(R, Rs, delta_phi)
        # evaluate derivative at initial R1=10, R2=20
        dR1 = dR1_dt(10.0, 20.0, phi_minus_i, delta_phi_i, a,b)
        if dR1<0:
            regions[i,j] = 'A'  # forward
        else:
            regions[i,j] = 'C' if lam>0 else 'B'

# Extract points
Z,J = np.meshgrid(ζ_vals, λ_vals, indexing='xy')
ZA = Z[regions=='A']; LA = J[regions=='A']
ZB = Z[regions=='B']; LB = J[regions=='B']
ZC = Z[regions=='C']; LC = J[regions=='C']

# exact paper points
paper = {
  'A':((-1,0.5),'rs'),
  'B':((-4,-1),'bo'),
  'C':((3,1),'ms'),
}

# --- 5) Plot and save ---
fig, ax = plt.subplots(figsize=(6,5))
ax.scatter(ZA,LA, s=20, color='#A6CEE3', label='A: forward')
ax.scatter(ZB,LB, s=20, color='#B2DF8A', label='B: rev clusters')
ax.scatter(ZC,LC, s=20, color='#FB9A99', label='C: rev bubbles')

# boundaries
lam_line = np.linspace(-2,2,100)
ax.plot(2*lam_line, lam_line, 'k-', lw=2)

# paper markers
for key,(pt,style) in paper.items():
    ax.plot(pt[0],pt[1], style, ms=10, mec='k')

ax.set_xlim(-4,4); ax.set_ylim(-2,2)
ax.set_xlabel(r'$\zeta$'); ax.set_ylabel(r'$\lambda$')
ax.set_title('Fig 4(a) via full Appendix E ODE classification')
ax.legend(loc='lower right', fontsize=10)
ax.set_aspect('equal','box')
plt.tight_layout()
plt.savefig('fig4a_physical.png', dpi=300)
plt.show()
