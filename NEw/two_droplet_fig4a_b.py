# ambplus_phase_diagram_and_two_droplet.py
# Reproduce Fig 4(a) and Fig 4(b) of Tjhung–Nardini–Cates via analytical pseudotension

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
# ----------------------------
# 1) Mean‐field phase diagram (Fig 4a)
#   pseudotension σ₀ = ζ − 2λ
# ----------------------------
def plot_phase_diagram():
    lam = np.linspace(-4, 4, 400)
    zeta = np.linspace(-4, 4, 400)
    L, Z = np.meshgrid(lam, zeta)
    sigma0 = Z - 2*L

    fig, ax = plt.subplots(figsize=(5,5))
    # filled regions
    ax.contourf(L, Z, sigma0, levels=[-1e3, 0, 1e3],
                colors=['#A6CEE3','#FB9A99'], alpha=0.6)
    # zero contour
    ax.contour(L, Z, sigma0, levels=[0], colors='k', linewidths=2)

    # annotations
    ax.text(-3.5,  2.5, "reverse\n(σ₀>0)", color='#E31A1C', fontsize=12)
    ax.text( 2.5, -3.5, "forward\n(σ₀<0)", color='#1F78B4', fontsize=12,
            ha='right')

    ax.set_xlabel('λ', fontsize=14)
    ax.set_ylabel('ζ', fontsize=14)
    ax.set_title('Mean‐field phase diagram (Fig 4a)\nσ₀ = ζ − 2λ', fontsize=16)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)

    # horizontal colorbar
    cmap = plt.get_cmap("coolwarm")
    norm = plt.Normalize(vmin=sigma0.min(), vmax=sigma0.max())
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(mappable, ax=ax,
                        orientation='horizontal',
                        pad=0.1, fraction=0.05)
    cbar.set_label('σ₀', fontsize=12)
    cbar.set_ticks([-8, -4, 0, 4, 8])

    plt.tight_layout()
    plt.show()



# ----------------------------
# 2) Two‐droplet evolution (Fig 4b)
#    Effective‐medium ODE:  ̇R = β/R * (1 - Rₛ/R)
# ----------------------------
def simulate_two_droplets(zeta, lam, R1_0=10, R2_0=20, Rs=15, t_span=(1,1e5)):
    """
    zeta, lam: parameters
    R1_0, R2_0: initial radii
    Rs: stationary radius estimate
    t_span: (t_min, t_max)
    """
    # pseudotension
    sigma0 = zeta - 2*lam
    # choose scales Δψ and Δφ so that β = sigma0 * Δψ / Δφ
    # for φ binodals at ±1, Δφ=2; pick Δψ=1 for simplicity
    beta = sigma0 * 1.0 / 2.0

    def rhs(t, R):
        R1, R2 = R
        dR1 = beta / R1 * (1.0 - Rs/R1)
        dR2 = beta / R2 * (1.0 - Rs/R2)
        return [dR1, dR2]

    sol = solve_ivp(rhs, t_span, [R1_0, R2_0],
                    dense_output=True, max_step=(t_span[1]-t_span[0])/1000)
    return sol

def plot_two_droplets():
    plt.figure(figsize=(6,4))

    # Forward Ostwald example
    sol_fwd = simulate_two_droplets(zeta=-1.0, lam=0.5,
                                     R1_0=15, R2_0=10, Rs=12, t_span=(1,1e4))
    plt.loglog(sol_fwd.t, sol_fwd.y[0], 'r-',  label='R₁ forward')
    plt.loglog(sol_fwd.t, sol_fwd.y[1], 'r--', label='R₂ forward')

    # Reverse Ostwald example
    sol_rev = simulate_two_droplets(zeta=3.0, lam=1.0,
                                     R1_0=15, R2_0=10, Rs=12, t_span=(1,1e4))
    plt.loglog(sol_rev.t, sol_rev.y[0], 'b-',  label='R₁ reverse')
    plt.loglog(sol_rev.t, sol_rev.y[1], 'b--', label='R₂ reverse')

    plt.xlabel('t', fontsize=12)
    plt.ylabel('R', fontsize=12)
    plt.title('Two‐droplet evolution (Fig 4b)', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ----------------------------
# Main
# ----------------------------
if __name__ == '__main__':
    plot_phase_diagram()
    plot_two_droplets()
