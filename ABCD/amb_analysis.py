# amb_analysis.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, sum as ndi_sum
from AMB_plus import solve_ambplus_2D  # see note below

# ---------------------------------------------------
# 1) Two-droplet initial condition generator
# ---------------------------------------------------
def make_two_droplet_ic(s_N, s_start, s_end,
                        r_frac=0.2, sep_frac=0.5,
                        phi_inside=1.0, phi_outside=-0.9):
    """
    Build a phi field with two circular droplets.
    r_frac: radius as fraction of domain length
    sep_frac: separation of centers as fraction of domain length
    """
    x = np.linspace(s_start, s_end, s_N, endpoint=False)
    y = np.linspace(s_start, s_end, s_N, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")
    L = s_end - s_start

    phi0 = phi_outside * np.ones((s_N, s_N))
    r = r_frac * L
    sep = sep_frac * L

    mask1 = (X + sep/2)**2 + Y**2 < r**2
    mask2 = (X - sep/2)**2 + Y**2 < r**2
    phi0[mask1] = phi_inside
    phi0[mask2] = phi_inside

    return phi0

# ---------------------------------------------------
# 2) Function to extract the two droplet radii
# ---------------------------------------------------
def extract_two_radii(phi, threshold=0.0):
    """
    Threshold phi to binary, label connected regions,
    compute area and convert to radius. Returns sorted radii.
    """
    binary = (phi > threshold)
    labeled, num = label(binary)
    if num < 2:
        # either droplets merged or disappeared
        return None
    areas = ndi_sum(binary, labeled, index=np.arange(1, num+1))
    radii = np.sqrt(areas / np.pi)
    # Sort and return the two largest (in case extras appear)
    return np.sort(radii)[-2:]

# ---------------------------------------------------
# 3) Reproduce Fig. 4(b): radii vs time for one (ζ,λ)
# ---------------------------------------------------
def make_fig4b():
    # Parameters (choose region A and region B examples)
    params = dict(
        s_N=128, s_start=-64, s_end=64,
        tau=0.02, t_len=5000,
        eps_val=1.0, a=-0.25, b=0.25,
        D=0.0, M=1.0,
    )
    # Example 1: Region A (forward)
    lamA, zetA = 0.5, -1.0
    # Example 2: Region B (reverse)
    lamB, zetB = -1.0, -4.0

    for label_case, (lam, zet) in [("A", (lamA, zetA)), ("B", (lamB, zetB))]:
        # Build IC
        phi0 = make_two_droplet_ic(params["s_N"],
                                   params["s_start"],
                                   params["s_end"])
        # Run solver; NEED solve_ambplus_2D to accept return_snapshots=True
        snaps = solve_ambplus_2D(
            phi_0        = phi0,
            lam_val      = lam,
            zeta         = zet,
            return_snapshots = True,
            **params
        )

        # Extract radii over time
        times = np.linspace(0, params["t_len"], len(snaps))
        radii = []
        for phi in snaps:
            r = extract_two_radii(phi)
            if r is None:
                radii.append([np.nan, np.nan])
            else:
                radii.append(r)
        radii = np.array(radii)  # shape (ntimesteps, 2)

        # Plot
        plt.figure()
        plt.plot(times, radii[:,0], label="Small droplet")
        plt.plot(times, radii[:,1], label="Large droplet")
        plt.xlabel("Time")
        plt.ylabel("Radius")
        plt.title(f"Fig4(b) Region {label_case}: λ={lam}, ζ={zet}")
        plt.legend()
        plt.savefig(f"fig4b_region_{label_case}.png", dpi=300)
        plt.close()

# ---------------------------------------------------
# 4) Reproduce Fig. 4(a): phase diagram in (ζ,λ)
# ---------------------------------------------------
def make_fig4a():
    # Grid of parameters
    zeta_vals   = np.linspace(-5, 5, 21)
    lambda_vals = np.linspace(-5, 5, 21)

    # Fixed simulation params
    sim_params = dict(
        s_N=64, s_start=-32, s_end=32,
        tau=0.05, t_len=2000,
        eps_val=1.0, a=-0.25, b=0.25,
        D=0.0, M=1.0,
    )

    phase = np.zeros((len(zeta_vals), len(lambda_vals)), dtype=int)
    # 0 = forward, 1 = reverse

    for i, zet in enumerate(zeta_vals):
        for j, lam in enumerate(lambda_vals):
            # IC
            phi0 = make_two_droplet_ic(sim_params["s_N"],
                                       sim_params["s_start"],
                                       sim_params["s_end"])
            snaps = solve_ambplus_2D(
                phi_0        = phi0,
                lam_val      = lam,
                zeta         = zet,
                return_snapshots = True,
                **sim_params
            )
            # Compare small droplet radius initial vs final
            r0  = extract_two_radii(snaps[0])
            rF  = extract_two_radii(snaps[-1])
            if r0 is None or rF is None:
                phase[i,j] = 0
            else:
                # If Δr_small > 0, reverse Ostwald
                phase[i,j] = int((rF[0] - r0[0]) > 0)

    # Plot
    plt.figure(figsize=(6,5))
    plt.pcolormesh(lambda_vals, zeta_vals, phase, cmap="Accent", shading="auto")
    plt.xlabel("λ")
    plt.ylabel("ζ")
    plt.title("Fig4(a): 0=forward, 1=reverse Ostwald")
    plt.colorbar(label="Ostwald\n(1=reverse)")
    plt.savefig("fig4a_phase_diagram.png", dpi=300)
    plt.close()

# ---------------------------------------------------
# 5) Main entry
# ---------------------------------------------------
if __name__ == "__main__":
    make_fig4b()
    make_fig4a()
    print("Done! Figures saved to disk.")
