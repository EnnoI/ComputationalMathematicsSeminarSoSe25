# fig4a_reproduction.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.transforms import Bbox

# --- 1) Define the domain and boundary curves ---

λ = np.linspace(-4, 4, 400)

# Boundary between A and C: pseudotension for bubbles (ζ = 2λ)
ζ_AC = 2*λ

# Boundary between A and B: pseudotension for clusters.
# In the paper this is computed numerically; here we approximate
# with a smooth curve passing through (-2,0) and (0,-1.5)
ζ_AB = -0.4*(λ+2)**2 - 0.1  # sketch -- adjust to taste

# Example points from two-droplet sims (red squares, blue circles)
# (ζ,λ) = (–1, +0.5) → A (forward), (–4,–1) → B (reverse clusters),
# (+3,+1) → C (reverse bubbles)
pts_forward = [(-1, 0.5)]
pts_cluster  = [(-4, -1)]
pts_bubble   = [(3, 1)]

# --- 2) Plot Setup ---
fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim(-4,4)
ax.set_ylim(-2,2)
ax.set_xlabel(r'$\zeta$', fontsize=14)
ax.set_ylabel(r'$\lambda$', fontsize=14)
ax.set_title('Mean-field phase diagram of AMB⁺ (Fig 4a)', fontsize=16)

# Region shading
ax.fill_between(λ, ζ_AB,  2.5, where=ζ_AB<=2*λ,    color='#A6CEE3', alpha=0.3, label='Region A')
ax.fill_between(λ, -2.5,   ζ_AB, where=ζ_AB> (2*λ), color='#B2DF8A', alpha=0.3, label='Region B')
ax.fill_between(λ, 2*λ,    2.5, where=(2*λ)>=ζ_AB,  color='#FB9A99', alpha=0.3, label='Region C')

# Boundary lines
ax.plot(ζ_AC,   λ,   'k-', lw=2, label=r'$σ_0=0$ (bubbles)' )
ax.plot(ζ_AB,   λ,   'k--', lw=2, label=r'$σ_0=0$ (clusters)')

# Example points
ax.scatter(*zip(*pts_forward), marker='s', color='r',  s=80, label='Forward Ostwald')
ax.scatter(*zip(*pts_cluster),  marker='o', color='b',  s=80, label='Reverse (clusters)')
ax.scatter(*zip(*pts_bubble),   marker='s', color='m',  s=80, label='Reverse (bubbles)')

# Labels A, B, C
ax.text(-0.5,  0.2, 'A\nforward\n(σ₀>0)', ha='center', va='center', fontsize=12)
ax.text(-3.0, -0.8, 'B\nreverse clusters\n(σ₀<0)', ha='center', va='center', fontsize=12)
ax.text( 2.5,  1.2, 'C\nreverse bubbles\n(σ₀<0)', ha='center', va='center', fontsize=12)

# Inset boxes (as placeholders; replace with your own images if desired)
def add_inset(ax, box, color):
    rect = Rectangle((box[0], box[1]), box[2], box[3],
                     linewidth=1, edgecolor='k', facecolor='none')
    ax.add_patch(rect)
    # tiny circle to hint at an inset
    cx = box[0] + box[2]*0.5
    cy = box[1] + box[3]*0.5
    ax.add_patch(Circle((cx,cy), 0.2, color=color, alpha=0.6))

# approximate positions and sizes
add_inset(ax, (-3.5, 0.5, 1.0, 0.6), 'blue')   # A inset
add_inset(ax, (-4.0,-1.7,1.5,0.8), 'green')   # B inset
add_inset(ax, ( 2.5, 0.6,1.2,0.7), 'purple')  # C inset

ax.legend(loc='lower right', fontsize=10)
ax.grid(False)
plt.tight_layout()

# --- 3) Save to disk ---
plt.savefig('fig4a.png', dpi=300)
plt.show()
