import numpy as np
import matplotlib.pyplot as plt
import sys

parameter_data = np.loadtxt("parameters.csv", delimiter=",", skiprows=1)
c_0, t_state, t_len, tau, eps_val, a, b, lam_val, zeta, D, M, s_start, s_end, s_N = parameter_data

def main():
    
    if len(sys.argv) > 1:

        filename = sys.argv[1]

        try:
            phi = np.load(filename)
        except Exception as e:
            print(f"Error reading file: {e}")
            sys.exit(1)

        print(f"Loaded {filename} with shape {phi.shape}")
        N = phi.shape[0]
    else:
        phi = None
        print("Could not load file!")

    
    fig, ax = plt.subplots()
    im = ax.imshow(phi, extent=[s_start, s_end, s_start, s_end], cmap='inferno', vmin=-1.5, vmax=1.5)  # set fixed vmin/vmax if possible
    fig.colorbar(im, ax=ax)
    fig.savefig('thumbnail.png', dpi=200, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()