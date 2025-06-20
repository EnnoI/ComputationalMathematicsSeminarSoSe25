import sys
import os
import numpy as np
import time
import pyfftw.interfaces as pyfftw
pyfftw.cache.enable()

def initial_rand_2D(X, Y):
    return 0.1 * np.random.standard_normal(size=(int(np.sqrt(X.size)), int(np.sqrt(Y.size))))

def inital_amb_seperated(X, Y):
    base = -0.9 * np.ones_like(X)
    half = X.shape[0] // 2 + 16
    base[32:32+half, :] = 1.0
    return 0.7 * base + 0.8 * initial_rand_2D(X, Y)

def initial_c_0_2D(X, Y, c_0=0.5, init_noise=0.001):
    return (2. * c_0 - 1.) + init_noise * np.random.standard_normal(size=(int(np.sqrt(X.size)), int(np.sqrt(Y.size))))

def initial_dot_2D(X, Y, r, L, c_0=0.5, S=2):
    radius_mask = X**2 + Y**2 < (r*L)**2
    c_0_outside = (S*r**2 - 1.)/(r**2 - 1.) * c_0
    c_0_inside  = S * c_0
    base = initial_c_0_2D(X, Y, c_0=c_0_outside)
    base[radius_mask] = (2. * c_0_inside - 1.)
    return base

def initial_dot_inner_outer_2D(X, Y, r, L, c_0_outside, c_0_inside):
    radius_mask = X**2 + Y**2 < (r*L)**2
    base = initial_c_0_2D(X, Y, c_0=c_0_outside)
    base[radius_mask] = (2. * c_0_inside - 1.)
    return base

def solve_ambplus_2D(phi_0=None, c_0=0.4, t_state=0.0, t_len=100.0, tau=0.01,
                     eps_val=1., a=-0.25, b=0.25, lam_val=1.75, zeta=2.0,
                     D=0.05, M=1., s_start=-32.*np.pi, s_end=32.*np.pi, s_N=200, return_snapshots=False):

    log_file = "log.csv"
    prev_iter = 0
    if return_snapshots:
        snapshots = []

    # Space discretization
    L = s_end - s_start
    x = np.linspace(s_start, s_end, s_N, endpoint=False)
    y = np.linspace(s_start, s_end, s_N, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")
    kx = np.fft.fftfreq(s_N, d=L/s_N) * 2*np.pi
    ky = np.fft.fftfreq(s_N, d=L/s_N) * 2*np.pi
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    K_2 = KX**2 + KY**2
    K_4 = K_2**2

    # Dealiasing mask
    K_cutoff = 0.67 * K_2.max()
    dealiasing_mask = K_2 < K_cutoff**2

    dx = abs(X[1][0] - X[0][0])
    dy = abs(Y[0][1] - Y[0][0])
    dS = dx * dy

    # Initial condition
    if phi_0 is None:
        phi = inital_amb_seperated(X, Y)
    else:
        phi = phi_0
        if os.path.exists(log_file):
            log_data = np.loadtxt(log_file, delimiter=",", skiprows=1)
            prev_iter = int(log_data[-1, 0])
            t_state = log_data[-1, 1]

    # Write parameters
    with open("parameters.csv", 'w') as f:
        f.write("c_0,t_state,t_len,tau,eps_val,a,b,lam_val,zeta,D,M,s_start,s_end,s_N\n")
        f.write(f"{c_0},{t_state},{t_len},{tau},{eps_val},{a},{b},{lam_val},{zeta},{D},{M},{s_start},{s_end},{s_N}\n")

    f_phi = pyfftw.numpy_fft.fft2(phi, norm="backward", threads=8)

    # Time stepping
    t_N = round(t_len / tau)
    gaussian_scale = np.sqrt(tau / dS)
    check = 20000  # log every 20000 steps

    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            f.write("iteration,time,total_mass,phi_min,phi_max\n")

    start = time.time()
    for ii in range(prev_iter, prev_iter + t_N):

        if ii % check == 0:
            end = time.time()
            print(f"Iterations took time: {end - start:.2f}s")
            out_file = f"phi_{ii}.npy"
            np.save(out_file, phi.real)
            print(f"phi saved to {out_file}")
            if return_snapshots and (ii % check == 0):
                snapshots.append(phi.real.copy())
            with open(log_file, 'a') as f:
                f.write(f"{ii},{t_state},{np.sum(phi.real)*dS/L**2},{np.min(phi.real)},{np.max(phi.real)}\n")
            start = time.time()

        # non‐linear and noise terms
        phi_3 = phi**3
        dphi_dx = pyfftw.numpy_fft.ifft2(1j*KX*f_phi, norm="backward", threads=8)
        dphi_dy = pyfftw.numpy_fft.ifft2(1j*KY*f_phi, norm="backward", threads=8)
        laplacian_phi = pyfftw.numpy_fft.ifft2(-K_2*f_phi, norm="backward", threads=8)

        lap_x_fft = pyfftw.numpy_fft.fft2(laplacian_phi * dphi_dx, norm="backward", threads=8)
        lap_y_fft = pyfftw.numpy_fft.fft2(laplacian_phi * dphi_dy, norm="backward", threads=8)
        grad_phi_2 = dphi_dx**2 + dphi_dy**2

        non_linear_term = (
            -K_2 * pyfftw.numpy_fft.fft2(b*phi_3 + lam_val*grad_phi_2, norm="backward", threads=8)
            - 1j*zeta*(KX*lap_x_fft + KY*lap_y_fft)
        )
        non_linear_term *= dealiasing_mask

        white_x = pyfftw.numpy_fft.fft2(np.random.standard_normal(KX.shape), norm="backward", threads=8)
        white_y = pyfftw.numpy_fft.fft2(np.random.standard_normal(KY.shape), norm="backward", threads=8)
        gaussian_term = -np.sqrt(2*D*M) * 1j * (KX*white_x + KY*white_y)

        f_phi = (f_phi + tau*M*non_linear_term + gaussian_scale*gaussian_term) \
                / (1. + tau*M*(a*K_2 + eps_val*K_4))

        phi = pyfftw.numpy_fft.ifft2(f_phi, norm="backward", threads=8)
        t_state += tau

    # final save
    out_file = f"phi_{ii+1}.npy"
    np.save(out_file, phi.real)
    print(f"phi saved to {out_file}")
    with open(log_file, 'a') as f:
        f.write(f"{ii+1},{t_state},{np.sum(phi.real)*dS/L**2},{np.min(phi.real)},{np.max(phi.real)}\n")
    if return_snapshots:
        return snapshots

def main():
    # Use argparse but ignore unknown flags (so Jupyter/IPython flags are skipped)
    import argparse
    parser = argparse.ArgumentParser(description="Solve the amb+ equation in 2D.")
    parser.add_argument('filename', nargs='?', help='Optional .npy file with initial phi')
    parser.add_argument('--N',       type=int, help='Grid size if no file is given')
    args, _ = parser.parse_known_args()

    if args.filename:
        try:
            phi_0 = np.load(args.filename)
            print(f"Loaded {args.filename} with shape {phi_0.shape}")
            N = phi_0.shape[0]
        except Exception as e:
            print(f"Error reading file: {e}")
            sys.exit(1)
    else:
        phi_0 = None
        N     = args.N or 128

    np.random.seed(0)
   

    # call the solver
    solve_ambplus_2D(
        phi_0=phi_0,
        c_0=0.6,
        s_N=N,
        tau=0.02,
        t_len=10000,
        D=0.0,
        zeta=2.25,
        lam_val=1.8,
        s_start=-64,
        s_end=64
    )

if __name__ == "__main__":
    main()
