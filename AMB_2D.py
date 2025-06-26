import sys
import os
import numpy as np
import time
import pyfftw.interfaces as pyfftw
pyfftw.cache.enable()



def initial_rand_2D(X, Y):
    return 0.1*np.random.standard_normal(size=(int(np.sqrt(X.size)), int(np.sqrt(Y.size))))

def inital_amb_seperated(X, Y):
    base = -0.9*np.ones_like(X)
    half = X.shape[0]//2 + 16
    base[32:32+half, :] = 1.0
    return 0.7*base + 0.8*initial_rand_2D(X, Y)

def initial_c_0_2D(X, Y, c_0=0.5, init_noise=0.001):
    return (2.*c_0 - 1.) + init_noise*np.random.standard_normal(size=(int(np.sqrt(X.size)), int(np.sqrt(Y.size))))

# We want the overall average to be c_0
# r, c_0 in [0,1], L length of domain, S factor of 
# how large the concentration in the circle is.
# TODO: this does not compute an overall average of c_0 (yet?)
def initial_dot_2D(X, Y, r, L, c_0=0.5, S=2):
    radius_mask = X**2 + Y**2 < (r*L)**2
    c_0_outside = (S*r**2 - 1.)/(r**2 - 1.) * c_0
    c_0_inside = S*c_0
    base = initial_c_0_2D(X, Y, c_0=c_0_outside)
    base[radius_mask] = (2. * c_0_inside - 1.)
    return base

def initial_dot_inner_outer_2D(X, Y, r, L, c_0_outside, c_0_inside):
    radius_mask = X**2 + Y**2 < (r*L)**2
    base = initial_c_0_2D(X, Y, c_0=c_0_outside)
    base[radius_mask] = (2. * c_0_inside - 1.)
    return base

def initial_two_dots(X, Y, R1, R2, x1, x2, L):
    base = initial_c_0_2D(X, Y, c_0=1.)
    radius_mask1 = (X - x1[0])**2 + (Y - x1[1])**2 < (R1*L)**2
    base[radius_mask1] = -1.
    radius_mask2 = (X - x2[0])**2 + (Y - x2[1])**2 < (R2*L)**2
    base[radius_mask2] = -1.
    return base


def solve_ambplus_2D(phi_0=None, c_0=0.4, t_state=0.0, t_len = 100.0, tau = 0.01, eps_val=1., a=-0.25, b=0.25, lam_val=01.75 , zeta= 3.0 , D=0.05, M=1., s_start = -32.*np.pi, s_end = 32.*np.pi, s_N = 200):
    
    log_file = "log.csv"
    prev_iter = 0

    # Setup space discretization:
    L = s_end - s_start
    x = np.linspace(s_start, s_end, s_N, endpoint=False)                # real space
    y = np.linspace(s_start, s_end, s_N, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")
    kx = np.fft.fftfreq(s_N, d=L/s_N)*2*np.pi  	                        # fourier space
    ky = np.fft.fftfreq(s_N, d=L/s_N)*2*np.pi
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    K_2 = KX**2 + KY**2
    K_4 = K_2**2

    # Create a dealisiasing mask
    K_cutoff = 0.67 * K_2.max()
    dealiasing_mask = K_2 < K_cutoff**2

    #KXKY = KX * KY
    dx = abs(X[1][0] - X[0][0])
    dy = abs(Y[0][1] - Y[0][0])
    dS = dx*dy
    
    # Setup the phis for our time step with initial condition
    if phi_0 is None:
        # phi = initial_c_0_2D(X, Y, c_0, 0.0)
        # phi = inital_amb_seperated(X, Y)
        # phi = initial_dot_inner_outer_2D(X, Y, r=0.25, L=L, c_0_outside=0.8, c_0_inside=0.2)
        phi = initial_two_dots(X, Y, 0.3, 0.05, (0.0, -22.0), (0.0, 28.0), L)
    else:
        phi = phi_0
        if os.path.exists(log_file):
            log_data = np.loadtxt(log_file, delimiter=",", skiprows=1)
            prev_iter = int(log_data[-1, 0])
            t_state = log_data[-1, 1]

    # log the parameters used:
    with open("parameters.csv", 'w') as f:
        f.write("c_0,t_state,t_len,tau,eps_val,a,b,lam_val,zeta,D,M,s_start,s_end,s_N\n")
        f.write(f"{c_0},{t_state},{t_len},{tau},{eps_val},{a},{b},{lam_val},{zeta},{D},{M},{s_start},{s_end},{s_N}\n")

    f_phi = pyfftw.numpy_fft.fft2(phi, threads=8)

    # Setup time discretization
    t_N = round(t_len/tau) 
    gaussian_scale = np.sqrt(tau / dS)

    # check 10 times during iteration
    # check = int(t_N/10)
    # check every 10000 iterations
    check = 10000

    # Setup the logging
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            f.write("iteration,time,total_mass,phi_min,phi_max\n")

    start = time.time()
    # Time stepping loop
    for ii in range(prev_iter, prev_iter+t_N):
        
        
        if ii % check == 0:
            end = time.time()
            print(f"Iterations took time: {end - start}")

            # Save the result
            out_file = f"phi_{ii}.npy"
            np.save(out_file, phi.real)
            print(f"phi saved to {out_file}")

            with open(log_file, 'a') as f:
                    f.write(f"{ii},{t_state},{np.sum(phi.real)*dS / L**2},{np.min(phi.real)},{np.max(phi.real)}\n")

            start = time.time()

        phi_3 = phi**3

        dphi_dx = pyfftw.numpy_fft.ifft2(1j * KX * f_phi, threads=8)
        dphi_dy = pyfftw.numpy_fft.ifft2(1j * KY * f_phi, threads=8)

        laplacian_phi = pyfftw.numpy_fft.ifft2(-K_2 * f_phi, threads=8)
        lapl_phi_prod_grad_phi_x_fft = pyfftw.numpy_fft.fft2(laplacian_phi * dphi_dx, threads=8)
        lapl_phi_prod_grad_phi_y_fft = pyfftw.numpy_fft.fft2(laplacian_phi * dphi_dy, threads=8)

        grad_phi_2 = dphi_dx**2 + dphi_dy**2

        non_linear_term = - K_2 * pyfftw.numpy_fft.fft2(b * phi_3 + lam_val * grad_phi_2, threads=8) - 1j * zeta * (KX * lapl_phi_prod_grad_phi_x_fft + KY * lapl_phi_prod_grad_phi_y_fft)
        
        # Dealiasing the nonlinear term may improve stability
        # non_linear_term *= dealiasing_mask

        white_noise_x_fft = pyfftw.numpy_fft.fft2(np.random.standard_normal(size=KX.shape), threads=8)
        white_noise_y_fft = pyfftw.numpy_fft.fft2(np.random.standard_normal(size=KY.shape), threads=8)

        gaussian_term = - np.sqrt(2*D*M) * 1j * (KX * white_noise_x_fft + KY * white_noise_y_fft)

        f_phi = (f_phi + tau * M * non_linear_term + gaussian_scale * gaussian_term) / (1. + tau * M *(a * K_2 + eps_val * K_4))

        # Bookkeeping, setup for next step
        phi = pyfftw.numpy_fft.ifft2(f_phi, threads=8)
        t_state += tau

    # print one last time
    out_file = f"phi_{ii+1}.npy"
    np.save(out_file, phi.real)
    print(f"phi saved to {out_file}")
    with open(log_file, 'a') as f:
        f.write(f"{ii+1},{t_state},{np.sum(phi.real)*dS / L**2},{np.min(phi.real)},{np.max(phi.real)}\n")

def main():
    
    if len(sys.argv) > 1:

        filename = sys.argv[1]

        try:
            phi_0 = np.load(filename)
        except Exception as e:
            print(f"Error reading file: {e}")
            sys.exit(1)

        print(f"Loaded {filename} with shape {phi_0.shape}")
        N = phi_0.shape[0]
    else:
        phi_0 = None
        N = 128

    #np.random.seed(0)

    # Solve an equation
    solve_ambplus_2D(phi_0, c_0=0.6, s_N=N, tau=0.02, t_len=15000, D=0.0, lam_val= 1.75 , zeta= 3.0 , s_start=-64, s_end=64)

if __name__ == "__main__":
    main()