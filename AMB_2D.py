import sys
import os
import numpy as np
import time

def initial_CH_tanh_2D(X, Y):
    return np.tanh(X**2 + Y**2) + 0.05 * (2 * np.random.rand(int(np.sqrt(X.size)), int(np.sqrt(Y.size))) - 1)

def initial_CH_rand_2D(X, Y):
    return 2. * np.random.rand(int(np.sqrt(X.size)), int(np.sqrt(Y.size))) - 1.

def inital_amb_seperated(X, Y):
    base = np.ones_like(X)
    half = X.shape[0]//2
    base[:half, :] = -1.0
    return 0.9*base + 0.1*initial_CH_rand_2D(X, Y)

def initial_c_0_2D(X, Y, c_0=0.5):
    return (2.*c_0 - 1.) + 0.001*np.random.standard_normal(size=(int(np.sqrt(X.size)), int(np.sqrt(Y.size))))

# We want the overall average to be c_0
# r, c_0 in [0,1], L length of domain, S factor of 
# how large the concentration in the circle is.
def initial_dot_2D(X, Y, r, L, c_0=0.5, S=2):
    radius_mask = X**2 + Y**2 < (r*L)**2
    c_0_outside = (S*r**2 - 1.)/(r**2 - 1.) * c_0
    c_0_inside = S*c_0
    base = initial_c_0_2D(X, Y, c_0=c_0_outside)
    base[radius_mask] = (2. * c_0_inside - 1.)
    return base


def solve_ambplus_2D(phi_0=None, c_0=0.4, t_state=0.0, t_len = 100.0, tau = 0.01, eps_val=1., a=-0.25, b=0.25, lam_val=1.75, zeta=2.0, D=0.05, M=1., s_start = -32.*np.pi, s_end = 32.*np.pi, s_N = 200):
    
    log_file = "log.csv"

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
        #phi = initial_c_0_2D(X, Y, c_0)
        #phi = inital_amb_seperated(X, Y)
        phi = initial_dot_2D(X, Y, 0.2, L, 0.8, S=0.5)
        prev_iter = 0
    else:
        phi = phi_0
        log_data = np.loadtxt(log_file, delimiter=",", skiprows=1)
        prev_iter = int(log_data[-1, 0])
        t_state = log_data[-1, 1]

    # log the parameters used:
    with open("parameters.csv", 'w') as f:
        f.write("c_0,t_state,t_len,tau,eps_val,a,b,lam_val,zeta,D,M,s_start,s_end,s_N\n")
        f.write(f"{c_0},{t_state},{t_len},{tau},{eps_val},{a},{b},{lam_val},{zeta},{D},{M},{s_start},{s_end},{s_N}\n")

    f_phi = np.fft.fft2(phi, norm="backward")

    # Setup time discretization
    t_N = round(t_len/tau) 
    gaussian_scale = np.sqrt(tau)

    # check 10 times during iteration
    # check = int(t_N/10)
    # check every 2000 iterations
    check = 2000

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

        dphi_dx = np.fft.ifft2(1j * KX * f_phi, norm="backward")
        dphi_dy = np.fft.ifft2(1j * KY * f_phi, norm="backward")

        laplacian_phi = np.fft.ifft2(-K_2 * f_phi, norm="backward")
        lapl_phi_prod_grad_phi_x_fft = np.fft.fft2(laplacian_phi * dphi_dx, norm="backward")
        lapl_phi_prod_grad_phi_y_fft = np.fft.fft2(laplacian_phi * dphi_dy, norm="backward")

        grad_phi_2 = dphi_dx**2 + dphi_dy**2

        non_linear_term = - K_2 * np.fft.fft2(b * phi_3 + lam_val * grad_phi_2, norm="backward") - 1j * zeta * (KX * lapl_phi_prod_grad_phi_x_fft + KY * lapl_phi_prod_grad_phi_y_fft)
        
        # Dealiasing the nonlinear term may improve stability
        non_linear_term *= dealiasing_mask

        white_noise_x_fft = np.fft.fft2(np.random.standard_normal(size=KX.shape), norm="backward")
        white_noise_y_fft = np.fft.fft2(np.random.standard_normal(size=KY.shape), norm="backward")

        gaussian_term = - np.sqrt(2*D*M) * 1j * (KX * white_noise_x_fft + KY * white_noise_y_fft)

        f_phi_new = (f_phi + tau * M * non_linear_term + gaussian_scale * gaussian_term) / (1 + tau * M *(a * K_2 + eps_val * K_4))

        # Bookkeeping, setup for next step
        phi = np.fft.ifft2(f_phi_new, norm="backward")
        t_state += tau
        f_phi = f_phi_new

    # print one last time
    out_file = f"phi_{ii+1}.npy"
    np.save(out_file, phi.real)
    print(f"phi saved to {out_file}")
    with open(log_file, 'a') as f:
        f.write(f"{ii+1},{t_state},{np.sum(phi.real)*dS / L**2},{np.min(phi.real)},{np.max(phi.real)}\n")

def solve_CH_2D(phi_0=None, t_state=0.0, t_len = 1000.0, tau = 0.1, eps_val=0.01, lam_val=1., s_start = -32.*np.pi, s_end = 32.*np.pi, s_N = 200):
    
    log_file = "log.csv"

    # Setup space discretization:
    x = np.linspace(s_start, s_end, s_N, endpoint=False)                # real space
    y = np.linspace(s_start, s_end, s_N, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")
    kx = np.fft.fftfreq(s_N, d=(s_end - s_start)/s_N)*2*np.pi           # fourier space
    ky = np.fft.fftfreq(s_N, d=(s_end - s_start)/s_N)*2*np.pi
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    K_2 = KX**2 + KY**2
    K_4 = K_2**2
    dx = X[0][1] - X[0][0]
    dy = Y[1][0] - Y[0][0]

    # Setup the phis for our time step with initial condition
    if phi_0 is None:
        phi = initial_c_0_2D(X, Y, 0.4)
    else:
        phi = phi_0
        log_data = np.loadtxt(log_file, delimiter=",", skiprows=1)
        prev_iter = int(log_data[-1, 0])
        t_state = log_data[-1, 1]

    f_phi = np.fft.fft2(phi)

    # Setup time discretization
    t_N = round(t_len/tau) 

    # checks
    check = int(t_N/10)

    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            f.write("iteration,time,total_energy,total_mass,phi_min,phi_max\n")    

    # Time stepping loop
    for ii in range(prev_iter, prev_iter+t_N):

        if ii % check == 0:
            # Save the result
            out_file = f"phi_{ii}.npy"
            np.save(out_file, phi.real)
            print(f"phi saved to {out_file}")

            dphi_dx = np.fft.ifft2(1j * KX * f_phi).real
            dphi_dy = np.fft.ifft2(1j * KY * f_phi).real
            grad_phi_2 = dphi_dx**2 + dphi_dy**2
            E = np.sum(eps_val/2. * grad_phi_2 + lam_val/4. * (phi.real**2 - 1.)**2) * dx*dy

            with open(log_file, 'a') as f:
                    f.write(f"{ii},{t_state},{E},{np.sum(phi.real)*dx*dy},{np.min(phi.real)},{np.max(phi.real)}\n")

        phi_3 = phi**3

        phi_3_fft = np.fft.fft2(phi_3)

        f_phi_new = (f_phi - tau * lam_val * K_2 * phi_3_fft) / (1 + tau * K_2 *(eps_val * K_2 - lam_val))

        phi_new = np.fft.ifft2(f_phi_new)

        t_state += tau

        # logging and bookkeeping:
        phi = phi_new
        f_phi = f_phi_new

    # print one last time
    out_file = f"phi_{ii+1}.npy"
    np.save(out_file, phi.real)
    print(f"phi saved to {out_file}")

    dphi_dx = np.fft.ifft2(1j * KX * f_phi).real
    dphi_dy = np.fft.ifft2(1j * KY * f_phi).real
    grad_phi_2 = dphi_dx**2 + dphi_dy**2
    E = np.sum(eps_val/2. * grad_phi_2 + lam_val/4. * (phi.real**2 - 1.)**2) * dx*dy

    with open(log_file, 'a') as f:
            f.write(f"{ii+1},{t_state},{E},{np.sum(phi.real)*dx*dy},{np.min(phi.real)},{np.max(phi.real)}\n")
    

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
        N = 101

    np.random.seed(0)

    # Solve an equation
    solve_ambplus_2D(phi_0, c_0=0.8, s_N=N, tau=0.02, t_len=800, D=0.05, zeta=1.5, lam_val=2.0, s_start=-16*np.pi, s_end=16*np.pi)

if __name__ == "__main__":
    main()