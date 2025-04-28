import numpy as np
np.random.seed(2025)
import matplotlib.pyplot as plt

def swift_hohenberg_2d_etdrk2(
    Nx=128,
    Ny=128,
    Lx=50.0,
    Ly=50.0,
    gamma=0.9,
    dt=0.005,
    nsteps=2000,
    noise_amp=0.01,
    dealias=True,
    t_snap=0.0
):
    """
    Solve the 2D Swift-Hohenberg equation using a pseudo-spectral (Fourier) method
    and an Exponential Time-Differencing Runge-Kutta 2 (ETDRK2) scheme.

    The Swift-Hohenberg equation is:
    $$\\frac{\\partial \\psi}{\\partial t} = \\gamma \\psi - (1 + \\nabla^2)^2 \\psi - \\psi^3.$$

    This solver implements:
      - Periodic boundary conditions in x and y.
      - 2D FFTs for spatial derivatives.
      - Dealiasing by the 2/3 rule for the cubic term.
      - An optional snapshot at time t_snap.

    Args:
        Nx (int, optional): Number of grid points in x. Defaults to 128.
        Ny (int, optional): Number of grid points in y. Defaults to 128.
        Lx (float, optional): Physical domain length in x. Defaults to 50.0.
        Ly (float, optional): Physical domain length in y. Defaults to 50.0.
        gamma (float, optional): Parameter gamma in the Swift-Hohenberg PDE. Defaults to 0.9.
        dt (float, optional): Time step for the ETDRK2 scheme. Defaults to 0.005.
        nsteps (int, optional): Total number of time steps to evolve. Defaults to 2000.
        noise_amp (float, optional): Amplitude of initial random noise. Defaults to 0.01.
        dealias (bool, optional): If True, applies the 2/3 dealiasing rule. Defaults to True.
        t_snap (float, optional): Time at which to store psi in addition to the final time.
            If t_snap <= 0 or t_snap is beyond the total simulated time, returns None for that snapshot.

    Returns:
        tuple:
            - psi_final (ndarray): Real-space field of psi at the final time (shape Nx x Ny).
            - psi_hist (list of ndarray): List of intermediate psi fields, sampled every ~nsteps/10.
            - psi_at_t_snap (ndarray or None): Real-space field at t ~= t_snap if valid, otherwise None.
    """

    # --- 1) Spatial grid and wave vectors ---
    dx, dy = Lx / Nx, Ly / Ny
    x = np.linspace(0, Lx, Nx, endpoint=False)
    y = np.linspace(0, Ly, Ny, endpoint=False)

    # Frequencies for FFT (assuming periodic boundary conditions)
    kx = 2.0 * np.pi * np.fft.fftfreq(Nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(Ny, d=dy)
    Kx, Ky = np.meshgrid(kx, ky, indexing='ij')
    K2 = Kx**2 + Ky**2

    # --- 2) Define the linear operator L in Fourier space ---
    # L = gamma - (1 - K^2)^2
    L_op = gamma - (1.0 - K2)**2

    # --- 3) Exponential time-differencing coefficients (ETDRK2) ---
    eLdt_half = np.exp(L_op * dt * 0.5)   # e^{L dt/2}
    eLdt_full = np.exp(L_op * dt)         # e^{L dt}

    def safe_div_exp(eL_val, L_val, dt_factor):
        """
        Compute (e^{L_val} - 1) / L_val in a safe manner,
        avoiding division by zero when L_val is near 0.

        Args:
            eL_val (ndarray): Precomputed exponential term, e^{L_val}.
            L_val (ndarray): The array of operator values L.
            dt_factor (float): The dt-related factor for the limiting case.

        Returns:
            ndarray: The array of (e^{L_val} - 1)/L_val with safe handling at L_val ~ 0.
        """
        out = np.zeros_like(L_val)
        small_mask = np.abs(L_val) < 1e-14
        out[~small_mask] = (eL_val[~small_mask] - 1.0) / L_val[~small_mask]
        # For near-zero values of L, use the first-order Taylor expansion
        out[small_mask] = dt_factor
        return out

    # Compute the ETDRK2 coefficients C_half, C_full
    C_half = safe_div_exp(eLdt_half, L_op, dt * 0.5)
    C_full = safe_div_exp(eLdt_full,  L_op, dt)

    # --- 4) Dealiasing setup (2/3 rule) ---
    def dealias_2d(arr):
        """
        Apply the 2/3 rule to zero out high-frequency modes in x and y.

        Args:
            arr (ndarray): 2D Fourier-space array (Nx x Ny).

        Returns:
            ndarray: The same array, with out-of-bounds frequencies set to 0.
        """
        cutoff_x = Nx // 3
        cutoff_y = Ny // 3
        arr[cutoff_x:-cutoff_x, :] = 0.0
        arr[:, cutoff_y:-cutoff_y] = 0.0
        return arr

    # --- 5) Nonlinear term function ---
    def nonlinear_term(psi_hat):
        """
        Compute the Fourier transform of - (psi^3).

        Args:
            psi_hat (ndarray): The Fourier-space representation of psi.

        Returns:
            ndarray: The Fourier-space representation of -(psi^3).
        """
        psi_real = np.fft.ifft2(psi_hat).real
        return -np.fft.fft2(psi_real**3)

    # --- 6) Initial condition (random noise) ---
    psi0 = noise_amp * (2.0 * np.random.rand(Nx, Ny) - 1.0)
    psi_hat = np.fft.fft2(psi0)

    # Prepare lists for storing snapshots
    psi_hist = []
    sample_interval = max(nsteps // 10, 1)

    # Prepare storage for optional snapshot at time t_snap
    psi_at_t_snap = None
    if t_snap > 0.0:
        step_snap = int(round(t_snap / dt))

    # --- 7) Time-stepping loop ---
    for step in range(nsteps):
        # (a) Nonlinear term at the current step
        F_n = nonlinear_term(psi_hat)
        if dealias:
            F_n = dealias_2d(F_n)

        # (b) Half-step in Fourier space
        psi_hat_half = eLdt_half * psi_hat + C_half * F_n

        # (c) Nonlinear term at the half-step
        F_half = nonlinear_term(psi_hat_half)
        if dealias:
            F_half = dealias_2d(F_half)

        # (d) Full-step update
        psi_hat_new = eLdt_full * psi_hat + C_full * F_half

        # Check for divergence or NaNs
        if not np.all(np.isfinite(psi_hat_new)):
            print(f"Solution diverged at step {step}. Stopping.")
            break

        psi_hat = psi_hat_new

        # Record the field at t_snap if within range
        if t_snap > 0.0 and step == step_snap and step_snap < nsteps:
            psi_at_t_snap = np.fft.ifft2(psi_hat).real.copy()

        # Store intermediate snapshots for visualization
        if step % sample_interval == 0 or step == nsteps - 1:
            psi_real = np.fft.ifft2(psi_hat).real
            psi_hist.append(psi_real.copy())

    # --- 8) Final real-space field ---
    psi_final = np.fft.ifft2(psi_hat).real

    return psi_final, psi_hist, psi_at_t_snap


if __name__ == "__main__":
    # Demonstration: request a snapshot at a large time, e.g. 9999.0
    psi_final, psi_hist, psi_at_3 = swift_hohenberg_2d_etdrk2(
        Nx=128,
        Ny=128,
        Lx=50.0,
        Ly=50.0,
        gamma=0.9,
        dt=0.005,
        nsteps=2000,
        noise_amp=0.01,
        dealias=True,
        t_snap=9999.0
    )

    # Plot the final field
    plt.figure(figsize=(6, 5))
    plt.imshow(psi_final, origin='lower', cmap='RdBu', extent=[0, 50, 0, 50])
    plt.colorbar(label='$\\psi(x,y)$')
    plt.title('2D Swift-Hohenberg (Final Field)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    # If we got a valid snapshot, we can also visualize that
    if psi_at_3 is not None:
        plt.figure(figsize=(6, 5))
        plt.imshow(psi_at_3, origin='lower', cmap='RdBu', extent=[0, 50, 0, 50])
        plt.colorbar(label='$\\psi(x,y)$')
        plt.title('Field at t = 3.0')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
