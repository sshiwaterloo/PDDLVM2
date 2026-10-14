import _bootstrap  # noqa: F401  (repo root on sys.path)
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D projection
from pddlvm.plotting import save_or_show, show_requested

def solve_burgers_split_step_dirichlet(
    nu=0.01,
    Nx=128,
    T=1.0,
    dt=0.001,
    Nsub=1,
    t_snap=0.5,
    plot_final=True,
    plot_3d=True
):
    """
    Implements a split-step method for the 1D viscous Burgers equation with Dirichlet boundaries
    using a sine-series approach for the diffusion part and a real-space explicit update for
    the nonlinear part. The domain is x in [-1, 1], with u(-1, t) = u(1, t) = 0.

    The PDE is:
        ∂_t u + u ∂_x u - ν ∂_{xx} u = 0,
    with
        u(-1, t) = 0,  u(1, t) = 0,
        u(x, 0) = -sin(π x).

    The algorithm is split into two parts each time step:
        (A) A diffusion step solved by discrete sine transform (DST), which enforces Dirichlet boundaries.
        (B) A nonlinear (advection) step in real space with an explicit Euler scheme.

    Args:
        nu (float): Viscosity parameter.
        Nx (int): Number of interior grid points (not counting boundary).
        T (float): Final simulation time.
        dt (float): Time step for splitting steps.
        Nsub (int): Number of sub-steps for the nonlinear portion per main step.
        t_snap (float): A time in [0, T] at which the solution is recorded and returned.
        plot_final (bool): Toggles plotting of the solution at final time T.
        plot_3d (bool): Toggles plotting of the entire solution surface (x vs. t).

    Returns:
        x (np.ndarray): 1D array of spatial points (interior only).
        u_final (np.ndarray): The numerical solution at final time T (shape: Nx).
        u_snap (np.ndarray): The numerical solution at time t_snap (shape: Nx).
        solution_history (np.ndarray): 2D array of shape (Nsteps+1, Nx), storing the solution at all steps.
                                       The row index corresponds to the time step, and columns to spatial points.
    """
    ############################################################################
    # 1. Spatial Grid and Boundary Setup
    ############################################################################
    # Domain is [-1, 1]. The boundary points are excluded from the interior grid.
    # Nx interior points are distributed in (-1, 1).
    # The total grid has Nx+2 points if boundaries are included, but this code only
    # stores and evolves the interior points. Boundary values remain zero (Dirichlet).
    x_full = np.linspace(-1.0, 1.0, Nx+2)  # total Nx+2 points
    x = x_full[1:-1]                      # interior points
    dx = x[1] - x[0]

    # Initial condition on the interior
    u = -np.sin(np.pi * x)

    ############################################################################
    # 2. Time Discretization
    ############################################################################
    Nsteps = int(np.ceil(T / dt))
    dt_eff = T / Nsteps
    dt_sub = dt_eff / Nsub

    # Times array (for recording the solution if needed)
    times = np.linspace(0, T, Nsteps+1)

    ############################################################################
    # 3. Discrete Sine Transform Utilities
    ############################################################################
    # The DST approach enforces u=0 at x=-1 and x=1. The transform is applied
    # only over the interior points, interpreting them as an odd extension.
    # Index k in the transform corresponds to mode sin(k * π * (x+1) / 2 ).
    #
    # DCT/DST details:
    #   For an array a of length Nx, define the positions as j=1..Nx,
    #   Then apply the DST-II / inverse DST-II or use direct libraries like scipy.
    #
    # This code implements a minimal DST / inverse DST (type-II).
    # The interior domain is mapped to [0, Nx+1] in an integer sense to match standard formulas.
    # The transform relies on the assumption that the function is zero at both ends.

    def dst(u_array):
        """
        Discrete sine transform (DST-II) of a real array (length Nx).
        Returns the sine-series coefficients (length Nx).
        """
        # index n=0..(Nx-1) internally. Shift or scale factors used for type-II.
        # Direct formula-based approach.
        U = np.zeros_like(u_array)
        for k_ in range(Nx):
            k = k_ + 1  # 1-based for formula
            s = 0.0
            for n_ in range(Nx):
                n = n_ + 1
                s += u_array[n_-0] * np.sin(np.pi * k * n / (Nx+1))
            U[k_] = 2.0 * s / (Nx+1)
        return U

    def idst(U_array):
        """
        Inverse discrete sine transform (DST-II).
        Reconstructs the spatial values from the sine coefficients.
        """
        u_rec = np.zeros_like(U_array)
        for n_ in range(Nx):
            n = n_ + 1
            s = 0.0
            for k_ in range(Nx):
                k = k_ + 1
                s += U_array[k_] * np.sin(np.pi * k * n / (Nx+1))
            u_rec[n_] = 2.0 * s / (Nx+1)
        return u_rec

    # Precompute wave numbers for the discrete sine transform
    # For mode k, the wavenumber factor is: alpha_k = (π k / (2L)), but domain length is 2.
    # The typical formula for Dirichlet BC on [-1,1] sets length = 2 => half-length = 1.
    # The discrete index k in [1..Nx].
    # Then the actual continuous wave number is kπ/(2*1) = kπ/2, but the correct discrete
    # approach for the Laplacian in DST is alpha_k^2 = (kπ/(2))^2.
    # This approach is consistent with the internal indexing.
    k_vals = np.arange(1, Nx+1, dtype=float)  # k=1..Nx
    alpha = (np.pi * k_vals) / 2.0  # matches domain half-length = 1

    ############################################################################
    # 4. Data Storage for 3D Plot
    ############################################################################
    solution_history = np.zeros((Nsteps+1, Nx))
    solution_history[0, :] = u

    ############################################################################
    # 5. Time Loop
    ############################################################################
    # Each iteration:
    #   (A) Solve diffusion:  u_t = ν u_xx  on [t, t+dt_eff]
    #        - transform by DST, multiply each mode by exp(-ν α_k^2 dt_eff).
    #   (B) Solve nonlinear:  u_t = -u u_x, sub-stepped in real space.

    def compute_spatial_derivative(u_array):
        """
        Computes derivative in real space with second-order finite differences,
        ignoring boundary since u=0 at both ends.
        """
        dudx = np.zeros_like(u_array)
        dudx[1:-1] = (u_array[2:] - u_array[:-2]) / (2.0 * dx)
        return dudx

    for step in range(Nsteps):
        # --- (A) Diffusion step via sine transform ---
        U_k = dst(u)
        # Each mode evolves by factor e^(-nu * alpha_k^2 * dt_eff)
        diff_factor = np.exp(-nu * alpha**2 * dt_eff)
        U_k *= diff_factor
        u = idst(U_k)

        # --- (B) Nonlinear step (split into Nsub sub-steps) ---
        for _ in range(Nsub):
            ux = compute_spatial_derivative(u)
            # Forward Euler for u_t = -u * u_x
            u = u - dt_sub * (u * ux)

        solution_history[step+1, :] = u

    ############################################################################
    # 6. Snapshot Retrieval
    ############################################################################
    # Finds the integer step closest to t_snap
    # No interpolation is performed here; the nearest step is returned.
    step_snap = int(round(t_snap / dt_eff))
    step_snap = max(0, min(step_snap, Nsteps))  # clamp range
    u_snap = solution_history[step_snap, :].copy()
    u_final = solution_history[-1, :].copy()

    ############################################################################
    # 7. Plot final slice
    ############################################################################
    if plot_final:
        plt.figure(figsize=(6, 4))
        # Plot interior domain from -1..1, but remember the boundary is always zero
        x_plot = x
        u_plot = u_final
        plt.plot(x_plot, u_plot, 'b-', label='u(x,T)')
        plt.hlines(0.0, x_plot[0], x_plot[-1], linestyles='--', color='gray', alpha=0.5)
        plt.title(f"Viscous Burgers with Dirichlet BC, T={T}, Nx={Nx}, nu={nu}")
        plt.xlabel("x")
        plt.ylabel("u(x,T)")
        plt.grid(True)
        plt.legend()
        save_or_show(plt.gcf(), "artifacts/burgers_spectral_u.png",
                     show=show_requested())

    ############################################################################
    # 8. Plot 3D surface
    ############################################################################
    if plot_3d:
        # The interior domain is x, times are times
        # shape => times: (Nsteps+1,), x: (Nx,)
        X, TT = np.meshgrid(x, times)
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, TT, solution_history, cmap='viridis', linewidth=0, antialiased=True)
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        ax.set_zlabel("u(x,t)")
        ax.set_title(f"Burgers Dirichlet BC, Nx={Nx}, ν={nu}")
        plt.tight_layout()
        save_or_show(plt.gcf(), "artifacts/burgers_spectral_3d.png",
                     show=show_requested())

    return x, u_final, u_snap, solution_history


if __name__ == "__main__":
    # Example usage of the modified function
    x_vals, u_end, u_mid, hist = solve_burgers_split_step_dirichlet(
        nu=0.01,
        Nx=64,
        T=1.0,
        dt=0.001,
        Nsub=5,
        t_snap=1.0,
        plot_final=True,
        plot_3d=True
    )