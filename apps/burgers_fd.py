import numpy as np
import matplotlib.pyplot as plt

def solve_burgers_explicit_fd(
    nu=0.01,
    Nx=201,
    T=1.0,
    t_snap=0.5,
    plot_result=True
):
    """
    Solves the 1D viscous Burgers' equation using an explicit finite-difference scheme.

    This function solves the PDE:
        ∂u/∂t + u (∂u/∂x) - ν (∂²u/∂x²) = 0
    on the domain x ∈ [-1, 1], t ∈ [0, T], subject to Dirichlet boundary conditions
    u(-1, t) = u(1, t) = 0 and an initial condition u(x, 0) = -sin(πx).

    A uniform spatial mesh of Nx points is used. Time integration is performed using
    a forward Euler scheme (explicit), with central differences in space for both
    the first and second derivatives. The function returns the mesh points x,
    the final solution u(x, T), and an additional snapshot of the solution u(x, t_snap).

    Args:
        nu (float): Viscosity parameter.
        Nx (int): Number of spatial grid points (including boundaries).
        T (float): Final time to which the PDE is integrated.
        t_snap (float): Time in [0, T] at which a snapshot of the solution is returned.
        plot_result (bool): If True, plots the final solution u(x, T).

    Returns:
        x (ndarray): 1D array of spatial grid points of shape (Nx,).
        u_final (ndarray): The numerical solution at time t = T, of shape (Nx,).
        u_snap (ndarray): The numerical solution at time t_snap, of shape (Nx,).

    Notes:
        - The spatial domain is discretized from x = -1 to x = 1 with Nx points,
          so the grid spacing is dx = 2 / (Nx - 1).
        - The time step dt is automatically adjusted so that the final time T
          is reached exactly in Nsteps steps.
        - Boundary conditions u(0) = u(Nx-1) = 0 are enforced at each time step.
        - The solution is stored at each iteration for the purpose of returning
          the snapshot at t_snap (the nearest time step is used; no interpolation
          is performed).

    Example:
        >>> x_vals, u_end, u_mid = solve_burgers_explicit_fd(nu=0.01, Nx=201, T=1.0, t_snap=0.5)
        >>> # x_vals is the spatial grid
        >>> # u_end is u(x, T=1.0)
        >>> # u_mid is u(x, t_snap=0.5)
    """
    # 1. Setup spatial grid
    x = np.linspace(-1.0, 1.0, Nx)
    dx = x[1] - x[0]

    # 2. Choose time step based on simple stability criteria
    #    dt is refined so that Nsteps * dt = T exactly.
    dt_est = 0.4 * min(dx / 1.0, dx**2 / nu)  # crude guess for stable dt
    Nsteps = int(np.ceil(T / dt_est))
    dt = T / Nsteps  # adjust so final time = T exactly

    # 3. Allocate array to store solution at each time step for snapshot retrieval
    solution_history = np.zeros((Nsteps + 1, Nx))

    # 4. Set initial condition
    u = -np.sin(np.pi * x)  # shape: (Nx,)
    # Enforce Dirichlet boundaries at x=-1 and x=1
    u[0] = 0.0
    u[-1] = 0.0

    # Store initial condition as step 0
    solution_history[0, :] = u

    # 5. Time-marching loop
    for n in range(Nsteps):
        u_old = u.copy()

        # Update interior points: j = 1, ..., Nx-2
        for j in range(1, Nx - 1):
            # Central difference for first derivative
            ux = (u_old[j + 1] - u_old[j - 1]) / (2.0 * dx)

            # Second derivative
            uxx = (u_old[j + 1] - 2.0 * u_old[j] + u_old[j - 1]) / (dx**2)

            # Explicit update
            # u_j^{n+1} = u_j^n - dt * (u_j^n * ux) + nu * dt * uxx
            u[j] = u_old[j] - dt * (u_old[j] * ux) + nu * dt * uxx

        # Enforce Dirichlet boundaries at each step
        u[0] = 0.0
        u[-1] = 0.0

        # Store solution at step n+1
        solution_history[n + 1, :] = u

    # 6. Find the snapshot closest to t_snap (no interpolation)
    #    Time step index for snapshot
    snap_index = int(round(t_snap / dt))
    snap_index = max(0, min(snap_index, Nsteps))  # clamp the index in [0, Nsteps]
    u_snap = solution_history[snap_index, :].copy()

    # 7. Prepare final outputs
    u_final = solution_history[-1, :].copy()

    # 8. Plot the final solution if requested
    if plot_result:
        plt.figure(figsize=(6, 4))
        plt.plot(x, u_final, 'b-', label=f'u(x, T={T})')
        plt.title(f'Viscous Burgers (Explicit FD), T={T}, Nx={Nx}, nu={nu}')
        plt.xlabel('x')
        plt.ylabel('u(x,T)')
        plt.grid(True)
        plt.legend()
        plt.show()

    return x, u_final, u_snap


if __name__ == '__main__':
    # Example usage
    x_vals, u_end, u_mid = solve_burgers_explicit_fd(
        nu=0.01,
        Nx=201,
        T=1.0,
        t_snap=0.5,
        plot_result=True
    )
    print(f"Final time solution shape = {u_end.shape}, snapshot shape = {u_mid.shape}")
import numpy as np
import matplotlib.pyplot as plt

def solve_burgers_explicit_fd(
    nu=0.01,
    Nx=201,
    T=1.0,
    t_snap=0.5,
    plot_result=True
):
    """
    Solves the 1D viscous Burgers' equation using an explicit finite-difference scheme.

    This function solves the PDE:
        ∂u/∂t + u (∂u/∂x) - ν (∂²u/∂x²) = 0
    on the domain x ∈ [-1, 1], t ∈ [0, T], subject to Dirichlet boundary conditions
    u(-1, t) = u(1, t) = 0 and an initial condition u(x, 0) = -sin(πx).

    A uniform spatial mesh of Nx points is used. Time integration is performed using
    a forward Euler scheme (explicit), with central differences in space for both
    the first and second derivatives. The function returns the mesh points x,
    the final solution u(x, T), and an additional snapshot of the solution u(x, t_snap).

    Args:
        nu (float): Viscosity parameter.
        Nx (int): Number of spatial grid points (including boundaries).
        T (float): Final time to which the PDE is integrated.
        t_snap (float): Time in [0, T] at which a snapshot of the solution is returned.
        plot_result (bool): If True, plots the final solution u(x, T).

    Returns:
        x (ndarray): 1D array of spatial grid points of shape (Nx,).
        u_final (ndarray): The numerical solution at time t = T, of shape (Nx,).
        u_snap (ndarray): The numerical solution at time t_snap, of shape (Nx,).

    Notes:
        - The spatial domain is discretized from x = -1 to x = 1 with Nx points,
          so the grid spacing is dx = 2 / (Nx - 1).
        - The time step dt is automatically adjusted so that the final time T
          is reached exactly in Nsteps steps.
        - Boundary conditions u(0) = u(Nx-1) = 0 are enforced at each time step.
        - The solution is stored at each iteration for the purpose of returning
          the snapshot at t_snap (the nearest time step is used; no interpolation
          is performed).

    Example:
        >>> x_vals, u_end, u_mid = solve_burgers_explicit_fd(nu=0.01, Nx=201, T=1.0, t_snap=0.5)
        >>> # x_vals is the spatial grid
        >>> # u_end is u(x, T=1.0)
        >>> # u_mid is u(x, t_snap=0.5)
    """
    # 1. Setup spatial grid
    x = np.linspace(-1.0, 1.0, Nx)
    dx = x[1] - x[0]

    # 2. Choose time step based on simple stability criteria
    #    dt is refined so that Nsteps * dt = T exactly.
    dt_est = 0.4 * min(dx / 1.0, dx**2 / nu)  # crude guess for stable dt
    Nsteps = int(np.ceil(T / dt_est))
    dt = T / Nsteps  # adjust so final time = T exactly

    # 3. Allocate array to store solution at each time step for snapshot retrieval
    solution_history = np.zeros((Nsteps + 1, Nx))

    # 4. Set initial condition
    u = -np.sin(np.pi * x)  # shape: (Nx,)
    # Enforce Dirichlet boundaries at x=-1 and x=1
    u[0] = 0.0
    u[-1] = 0.0

    # Store initial condition as step 0
    solution_history[0, :] = u

    # 5. Time-marching loop
    for n in range(Nsteps):
        u_old = u.copy()

        # Update interior points: j = 1, ..., Nx-2
        for j in range(1, Nx - 1):
            # Central difference for first derivative
            ux = (u_old[j + 1] - u_old[j - 1]) / (2.0 * dx)

            # Second derivative
            uxx = (u_old[j + 1] - 2.0 * u_old[j] + u_old[j - 1]) / (dx**2)

            # Explicit update
            # u_j^{n+1} = u_j^n - dt * (u_j^n * ux) + nu * dt * uxx
            u[j] = u_old[j] - dt * (u_old[j] * ux) + nu * dt * uxx

        # Enforce Dirichlet boundaries at each step
        u[0] = 0.0
        u[-1] = 0.0

        # Store solution at step n+1
        solution_history[n + 1, :] = u

    # 6. Find the snapshot closest to t_snap (no interpolation)
    #    Time step index for snapshot
    snap_index = int(round(t_snap / dt))
    snap_index = max(0, min(snap_index, Nsteps))  # clamp the index in [0, Nsteps]
    u_snap = solution_history[snap_index, :].copy()

    # 7. Prepare final outputs
    u_final = solution_history[-1, :].copy()

    # 8. Plot the final solution if requested
    if plot_result:
        plt.figure(figsize=(6, 4))
        plt.plot(x, u_final, 'b-', label=f'u(x, T={T})')
        plt.title(f'Viscous Burgers (Explicit FD), T={T}, Nx={Nx}, nu={nu}')
        plt.xlabel('x')
        plt.ylabel('u(x,T)')
        plt.grid(True)
        plt.legend()
        plt.show()

    return x, u_final, u_snap


if __name__ == '__main__':
    # Example usage
    x_vals, u_end, u_mid = solve_burgers_explicit_fd(
        nu=0.01,
        Nx=201,
        T=1.0,
        t_snap=0.5,
        plot_result=True
    )
    print(f"Final time solution shape = {u_end.shape}, snapshot shape = {u_mid.shape}")
