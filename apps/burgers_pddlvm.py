"""
burgers_pddlvm.py
-----------------
Demonstration of a purely PDDLVM-based 1D Burgers solver that compares
the PDDLVM solution against a deterministic explicit finite-difference
solver at a specified snapshot time. Both solutions are plotted for
visual inspection.

Steps:
1) Defines the PDE residual for a steady Burgers equation (u * u_x - nu u_xx = 0).
2) Builds PDDLVM forward and inverse networks using NeuralNet.py.
3) Trains the PDDLVM to minimize the PDE residual plus a latent-variable prior.
4) Calls the explicit finite-difference solver solve_burgers_explicit_fd to obtain
   a deterministic solution snapshot, and compares it with the trained PDDLVM output.
"""

import sys
print(sys.executable)
print(sys.path)

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams['text.usetex'] = False
mpl.rcParams['axes.unicode_minus'] = False

tf.random.set_seed(2025)

from pddlvm.NeuralNet import NeuralN

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

    Parameters
    ----------
    nu : float
        Viscosity parameter.
    Nx : int
        Number of spatial grid points (including boundaries).
    T : float
        Final time to which the PDE is integrated.
    t_snap : float
        Time in [0, T] at which a snapshot of the solution is returned.
    plot_result : bool
        If True, plots the final solution u(x, T).

    Returns
    -------
    x : ndarray
        1D array of spatial grid points of shape (Nx,).
    u_final : ndarray
        The numerical solution at time t = T, of shape (Nx,).
    u_snap : ndarray
        The numerical solution at time t_snap, of shape (Nx,).

    Notes
    -----
    - The spatial domain is discretized from x = -1.0 to x = 1.0 with Nx points,
      so the grid spacing is dx = 2 / (Nx - 1).
    - The time step dt is automatically adjusted so that the final time T
      is reached exactly in Nsteps steps.
    - Boundary conditions u(0) = u(Nx-1) = 0 are enforced at each time step.
    - The solution is stored at each iteration for returning the snapshot
      at t_snap (the nearest time step is used; no interpolation is performed).
    """
    import numpy as np
    import matplotlib.pyplot as plt

    x = np.linspace(-1.0, 1.0, Nx)
    dx = x[1] - x[0]

    dt_est = 0.4 * min(dx / 1.0, dx**2 / nu)  # crude guess for stable dt
    Nsteps = int(np.ceil(T / dt_est))
    dt = T / Nsteps

    solution_history = np.zeros((Nsteps + 1, Nx))

    # Initial condition
    u = -np.sin(np.pi * x)
    u[0] = 0.0
    u[-1] = 0.0
    solution_history[0, :] = u

    for n in range(Nsteps):
        u_old = u.copy()

        for j in range(1, Nx - 1):
            ux = (u_old[j + 1] - u_old[j - 1]) / (2.0 * dx)
            uxx = (u_old[j + 1] - 2.0 * u_old[j] + u_old[j - 1]) / (dx**2)
            u[j] = u_old[j] - dt * (u_old[j] * ux) + nu * dt * uxx

        u[0] = 0.0
        u[-1] = 0.0
        solution_history[n + 1, :] = u

    snap_index = int(round(t_snap / dt))
    snap_index = max(0, min(snap_index, Nsteps))
    u_snap = solution_history[snap_index, :].copy()
    u_final = solution_history[-1, :].copy()

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

Lx = 1.0
nu = 0.01
Ncol = 50

x_cols_np = np.linspace(0.0, Lx, Ncol)
x_cols = tf.constant(x_cols_np, dtype=tf.float32)

def initial_condition(x):
    """
    Provides an initial condition for the steady Burgers approach.

    Parameters
    ----------
    x : tf.Tensor
        One-dimensional spatial coordinate array.

    Returns
    -------
    tf.Tensor
        The initial condition, -sin(πx).
    """
    return -tf.math.sin(np.pi * x)

@tf.function
def pde_residual_tf(u_batch, x, nu):
    """
    Computes the PDE residual for a steady Burgers equation in 1D:
    PDE = u * u_x - nu * u_xx = 0, with boundary conditions at x=0 and x=1.

    Parameters
    ----------
    u_batch : tf.Tensor
        Shape (batch, Ncol). Each row represents a candidate solution u(x).
    x : tf.Tensor
        Shape (Ncol,). Spatial grid points in [0, Lx].
    nu : float
        Viscosity parameter.

    Returns
    -------
    tf.Tensor
        Shape (batch, Ncol). PDE residual, with boundary constraints appended
        at columns 0 and -1.
    """
    r = tf.zeros_like(u_batch)
    h = x[1] - x[0]

    u_im1 = u_batch[:, :-2]
    u_i   = u_batch[:, 1:-1]
    u_ip1 = u_batch[:, 2:]

    u_x   = (u_ip1 - u_im1) / (2.0 * h)
    u_xx  = (u_ip1 - 2.0 * u_i + u_im1) / (h * h)
    pde_interior = u_i * u_x - nu * u_xx

    bc_left  = tf.reshape(u_batch[:, 0]  - initial_condition(x[0]),  [-1, 1])
    bc_right = tf.reshape(u_batch[:, -1] - initial_condition(x[-1]), [-1, 1])

    r_cat = tf.concat([bc_left, pde_interior, bc_right], axis=1)
    return r_cat

z_dim = 2
forward_layout = np.array([z_dim, 128, 128, Ncol])
ForwardNet = NeuralN("ForwardNet", forward_layout)

inverse_layout = np.array([Ncol, 64, 64, z_dim])
InverseNet = NeuralN("InverseNet", inverse_layout)

sigma_r = 1e-4
optimizer = tf.keras.optimizers.Adam(1e-3)
tfd = tfp.distributions

def prior_z(batch_size):
    """
    Draws samples from a standard normal distribution in z_dim dimensions.

    Parameters
    ----------
    batch_size : int
        Number of latent samples to generate.

    Returns
    -------
    tf.Tensor
        Shape (batch_size, z_dim). Each row is a sample from N(0,I).
    """
    return tf.random.normal(shape=(batch_size, z_dim))

def log_diag_gaussian(sample, mean, logvar):
    """
    Computes the log probability under a diagonal Gaussian:
    N(sample; mean, exp(logvar)).

    Parameters
    ----------
    sample : tf.Tensor
        Shape (batch, dim).
    mean : tf.Tensor
        Shape (batch, dim). The Gaussian mean.
    logvar : tf.Tensor
        Shape (batch, dim). The log-variance.

    Returns
    -------
    tf.Tensor
        Shape (batch,). The log probability for each batch sample.
    """
    log2pi = tf.math.log(2.0 * np.pi)
    return tf.reduce_sum(
        -0.5 * ((sample - mean)**2 * tf.exp(-logvar) + logvar + log2pi),
        axis=1
    )

def log_prior_z(z):
    """
    Evaluates the standard normal log prior on z.

    Parameters
    ----------
    z : tf.Tensor
        Shape (batch, z_dim). The latent variable.

    Returns
    -------
    tf.Tensor
        Shape (batch,). log N(z; 0,I) for each batch row.
    """
    log2pi = tf.math.log(2.0 * np.pi)
    return tf.reduce_sum(-0.5 * (z**2 + log2pi), axis=1)

@tf.function
def train_step_pddlvm(batch_size=16):
    """
    Performs one training iteration of the PDDLVM approach for the Burgers PDE.

    1) Samples latent z ~ N(0,I).
    2) Maps z -> (mean_u, logvar_u), reparameterizes to obtain u_samp.
    3) Computes PDE residual r_samp => log likelihood logp_r.
    4) Computes inverse net p_beta(z|u_samp) => logp_beta, plus prior logp_z.
    5) Sums to get ELBO => logp_r + logp_beta + logp_z, which is maximized.

    Parameters
    ----------
    batch_size : int
        Number of latent samples in this mini-batch.

    Returns
    -------
    tf.Tensor
        The ELBO (negative of the computed loss).
    """
    with tf.GradientTape() as tape:
        z_samp = prior_z(batch_size)
        mean_u, logvar_u = ForwardNet.Map([z_samp])
        eps = tf.random.normal(shape=mean_u.shape, dtype=tf.float32)
        u_samp = mean_u + tf.exp(0.5 * logvar_u) * eps

        r_samp = pde_residual_tf(u_samp, x_cols, nu)
        logp_r = -0.5 * tf.reduce_sum(r_samp**2, axis=1) / (sigma_r**2) \
                 - 0.5 * Ncol * tf.math.log(2.0 * np.pi * (sigma_r**2))

        mean_b, logvar_b = InverseNet.Map([u_samp])
        logp_beta = log_diag_gaussian(z_samp, mean_b, logvar_b)
        logp_z = log_prior_z(z_samp)

        elbo_samp = logp_r + logp_beta + logp_z
        elbo = tf.reduce_mean(elbo_samp)
        loss = -elbo

    grads = tape.gradient(loss, ForwardNet.trainable_variables + InverseNet.trainable_variables)
    optimizer.apply_gradients(zip(grads, ForwardNet.trainable_variables + InverseNet.trainable_variables))
    return elbo

def main_train():
    """
    Executes the main PDDLVM training loop for a predetermined number of iterations.
    Plots the ELBO history over training steps.

    Returns
    -------
    list of float
        A record of ELBO values at logging intervals.
    """
    nIter = 1000
    elbo_log = []
    for it in range(nIter):
        e = train_step_pddlvm(16)
        if it % 100 == 0:
            print(f"Iter={it}, ELBO={e.numpy():.4f}")
            elbo_log.append(e.numpy())

    plt.figure()
    plt.plot(np.arange(len(elbo_log)) * 100, elbo_log, 'o-')
    plt.xlabel("Iteration")
    plt.ylabel("ELBO (PDDLVM + Burgers PDE)")
    plt.title("1D Burgers PDE with PDDLVM approach + NeuralNet.py")
    plt.show()

    return elbo_log

if __name__ == "__main__":

    # 1) Run training
    elbo_log = main_train()

    # 2) Call explicit FD solver to get snapshot
    from_burgers_x, from_burgers_final, from_burgers_snap = solve_burgers_explicit_fd(
        nu=0.01,
        Nx=201,
        T=1.0,
        t_snap=0.5,
        plot_result=False  # turn off auto-plot
    )

    # 3) Compare final solution vs FD snapshot
    # We'll sample a zero-latent for demonstration
    z_test = tf.zeros((1, z_dim), dtype=tf.float32)
    mean_u_test, logvar_u_test = ForwardNet.Map([z_test])
    u_pred_np = mean_u_test[0].numpy()  # shape=(Ncol,)

    # For plotting, resample the FD solution onto x_cols_np or do a rough comparison
    # We'll just do a direct comparison by plotting them separately,
    # acknowledging the two different grids (Ncol vs Nx=201).
    # For a fair comparison, upsample or interpolate from FD's x domain to x_cols_np or vice versa.
    # We'll do a naive approach => for demonstration only.

    # Quick interpolation of from_burgers_snap onto x_cols_np
    from scipy.interpolate import interp1d
    f_snap = interp1d(from_burgers_x, from_burgers_snap, kind='cubic')
    snap_on_pddlvm_grid = f_snap(x_cols_np)  # shape=(Ncol,)

    # Also get the initial condition for reference
    x_tf = tf.constant(x_cols_np, dtype=tf.float32)
    u_init_np = initial_condition(x_tf).numpy()

    plt.figure()
    plt.plot(x_cols_np, u_pred_np, 'r-', label=f'PDDLVM solution (sigma={sigma_r}, t=0.5)')
    plt.plot(x_cols_np, u_init_np, 'k--', label='initial condition')
    plt.plot(x_cols_np, snap_on_pddlvm_grid, 'b-', label='Explicit FD snapshot(t=0.5)')
    plt.title("Compare PDDLVM solution vs. FD snapshot\nBurgers PDE at t=0.5")
    plt.legend()
    plt.show()
