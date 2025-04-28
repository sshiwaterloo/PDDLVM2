"""
ns_pddlvm.py
------------
Demonstration of a purely PDDLVM approach for a 2D Navier–Stokes solver in vorticity form,
comparing the final solution at time t_snap to a pseudo-spectral "ground truth."

This script illustrates:
1) Construction of a forward network that maps latent variables to vorticity fields.
2) Construction of an inverse network that maps vorticity fields back to latent variables.
3) A training loop that treats PDE residual as a Gaussian likelihood for residual=0, plus a prior on z.
4) Comparison of the final predicted vorticity with a reference solution from a spectral solver.
"""

import sys
print(sys.executable)
print(sys.path)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams["text.usetex"] = False
mpl.rcParams["axes.unicode_minus"] = False

import tensorflow as tf
import tensorflow_probability as tfp

tf.random.set_seed(2025)

##########################
# 1) Import the spectral solver
##########################
from ns_spectral import ns_spectral_solve


##########################
# PDE Setup
##########################
nx, ny = 32, 32
nTotal = nx * ny
nu = 0.01

tmax = 1.0
t_snap = 0.2  # Time in [0, tmax] where the script compares the PDDLVM solution with ground truth.


##########################
# 2) Forward and Inverse Networks
##########################
from pddlvm.NeuralNet import NeuralN

layout_forward = np.array([4, 128, 128, nTotal])
ForwardNet = NeuralN("ForwardNet", layout_forward)
"""
Forward network that consumes a latent vector z in R^4 and produces (mean_w, logvar_w),
where mean_w and logvar_w each have dimension nTotal. The final reparameterized output
is interpreted as the vorticity field on a 2D grid of size (ny, nx).
"""

layout_inverse = np.array([nTotal, 64, 64, 4])
InverseNet = NeuralN("InverseNet", layout_inverse)
"""
Inverse network that consumes a flattened vorticity field (batch, nTotal) and outputs
(mean_z, logvar_z) in R^4. The code attempts to invert the PDE solution to recover
the latent vector z.
"""

sigma_r = 1e-2
optimizer = tf.keras.optimizers.Adam(1e-3)

def prior_z(batch_size=16):
    """
    Samples latent vectors from a standard normal prior in R^4.

    Parameters
    ----------
    batch_size : int
        Number of latent samples.

    Returns
    -------
    tf.Tensor
        Shape (batch_size, 4). Each row is z ~ N(0,I).
    """
    return tf.random.normal(shape=(batch_size, 4))

def log_diag_gaussian(sample, mean, logvar):
    """
    Computes log probability under a diagonal Gaussian distribution:
    log N(sample; mean, exp(logvar)).

    Parameters
    ----------
    sample : tf.Tensor
        Shape (batch, dim).
    mean : tf.Tensor
        Shape (batch, dim). Gaussian mean.
    logvar : tf.Tensor
        Shape (batch, dim). Log-variance.

    Returns
    -------
    tf.Tensor
        Shape (batch,). Log probability of each sample.
    """
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -0.5 * ((sample - mean)**2 * tf.exp(-logvar) + logvar + log2pi),
        axis=1
    )

def log_prior_z(z):
    """
    Evaluates the standard normal log prior of z in R^4.

    Parameters
    ----------
    z : tf.Tensor
        Shape (batch, 4). Latent variable.

    Returns
    -------
    tf.Tensor
        Shape (batch,). Log probability under N(0,I).
    """
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(-0.5 * (z**2 + log2pi), axis=1)


##########################
# PDE Residual operator
##########################
dx = 1.0
dy = 1.0

@tf.function
def poisson_solve_2d(w_2d):
    """
    Naive Poisson solve for streamfunction psi from vorticity w:
      Laplacian(psi) = - w
    using a Jacobi iteration or similarly. This is purely for demonstration.

    Parameters
    ----------
    w_2d : tf.Tensor
        Shape (batch, ny, nx). 2D vorticity field for each batch sample.

    Returns
    -------
    psi_2d : tf.Tensor
        Shape (batch, ny, nx). Approximated streamfunction.
    """
    batch_size = tf.shape(w_2d)[0]
    # Initialize psi to zero
    psi = tf.zeros_like(w_2d)

    # Number of Jacobi iterations
    n_iter = 30

    # Expand boundaries by 1 for wrap
    for _ in range(n_iter):
        # Periodic padding
        psi_pad = tf.concat([psi[:, :, -1:], psi, psi[:, :, :1]], axis=2)  # wrap x
        psi_pad = tf.concat([psi_pad[:, -1:, :], psi_pad, psi_pad[:, :1, :]], axis=1)  # wrap y

        # similarly for w
        w_pad = tf.concat([w_2d[:, :, -1:], w_2d, w_2d[:, :, :1]], axis=2)
        w_pad = tf.concat([w_pad[:, -1:, :], w_pad, w_pad[:, :1, :]], axis=1)

        # central neighbors
        psi_center = psi_pad[:, 1:-1, 1:-1]
        psi_up     = psi_pad[:, 0:-2, 1:-1]
        psi_down   = psi_pad[:, 2:,   1:-1]
        psi_left   = psi_pad[:, 1:-1, 0:-2]
        psi_right  = psi_pad[:, 1:-1, 2:]

        w_center   = w_pad[:, 1:-1, 1:-1]

        # Jacobi update for Laplacian(psi) = - w => psi_new = (neighbors - dx^2 * w_center)/4
        # assuming dx=dy for demonstration
        psi_new = 0.25 * (psi_up + psi_down + psi_left + psi_right + (dx*dx)*w_center)
        psi = psi_new

    return psi

@tf.function
def pde_residual_tf(w_batch):
    """
    Computes a finite-difference approximation of the 2D Navier–Stokes vorticity
    equation in "steady" form:

      0 = (u · ∇)w - nu * Laplacian(w)

    by:
      1) Reshaping w_batch to (batch, ny, nx)
      2) Poisson solve for streamfunction psi => velocity (u,v)
      3) Evaluate convective term (u · ∇)w and Laplacian of w
      4) Return (batch, Nx*Ny) flattened PDE mismatch

    Parameters
    ----------
    w_batch : tf.Tensor
        Shape (batch, nx*ny). Each row is a flattened vorticity field.

    Returns
    -------
    tf.Tensor
        Shape (batch, nx*ny). PDE residual for each sample.
    """
    batch_size = tf.shape(w_batch)[0]

    # reshape w => (batch, ny, nx)
    w_2d = tf.reshape(w_batch, [batch_size, ny, nx])

    # 1) Poisson solve => streamfunction
    psi_2d = poisson_solve_2d(w_2d)

    # 2) velocity from psi => u_x= +dpsi/dy, u_y= -dpsi/dx
    # do periodic wrap again
    psi_pad = tf.concat([psi_2d[:, :, -1:], psi_2d, psi_2d[:, :, :1]], axis=2)
    psi_pad = tf.concat([psi_pad[:, -1:, :], psi_pad, psi_pad[:, :1, :]], axis=1)

    psi_center = psi_pad[:, 1:-1, 1:-1]
    psi_up     = psi_pad[:, 0:-2, 1:-1]
    psi_down   = psi_pad[:, 2:,   1:-1]
    psi_left   = psi_pad[:, 1:-1, 0:-2]
    psi_right  = psi_pad[:, 1:-1, 2:]

    # central difference
    dpsi_dy = 0.5*(psi_down - psi_up)/dy
    dpsi_dx = 0.5*(psi_right - psi_left)/dx

    # velocity => (u, v) = (+dpsi_dy, -dpsi_dx)
    u_2d = dpsi_dy
    v_2d = -dpsi_dx

    # 3) convective term => (u · ∇) w
    # pad w => periodic
    w_pad = tf.concat([w_2d[:, :, -1:], w_2d, w_2d[:, :, :1]], axis=2)
    w_pad = tf.concat([w_pad[:, -1:, :], w_pad, w_pad[:, :1, :]], axis=1)

    w_center = w_pad[:, 1:-1, 1:-1]
    w_up     = w_pad[:, 0:-2, 1:-1]
    w_down   = w_pad[:, 2:,   1:-1]
    w_left   = w_pad[:, 1:-1, 0:-2]
    w_right  = w_pad[:, 1:-1, 2:]

    dw_dx = 0.5*(w_right - w_left)/dx
    dw_dy = 0.5*(w_down  - w_up)/dy

    conv_term = u_2d * dw_dx + v_2d * dw_dy

    # 4) Laplacian of w => central difference approach
    lap_w = (w_up + w_down + w_left + w_right - 4.0*w_center)/(dx*dx)

    # PDE residual => conv_term - nu*lap_w
    # shape => (batch, ny, nx)
    pde_res_2d = conv_term - nu*lap_w

    # flatten => (batch, nx*ny)
    pde_res_flat = tf.reshape(pde_res_2d, [batch_size, nx*ny])

    return pde_res_flat

@tf.function
def train_step_pddlvm(batch_size=16):
    """
    Performs one training step of the PDE-driven PDDLVM approach:

    1) Samples a batch of latent vectors z ~ N(0,I).
    2) Passes z through ForwardNet => (mean_w, logvar_w), reparameterizes to get w_samp.
    3) Evaluates PDE residual => log-likelihood of residual=0 => logp_r.
    4) Evaluates InverseNet => distribution of z given w_samp => logp_beta.
    5) Adds log prior logp_z => standard normal on z.
    6) Summation => ELBO. Minimizes negative ELBO.

    Parameters
    ----------
    batch_size : int
        Number of samples in mini-batch.

    Returns
    -------
    tf.Tensor
        The ELBO as a scalar float, representing negative of the final loss.
    """
    with tf.GradientTape() as tape:
        z_samp = prior_z(batch_size)
        mean_w, logvar_w = ForwardNet.Map([z_samp])
        eps = tf.random.normal(shape=tf.shape(mean_w))
        w_samp = mean_w + tf.exp(0.5 * logvar_w) * eps

        r_samp = pde_residual_tf(w_samp)
        M = tf.shape(r_samp)[1]
        logp_r = (
            -0.5 * tf.reduce_sum(r_samp**2, axis=1) / (sigma_r**2)
            - 0.5 * tf.cast(M, tf.float32) * tf.math.log(2. * np.pi * (sigma_r**2))
        )

        mean_b, logv_b = InverseNet.Map([w_samp])
        logp_beta = log_diag_gaussian(z_samp, mean_b, logv_b)
        logp_z = log_prior_z(z_samp)

        elbo_samp = logp_r + logp_beta + logp_z
        elbo = tf.reduce_mean(elbo_samp)
        loss = -elbo

    grads = tape.gradient(loss, ForwardNet.trainable_variables + InverseNet.trainable_variables)
    optimizer.apply_gradients(zip(grads, ForwardNet.trainable_variables + InverseNet.trainable_variables))
    return elbo


##########################
# 3) Train
##########################
nIter = 1000
elbo_log = []
for i in range(nIter):
    e = train_step_pddlvm(16)
    if i % 100 == 0:
        print(f"Iter={i}, ELBO={e.numpy():.4f}")
        elbo_log.append(e.numpy())

plt.figure()
plt.plot(np.arange(len(elbo_log))*50, elbo_log, 'o-')
plt.xlabel("Iteration")
plt.ylabel("log-ELBO")
plt.title(f"2D Navier–Stokes PDE with PDDLVM - Demo (time snapshot={t_snap})")
plt.show()


##########################
# 4) Ground truth from ns_spectral => w(t_snap) & velocity
##########################
w0 = np.zeros((ny, nx), dtype=np.float64)
dt = 0.001
w_gt, ux_gt, uy_gt = ns_spectral_solve(
    nx=nx,
    ny=ny,
    nu=nu,
    tmax=tmax,
    dt=dt,
    t_snap=t_snap,
    w0=w0,
    return_velocity=True
)
"""
w_gt : ndarray of shape (ny, nx)
    The reference vorticity at t=t_snap.
ux_gt, uy_gt : ndarray of shape (ny, nx)
    The reference velocity field in x, y directions.
"""


##########################
# 5) Compare final
##########################
mean_w_test, logv_w_test = ForwardNet.Map([tf.zeros((1,4), dtype=tf.float32)])
w_pred_np = mean_w_test[0].numpy().reshape((ny, nx))

fig, ax = plt.subplots(ncols=2, figsize=(10,4))
im0 = ax[0].contourf(w_gt, 50, cmap='jet')
ax[0].set_title(f"Ground truth w(x,y) at t={t_snap}")
im1 = ax[1].contourf(w_pred_np, 50, cmap='jet')
ax[1].set_title(f"PDDLVM mean w(x,y) at t={t_snap}")
plt.colorbar(im1, ax=ax[1])
plt.tight_layout()
plt.show()

# Additional velocity comparisons can be added if the model is extended to produce velocity fields.