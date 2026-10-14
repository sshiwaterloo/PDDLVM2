"""
sh_pddlvm.py
------------
Demonstration of a pure PDDLVM approach for the 2D Swift-Hohenberg equation. The PDE is:

  psi_t = gamma * psi - (1 + ∇^2)^2 psi - psi^3

The code treats the time derivative as zero at t_snap, effectively modeling a "steady" PDE snapshot.
It then compares the learned PDDLVM solution against a numeric reference from `sh_spectral.py` at time t_snap.

Usage:
  1) python sh_pddlvm.py
  2) Trains a PDDLVM for a 2D domain, then calls `sh_spectral.py` to obtain a ground-truth snapshot at t_snap
  3) Displays side-by-side plots of ground-truth and predicted solutions

WARNING (physics consistency): the PDE residual here is the *steady*
Swift-Hohenberg operator (the time derivative is dropped), while
`sh_spectral` integrates the *time-dependent* equation to t_snap.  The
side-by-side comparison is a sanity check only.
"""

import _bootstrap  # noqa: F401  (repo root on sys.path)
import sys
print(sys.executable)
print(sys.path)

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_probability as tfp

tf.random.set_seed(42)

##########################
# 1) Swift–Hohenberg spectral solver import
##########################
from sh_spectral import swift_hohenberg_2d_etdrk2

##########################
# PDE/Domain Setup
##########################
Nx, Ny = 128, 128
Lx, Ly = 50.0, 50.0
gamma = 0.9

dt_spec = 0.005
nsteps  = 2000
t_snap  = 3.0

##########################
# 2) Construct PDDLVM for "steady" PDE
##########################
# The PDE is simplified as:
#   0 = gamma * psi - (1 + ∇^2)^2 psi - psi^3
# ignoring time derivative. A real code would compute a numeric PDE residual.

nTotal = Nx * Ny
z_dim  = 8

from pddlvm.NeuralNet import NeuralN
from pddlvm.plotting import save_or_show

SHOW = "--show" in sys.argv

layout_forward = np.array([z_dim, 128, 128, nTotal])
ForwardNet = NeuralN("ForwardNet", layout_forward)
"""
Forward network:
    Input: latent vector z (dim=z_dim)
    Output: (mean_psi, logvar_psi), each shape=(batch, nTotal)
    Reparameterized to produce a 2D field psi_samp in shape (Ny, Nx) flattened.
"""

layout_inverse = np.array([nTotal, 64, 64, z_dim])
InverseNet = NeuralN("InverseNet", layout_inverse)
"""
Inverse network:
    Input: a flattened field psi in shape=(batch, nTotal)
    Output: (mean_z, logvar_z) in R^(z_dim), allowing reconstruction of the latent vector.
"""

optimizer = tf.keras.optimizers.Adam(1e-3)
sigma_r = 1e-2

def prior_z(batch_size=16):
    """
    Samples latent vectors from a standard normal distribution.

    Parameters
    ----------
    batch_size : int
        Number of latent samples

    Returns
    -------
    tf.Tensor
        Shape (batch_size, z_dim). Each row is z ~ N(0,I)
    """
    return tf.random.normal(shape=(batch_size, z_dim))

def log_diag_gaussian(sample, mean, logvar):
    """
    Computes log probability under a diagonal Gaussian:
    log N(sample; mean, exp(logvar)).

    Parameters
    ----------
    sample : tf.Tensor
        Shape (batch, dim)
    mean : tf.Tensor
        Shape (batch, dim). Mean vector
    logvar : tf.Tensor
        Shape (batch, dim). Log-variance

    Returns
    -------
    tf.Tensor
        Shape (batch,). Log probability for each sample
    """
    log2pi = tf.math.log(2.0 * np.pi)
    return tf.reduce_sum(
        -0.5 * ((sample - mean)**2 * tf.exp(-logvar) + logvar + log2pi),
        axis=1
    )

def log_prior_z(z):
    """
    Evaluates standard normal prior on latent variable z in R^(z_dim).

    Parameters
    ----------
    z : tf.Tensor
        Shape (batch, z_dim)

    Returns
    -------
    tf.Tensor
        Shape (batch,). Log probability of z under N(0,I)
    """
    log2pi = tf.math.log(2.0 * np.pi)
    return tf.reduce_sum(-0.5 * (z**2 + log2pi), axis=1)

##########################
# 3) Placeholder PDE Residual
##########################
# Compute dx, dy from the domain sizes
dx = Lx / Nx
dy = Ly / Ny

@tf.function
def pde_residual_tf(psi_batch):
    """
    Computes a finite-difference residual of the Swift–Hohenberg PDE in steady form:
      residual = gamma * psi - (1 + Laplacian)^2 psi - psi^3

    A simple central FD approach is used for Laplacian(psi). The domain is flattened row-major.
    The function constructs a 2D field for each batch sample, then returns a flattened residual.

    Parameters
    ----------
    psi_batch : tf.Tensor
        Shape (batch, Nx*Ny). Flattened 2D fields of psi.

    Returns
    -------
    tf.Tensor
        Shape (batch, Nx*Ny). Residual of the PDE for each batch sample.
    """
    batch_size = tf.shape(psi_batch)[0]

    # Reshape from (batch, Nx*Ny) -> (batch, Nx, Ny)
    psi_2d = tf.reshape(psi_batch, [batch_size, Nx, Ny])

    # Compute a single Laplacian using central differences.
    # For each sample in the batch, handle interior points, ignoring boundary or using periodic wrap as desired.

    # Example shows periodic approach for demonstration:
    # pad each dimension by 1 on both sides, then do central differences.

    # pad psi along x and y with wrap
    psi_pad = tf.concat([psi_2d[:, -1:, :], psi_2d, psi_2d[:, 0:1, :]], axis=1)    # wrap in x-direction
    psi_pad = tf.concat([psi_pad[:, :, -1:], psi_pad, psi_pad[:, :, 0:1]], axis=2) # wrap in y-direction

    # Indices: center => (i+1, j+1)
    #  neighbors => up, down, left, right for the 2D domain
    # Laplacian(psi)(i,j) ~ (psi(i+1,j)+psi(i-1,j)+psi(i,j+1)+psi(i,j-1)-4*psi(i,j)) / (dx^2) or (dy^2)
    # Here, dx=dy => uniform. The code merges them as needed.

    # Slicing for neighbors:
    psi_center = psi_pad[:, 1:-1, 1:-1]
    psi_up     = psi_pad[:, 0:-2, 1:-1]
    psi_down   = psi_pad[:, 2:,   1:-1]
    psi_left   = psi_pad[:, 1:-1, 0:-2]
    psi_right  = psi_pad[:, 1:-1, 2:]

    # Laplacian with dx=dy
    lap_psi = (psi_up + psi_down + psi_left + psi_right - 4.0*psi_center) / (dx*dx)

    # Next, apply Laplacian again to get (Laplacian^2)(psi). Also add psi first, to form (1 + Lap)^2.

    # (1 + Lap) * psi => psi_center + lap_psi
    # so define "temp = psi_center + lap_psi", then apply the same FD approach to "temp"
    temp = psi_center + lap_psi

    # For the second pass, must also do periodic wrap again:
    temp_pad = tf.concat([temp[:, -1:, :], temp, temp[:, 0:1, :]], axis=1)
    temp_pad = tf.concat([temp_pad[:, :, -1:], temp_pad, temp_pad[:, :, 0:1]], axis=2)

    temp_center = temp_pad[:, 1:-1, 1:-1]
    temp_up     = temp_pad[:, 0:-2, 1:-1]
    temp_down   = temp_pad[:, 2:,   1:-1]
    temp_left   = temp_pad[:, 1:-1, 0:-2]
    temp_right  = temp_pad[:, 1:-1, 2:]

    lap_temp = (temp_up + temp_down + temp_left + temp_right - 4.0*temp_center) / (dx*dx)
    # This yields (1+Lap)^2 psi = temp_center + lap_temp.

    op_psi = temp_center + lap_temp  # (1 + Laplacian)^2 psi

    # Subtract psi^3 and add gamma*psi_center => final PDE mismatch
    # PDE => gamma psi - (1+∇^2)^2 psi - psi^3
    #    => gamma*psi_center - op_psi - (psi_center^3)

    psi_cubed = tf.pow(psi_center, 3.0)
    residual_2d = gamma*psi_center - op_psi - psi_cubed

    # Flatten to shape (batch, Nx*Ny)
    residual_flat = tf.reshape(residual_2d, [batch_size, Nx*Ny])

    return residual_flat

@tf.function
def train_step_pddlvm(batch_size=16):
    """
    Single PDDLVM training step.

    1) Samples a batch of latent z from a standard normal prior
    2) ForwardNet => reparameterized psi_samp
    3) PDE residual => log-likelihood of residual=0 => logp_r
    4) InverseNet => distribution p_beta(z|psi_samp) => logp_beta
    5) Summation with prior => yields the ELBO => minimized by gradient-based updates

    Parameters
    ----------
    batch_size : int
        Size of the latent sample batch

    Returns
    -------
    tf.Tensor
        Scalar float for the current ELBO
    """
    with tf.GradientTape() as tape:
        z_samp = prior_z(batch_size)
        mean_psi, logvar_psi = ForwardNet.Map([z_samp])
        eps = tf.random.normal(shape=tf.shape(mean_psi))
        psi_samp = mean_psi + tf.exp(0.5*logvar_psi)*eps

        r_samp = pde_residual_tf(psi_samp)
        M = tf.shape(r_samp)[1]
        logp_r = (
            -0.5 * tf.reduce_sum(r_samp**2, axis=1) / (sigma_r**2)
            - 0.5 * tf.cast(M, tf.float32) * tf.math.log(2.0 * np.pi * (sigma_r**2))
        )

        mean_b, logvar_b = InverseNet.Map([psi_samp])
        logp_beta = log_diag_gaussian(z_samp, mean_b, logvar_b)
        logp_z = log_prior_z(z_samp)

        elbo_samp = logp_r + logp_beta + logp_z
        elbo = tf.reduce_mean(elbo_samp)
        loss = -elbo

    grads = tape.gradient(loss, ForwardNet.trainable_variables + InverseNet.trainable_variables)
    optimizer.apply_gradients(zip(grads, ForwardNet.trainable_variables + InverseNet.trainable_variables))
    return elbo


##########################
# 4) Main training
##########################
nIter = 10_000
elbo_log = []
for step in range(nIter):
    e = train_step_pddlvm(batch_size=16)
    if step % 100 == 0:
        print(f"Iter={step}, ELBO={e.numpy():.4f}")
        elbo_log.append(e.numpy())

plt.figure()
plt.plot(np.arange(len(elbo_log))*100, elbo_log, 'o-')
plt.xlabel("Iteration")
plt.ylabel("log-ELBO")
plt.title(f"Swift-Hohenberg PDE with PDDLVM - Demo (time snapshot={t_snap})")
save_or_show(plt.gcf(), "artifacts/sh_pddlvm_train.png", show=SHOW)

##########################
# 5) Obtain numeric reference from sh_spectral
print("WARNING: steady PDDLVM residual vs transient spectral snapshot -- "
      "different physics; comparison is a sanity check only.")
##########################
psi_final, psi_hist, psi_at_snap = swift_hohenberg_2d_etdrk2(
    Nx=Nx,
    Ny=Ny,
    Lx=Lx,
    Ly=Ly,
    gamma=gamma,
    dt=dt_spec,
    nsteps=nsteps,
    noise_amp=0.01,
    dealias=True,
    t_snap=t_snap
)
"""
psi_final : ndarray
    The final real-space field at time = nsteps*dt_spec if no snapshot was found
psi_hist : list of 2D arrays
    A history of psi fields
psi_at_snap : ndarray or None
    The real-space field at t_snap if within the run time
"""

if psi_at_snap is None:
    print(f"No valid snapshot found at t={t_snap}, possibly t_snap> nsteps*dt_spec")
    psi_gt = psi_final
    used_time = nsteps * dt_spec
else:
    psi_gt = psi_at_snap
    used_time = t_snap

##########################
# 6) Compare final results
##########################
mean_psi_test, logv_psi_test = ForwardNet.Map([tf.zeros((1,z_dim), dtype=tf.float32)])
psi_pred_np = mean_psi_test[0].numpy().reshape((Ny, Nx))

fig, ax = plt.subplots(ncols=2, figsize=(12, 5))

im0 = ax[0].imshow(psi_gt, origin='lower', cmap='RdBu', extent=[0, Lx, 0, Ly])
ax[0].set_title(f"Swift-Hohenberg ground truth at t={used_time}")
plt.colorbar(im0, ax=ax[0], label="psi(x,y)")

im1 = ax[1].imshow(psi_pred_np, origin='lower', cmap='RdBu', extent=[0, Lx, 0, Ly])
ax[1].set_title(f"PDDLVM mean psi at t={used_time} (pretend PDE residual)")
plt.colorbar(im1, ax=ax[1], label="psi(x,y)")

plt.tight_layout()
save_or_show(plt.gcf(), "artifacts/sh_pddlvm_compare.png", show=SHOW)