"""heat3d_pddlvm.py — 3D time-dependent PDDLVM for the heat equation.

    u_t = nu Lap(u) + f   on [0,1]^3, homogeneous Dirichlet, u(x,y,z,0) = u0

The forward network maps a latent z to the full 4D space-time field
u(x, y, z, t) on a grid; the physics likelihood is the space-time residual
(backward Euler in time, 7-point Laplacian) plus the initial condition and
the six boundary faces.  The reference is the analytic manufactured
solution (pddlvm.numerics.mms.heat3d_mms_u), which is independent of both
the FD operator and the network.

Outputs (under artifacts/):
- heat3d_pddlvm_loss.png            training curves
- heat3d_pddlvm_error.png           L2 error vs time slices
- heat3d_pddlvm_animation.gif       side-by-side animation (learned /
                                    ground truth / |error|) over time

The pipeline itself is verified by the MMS tests in
pddlvm/tests/test_heat3d.py.
"""

import _bootstrap  # noqa: F401  (repo root on sys.path)
import sys

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from pddlvm.NeuralNet import NeuralN
from pddlvm.numerics.heat3d import heat3d_residual_tf
from pddlvm.numerics.mms import (
    _HEAT_L1,
    _HEAT_L2,
    heat3d_mms_u,
)
from pddlvm.probabilistic import (
    Trainer,
    build_dataset,
    combine_terms,
    diag_gaussian_log_prob,
    gaussian_residual_log_likelihood,
    reparameterize,
)
from pddlvm.plotting import save_or_show

SHOW = "--show" in sys.argv

NU = 0.05
N = 14          # grid points per axis
NT = 8          # time levels
T_MAX = 0.4
Z_DIM = 2


def source_tf(x, y, z, t):
    """f* = u*_t - nu Lap(u*) with TensorFlow ops (4D broadcastable)."""
    m1 = ((NU * _HEAT_L1 - 1.0) * tf.sin(np.pi * x) * tf.sin(2 * np.pi * y)
          * tf.sin(3 * np.pi * z) * tf.exp(-t))
    m2 = (0.5 * (NU * _HEAT_L2 - 2.0) * tf.sin(2 * np.pi * x)
          * tf.sin(np.pi * y) * tf.sin(2 * np.pi * z) * tf.exp(-2 * t))
    return m1 + m2


def ic_tf(x, y, z, t):
    del t
    return (tf.sin(np.pi * x) * tf.sin(2 * np.pi * y) * tf.sin(3 * np.pi * z)
            + 0.5 * tf.sin(2 * np.pi * x) * tf.sin(np.pi * y)
            * tf.sin(2 * np.pi * z))


def ground_truth():
    """Analytic manufactured solution on the full space-time grid."""
    x = np.linspace(0.0, 1.0, N)
    t = np.linspace(0.0, T_MAX, NT)
    xg = x[None, None, None, :]
    yg = x[None, None, :, None]
    zg = x[None, :, None, None]
    tg = t[:, None, None, None]
    return x, t, heat3d_mms_u(xg, yg, zg, tg)


def main():
    n_total = N ** 3 * NT
    x = tf.constant(np.linspace(0.0, 1.0, N), tf.float32)
    t = tf.constant(np.linspace(0.0, T_MAX, NT), tf.float32)
    steps = 100 if "--quick" in sys.argv else 2500
    batch_size = 8
    sigma_r = 5e-2
    seed = 2025
    tf.random.set_seed(seed)
    np.random.seed(seed)

    forward_net = NeuralN("Forward3D", np.array([Z_DIM, 128, 128, n_total]),
                          init_seed=seed)
    inverse_net = NeuralN("Inverse3D", np.array([n_total, 64, 64, Z_DIM]),
                          init_seed=seed + 1)

    def residual_batch(u_batch):
        return tf.vectorized_map(
            lambda ug: heat3d_residual_tf(
                ug, x, x, x, t, NU, ic_fn=ic_tf, source_fn=source_tf),
            u_batch)

    def forward(batch):
        z = tf.random.normal([batch_size, Z_DIM])
        mean_u, logvar_u = forward_net.Map([z])
        u_flat = reparameterize(mean_u, logvar_u)
        u_grid = tf.reshape(u_flat, [batch_size, NT, N, N, N])
        r = residual_batch(u_grid)
        logp_r = gaussian_residual_log_likelihood(r, sigma_r)
        mean_b, logvar_b = inverse_net.Map([u_flat])
        logp_beta = diag_gaussian_log_prob(z, mean_b, logvar_b)
        logp_z = diag_gaussian_log_prob(z, tf.zeros_like(z), tf.zeros_like(z))
        logq = diag_gaussian_log_prob(u_flat, mean_u, logvar_u)
        return combine_terms(logp_r, logp_beta + logp_z, logq)

    variables = (forward_net.trainable_variables
                 + inverse_net.trainable_variables)
    trainer = Trainer(forward, variables, tf.keras.optimizers.Adam(3e-3),
                      jit=True)
    dummy = build_dataset(tf.zeros((1, 1)), batch_size=1,
                          shuffle=False).repeat()
    history = trainer.fit(dummy, steps=steps, log_every=steps)
    print(f"steps={steps}: nelbo {history['nelbo'][0]:.2f} -> "
          f"{history['nelbo'][-1]:.2f}")

    # ---------- comparison against the analytic ground truth ----------
    x_np, t_np, u_gt = ground_truth()
    z0 = tf.zeros((1, Z_DIM))
    mean_u, _ = forward_net.Map([z0])
    u_learned = mean_u.numpy().reshape(NT, N, N, N)

    errs = []
    for k in range(NT):
        errs.append(float(np.sqrt(np.mean(
            (u_learned[k] - u_gt[k]) ** 2))))
    rel = float(np.sqrt(np.mean((u_learned - u_gt) ** 2)
                        / np.mean(u_gt ** 2)))
    print("per-slice L2 errors vs ground truth:",
          "  ".join(f"{e:.3f}" for e in errs))
    print(f"relative L2 (space-time) vs ground truth: {rel:.4f}")

    # error curve
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t_np, errs, "o-")
    ax.set_xlabel("t")
    ax.set_ylabel("L2 error vs analytic ground truth")
    ax.set_title("3D heat PDDLVM vs ground truth")
    ax.grid(True)
    save_or_show(fig, "artifacts/heat3d_pddlvm_error.png", show=SHOW)

    # training curves
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["step"], history["nelbo"], label="NELBO")
    ax.plot(history["step"], history["log_likelihood"],
            label="physics log-likelihood")
    ax.legend()
    ax.grid(True)
    save_or_show(fig, "artifacts/heat3d_pddlvm_loss.png", show=SHOW)

    # ---------- animation: learned vs ground truth vs |error| ----------
    kz = N // 2
    vmax = max(np.max(np.abs(u_gt)), np.max(np.abs(u_learned)))
    frames = []
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for k in range(NT):
        art = []
        for axx, arr, cmap in (
                (axes[0], u_learned[k, kz], "viridis"),
                (axes[1], u_gt[k, kz], "viridis"),
                (axes[2], np.abs(u_learned[k, kz] - u_gt[k, kz]),
                 "inferno")):
            im = axx.imshow(arr, origin="lower", cmap=cmap,
                            vmin=-vmax if cmap == "viridis" else 0.0,
                            vmax=vmax if cmap == "viridis" else None,
                            animated=False)
            art.append(im)
        axes[0].set_title(f"learned u (z={kz}/{N}), t={t_np[k]:.2f}")
        axes[1].set_title(f"ground truth u, t={t_np[k]:.2f}")
        axes[2].set_title(f"|error|, t={t_np[k]:.2f}")
        frames.append(art)
    for axx in axes:
        axx.set_xlabel("x")
        axx.set_ylabel("y")
    ani = animation.ArtistAnimation(fig, frames, interval=500, blit=False)
    import os
    os.makedirs("artifacts", exist_ok=True)
    ani.save("artifacts/heat3d_pddlvm_animation.gif",
             writer=animation.PillowWriter(fps=2))
    plt.close(fig)
    print("animation saved: artifacts/heat3d_pddlvm_animation.gif")


if __name__ == "__main__":
    main()
