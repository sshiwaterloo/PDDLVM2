"""burgers_pddlvm.py — time-dependent PDDLVM for 1D Burgers.

The PDDLVM here is genuinely time-dependent: the forward network maps a
latent z to a full space-time field u(x, t) on a grid, and the physics
likelihood is the space-time residual

    u_t + u u_x - nu u_xx = 0          (interior points)
    u(x, 0) = -sin(pi x)               (initial condition)
    u(-1, t) = u(1, t) = 0             (boundary conditions)

built from pddlvm.numerics.fd.burgers_space_time_residual_tf (verified by
MMS in pddlvm.tests.test_numerics.TestBurgersSpaceTime).  The reference
solution is the transient theta-IMEX solver from pddlvm.numerics.fd, so the
final comparison is between two time-dependent solvers of the SAME physics
(the legacy steady-vs-transient inconsistency is gone).

Steps:
1) Build forward (z -> u(x,t)) and inverse (u -> z) networks.
2) Train with the prior-mode ELBO: physics residual likelihood + beta
   prior + standard normal prior on z - alpha entropy term.
3) Compare the learned mean field at t_snap against the transient FD
   reference; save figures instead of blocking on plt.show().
"""

import _bootstrap  # noqa: F401  (repo root on sys.path)
import sys

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

from pddlvm.NeuralNet import NeuralN
from pddlvm.numerics.fd import (
    burgers_space_time_residual_tf,
    solve_burgers_transient,
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

tfd = tfp.distributions


def build_problem(nx=51, nt=21, nu=0.01, t_max=1.0, z_dim=2, sigma_r=1e-2,
                  seed=2025):
    """Space-time Burgers problem: grids, residual operator, networks."""
    tf.random.set_seed(seed)
    np.random.seed(seed)

    x = np.linspace(-1.0, 1.0, nx)
    t = np.linspace(0.0, t_max, nt)
    x_tf = tf.constant(x, tf.float32)
    t_tf = tf.constant(t, tf.float32)

    def ic(xv):
        return -tf.math.sin(np.pi * xv)

    n_total = nx * nt
    forward = NeuralN("ForwardNet", np.array([z_dim, 128, 128, n_total]),
                      init_seed=seed)
    inverse = NeuralN("InverseNet", np.array([n_total, 64, 64, z_dim]),
                      init_seed=seed + 1)

    def physics_residual(u_grid):
        # u_grid: [batch, Nt, Nx]
        return tf.vectorized_map(
            lambda ug: burgers_space_time_residual_tf(
                ug, x_tf, t_tf, nu, ic_fn=ic, bc_left=0.0, bc_right=0.0),
            u_grid)

    return {
        "x": x, "t": t, "nu": nu, "z_dim": z_dim, "sigma_r": sigma_r,
        "n_total": n_total, "nx": nx, "nt": nt,
        "forward": forward, "inverse": inverse,
        "residual_fn": physics_residual,
    }


def make_forward(batch_size, p):
    """Prior-mode ELBO forward: sample z, generate u(x,t), evaluate physics."""

    def forward(batch):
        z = tf.random.normal([batch_size, p["z_dim"]])
        mean_u, logvar_u = p["forward"].Map([z])
        u_flat = reparameterize(mean_u, logvar_u)
        u_grid = tf.reshape(u_flat, [batch_size, p["nt"], p["nx"]])

        r = p["residual_fn"](u_grid)  # [batch, n_res]
        logp_r = gaussian_residual_log_likelihood(r, p["sigma_r"])

        mean_b, logvar_b = p["inverse"].Map([u_flat])
        logp_beta = diag_gaussian_log_prob(z, mean_b, logvar_b)

        logp_z = diag_gaussian_log_prob(z, tf.zeros_like(z),
                                        tf.zeros_like(z))
        logq_alpha = diag_gaussian_log_prob(u_flat, mean_u, logvar_u)

        return combine_terms(logp_r, logp_beta + logp_z, logq_alpha)

    return forward


def reference_solution(p, t_snap=0.5):
    """Transient FD reference for the same IC/BC/nu on x in [-1, 1]."""
    x = p["x"]
    u0 = -np.sin(np.pi * x)
    u0[0] = u0[-1] = 0.0
    dx = x[1] - x[0]
    dt = 0.9 * dx / max(1.0, np.max(np.abs(u0)))
    u_final, t_hist, u_hist = solve_burgers_transient(
        x, u0, nu=p["nu"], source=lambda xx, t: np.zeros_like(xx),
        t0=0.0, t1=p["t"][-1], dt=dt, bc_left=0.0, bc_right=0.0)
    u_snap = np.array([np.interp(t_snap, t_hist, [uu[i] for uu in u_hist])
                       for i in range(len(x))])
    return u_snap


def main():
    p = build_problem()
    steps = 1500 if "--quick" not in sys.argv else 150
    lr = 1e-3
    batch_size = 16
    variables = (p["forward"].trainable_variables
                 + p["inverse"].trainable_variables)
    optimizer = tf.keras.optimizers.Adam(lr)
    trainer = Trainer(make_forward(batch_size, p), variables, optimizer)
    dummy = build_dataset(tf.zeros((1, 1)), batch_size=1,
                          shuffle=False).repeat()
    history = trainer.fit(dummy, steps=steps, log_every=250)
    print(f"steps={steps}: nelbo {history['nelbo'][0]:.2f} -> "
          f"{history['nelbo'][-1]:.2f}")

    # learned mean field at the snapshot time (nearest grid level)
    t_snap = 0.5
    k_snap = int(np.argmin(np.abs(p["t"] - t_snap)))
    z0 = tf.zeros((1, p["z_dim"]))
    mean_u, _ = p["forward"].Map([z0])
    u_learned = mean_u.numpy().reshape(p["nt"], p["nx"])[k_snap, :]

    u_ref = reference_solution(p, t_snap=t_snap)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(p["x"], u_learned, "r-", label="PDDLVM mean u(x, t=0.5)")
    ax1.plot(p["x"], u_ref, "b--", label="theta-IMEX FD reference")
    ax1.plot(p["x"], -np.sin(np.pi * p["x"]), "k:", label="initial condition")
    ax1.set_title("same physics: time-dependent PDDLVM vs FD")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history["step"], history["nelbo"], label="NELBO")
    ax2.plot(history["step"], history["log_likelihood"],
             label="physics log-likelihood")
    ax2.set_title("training")
    ax2.legend()
    ax2.grid(True)
    save_or_show(fig, "artifacts/burgers_pddlvm_spacetime.png", show=SHOW)
    print(f"||u_learned - u_ref||_inf at t={t_snap}: "
          f"{np.max(np.abs(u_learned - u_ref)):.3e}")


if __name__ == "__main__":
    main()
