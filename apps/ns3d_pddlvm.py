"""ns3d_pddlvm.py — 3D Navier-Stokes PDDLVM demos.

Usage:
    python apps/ns3d_pddlvm.py --problem tgv [--quick]   # Taylor-Green
    python apps/ns3d_pddlvm.py --problem es  [--quick]   # Ethier-Steinman

The forward decoder (ConvFieldNet) maps a latent z to (u, v, w, p) on a
space-time grid; the physics likelihood is the NS residual (momentum x3 +
divergence + IC + BCs) from pddlvm.numerics.ns3d.

Ground truths:
- TGV: converged pseudospectral solver (pddlvm.numerics.ns3d_spectral),
  validated by energy conservation.  Comparison is phase-free: kinetic
  energy E(t) and the shell-averaged energy spectrum E(k).
- ES: the analytic Ethier-Steinman solution (Dirichlet data make the
  solution unique, so pointwise comparison is valid).

Outputs (artifacts/): loss/error PNGs and a slice-animation GIF.

The numerics underneath are fully verified (ES MMS, spectral energy
conservation to machine precision, see test_ns3d.py).
"""

import _bootstrap  # noqa: F401  (repo root on sys.path)

import os
import sys

import numpy as np
import tensorflow as tf

# Deterministic training (eliminates GPU reduce/concat non-determinism that
# otherwise swamps the seed-to-seed signal).
tf.config.experimental.enable_op_determinism()

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from pddlvm.NeuralNet import ConvFieldNet, NeuralN
from pddlvm.numerics.ns3d import ns3d_residual_tf
from pddlvm.numerics.ns3d_spectral import spectral_solve
from pddlvm.numerics.turbulence import energy_spectrum, kinetic_energy, tgv_initial
from pddlvm.numerics.mms import es3d_grid
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
if "--problem" in sys.argv:
    PROBLEM = sys.argv[sys.argv.index("--problem") + 1]
else:
    PROBLEM = "tgv"
assert PROBLEM in ("tgv", "es"), "--problem must be tgv or es"

N, NT = 16, 5
# Latent dimension: raised 2 -> 8 (the ES 0.82 stuck state was the Z_DIM=2
# capacity bottleneck) and made CLI-configurable via --z-dim.
Z_DIM = 8
if "--z-dim" in sys.argv:
    Z_DIM = int(sys.argv[sys.argv.index("--z-dim") + 1])
NU = 0.02 if PROBLEM == "tgv" else 0.1
T_MAX = 0.4 if PROBLEM == "tgv" else 0.2
L = 2.0 * np.pi
SIGMA_R = 0.1 if PROBLEM == "tgv" else 0.05


def es_tf(x, y, z, t):
    a, d = np.pi / 4.0, np.pi / 2.0
    e = tf.exp(-NU * d ** 2 * t)
    ex, ey, ez = tf.exp(a * x), tf.exp(a * y), tf.exp(a * z)
    u = -a * (ex * tf.sin(a * y + d * z) + ez * tf.cos(a * x + d * y)) * e
    v = -a * (ey * tf.sin(a * z + d * x) + ex * tf.cos(a * y + d * z)) * e
    w = -a * (ez * tf.sin(a * x + d * y) + ey * tf.cos(a * z + d * x)) * e
    p = -0.5 * a ** 2 * (
        tf.exp(2 * a * x) + tf.exp(2 * a * y) + tf.exp(2 * a * z)
        + 2 * tf.sin(a * x + d * y) * tf.cos(a * z + d * x)
        * tf.exp(a * (y + z))
        + 2 * tf.sin(a * y + d * z) * tf.cos(a * x + d * y)
        * tf.exp(a * (z + x))
        + 2 * tf.sin(a * z + d * x) * tf.cos(a * y + d * z)
        * tf.exp(a * (x + y))) * tf.exp(-2 * NU * d ** 2 * t)
    return u, v, w, p


def build_problem(seed=2025):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    if PROBLEM == "tgv":
        x_np = np.linspace(0.0, L, N, endpoint=False)
        x = tf.constant(x_np, tf.float32)
        t = tf.constant(np.linspace(0.0, T_MAX, NT), tf.float32)

        def ic_tf(xv, yv, zv):
            return (tf.sin(xv) * tf.cos(yv) * tf.cos(zv),
                    -tf.cos(xv) * tf.sin(yv) * tf.cos(zv),
                    tf.zeros_like(xv))

        mode, bc = "periodic", None
    else:
        x_np = np.linspace(-1.0, 1.0, N)
        x = tf.constant(x_np, tf.float32)
        t = tf.constant(np.linspace(0.0, T_MAX, NT), tf.float32)

        def ic_tf(xv, yv, zv):
            return es_tf(xv, yv, zv, 0.0)[:3]

        u_es, v_es, w_es, _ = es3d_grid(x_np, t.numpy(), NU)
        bc = (tf.constant(u_es, tf.float32), tf.constant(v_es, tf.float32),
              tf.constant(w_es, tf.float32))
        mode = "dirichlet"

    forward_net = ConvFieldNet("NS3D", z_dim=Z_DIM,
                               field_shape=(N, N, N), channels=4 * NT,
                               base_filters=16, init_seed=seed)
    forward_net.Map([tf.zeros((1, Z_DIM))])  # build -> trainable_variables
    inverse_net = NeuralN("Inv3D", np.array([N ** 3 * NT * 3, 64, Z_DIM]),
                          init_seed=seed + 1)
    return {"x": x, "t": t, "x_np": x_np, "mode": mode, "bc": bc,
            "ic": ic_tf, "forward": forward_net, "inverse": inverse_net}


def make_forward(batch, p, data_frac=0.0, lam=0.1):
    def forward(_):
        if data_frac > 0:
            # Data-assimilation: the latent is DEGENERATE for a single observed
            # field, so train DETERMINISTICALLY at z=0 with the mean field (no
            # reparameterize noise).  Sampling z~N(0,I) + reparameterizing adds
            # per-step noise the decoder must average out, which makes the
            # landscape harder to optimize.  Prediction already uses z=0/mean,
            # so this aligns training with inference.
            z = tf.zeros([batch, Z_DIM])
        else:
            z = tf.random.normal([batch, Z_DIM])
        mean_flat, logvar_flat = p["forward"].Map([z])   # [b, n, n, n, 4*nt]
        # channels encode the time snapshots: -> [b, nt, n, n, n, 4]
        mean_f = tf.transpose(
            tf.reshape(mean_flat, [batch, N, N, N, NT, 4]),
            [0, 4, 1, 2, 3, 5])
        if data_frac > 0:
            f = mean_f
        else:
            logvar_f = tf.transpose(
                tf.reshape(logvar_flat, [batch, N, N, N, NT, 4]),
                [0, 4, 1, 2, 3, 5])
            logvar_f = tf.clip_by_value(logvar_f, -10.0, 10.0)  # guard exp blowup
            f = reparameterize(mean_f, logvar_f)

        def one(uvwp):
            return ns3d_residual_tf(
                uvwp[..., 0], uvwp[..., 1], uvwp[..., 2], uvwp[..., 3],
                p["x"], p["t"], NU,
                mode=p["mode"], ic=p["ic"], bc=p["bc"])

        r = tf.vectorized_map(one, f)
        phys_loss = tf.reduce_mean(r ** 2)
        if data_frac > 0:
            # data-assimilation: fit observed GT velocity,
            # regularized by the physics residual.
            du = (f[..., 0] - p["obs_u"][None]) * p["obs_mask"][None, None]
            dv = (f[..., 1] - p["obs_v"][None]) * p["obs_mask"][None, None]
            dw = (f[..., 2] - p["obs_w"][None]) * p["obs_mask"][None, None]
            data_loss = tf.reduce_mean(du ** 2 + dv ** 2 + dw ** 2)
            total = data_loss + lam * phys_loss
        else:
            total = phys_loss
        return combine_terms(-total, 0.0, 0.0)
    return forward


def tgv_ground_truth():
    """Spectral solve on 2N, downsampled to the net's N grid."""
    u0, v0, w0 = tgv_initial(2 * N, L)
    n_steps = int(round(T_MAX / 0.01))
    u, v, w, snaps = spectral_solve(
        u0, v0, w0, nu=NU, dt=0.01, n_steps=n_steps,
        snapshot_every=max(n_steps // (NT - 1), 1))
    fields = []
    if snaps:
        for _, (uu, vv, ww) in snaps[:NT]:
            fields.append((uu[::2, ::2, ::2], vv[::2, ::2, ::2],
                           ww[::2, ::2, ::2]))
    else:
        fields.append((u[::2, ::2, ::2], v[::2, ::2, ::2],
                       w[::2, ::2, ::2]))
    while len(fields) < NT:
        fields.append(fields[-1])
    return fields


def es_ground_truth():
    u, v, w, _ = es3d_grid(np.linspace(-1.0, 1.0, N),
                           np.linspace(0.0, T_MAX, NT), NU)
    return list(zip(u, v, w))


def main():
    steps = 120 if "--quick" in sys.argv else 1200
    if "--steps" in sys.argv:
        steps = int(sys.argv[sys.argv.index("--steps") + 1])
    batch = 4
    data_frac = 0.0
    lam = 0.1
    if "--data-frac" in sys.argv:
        data_frac = float(sys.argv[sys.argv.index("--data-frac") + 1])
    if "--lam" in sys.argv:
        lam = float(sys.argv[sys.argv.index("--lam") + 1])

    p = build_problem()
    if data_frac > 0:
        gt_fields = (tgv_ground_truth() if PROBLEM == "tgv"
                     else es_ground_truth())
        u_gt = np.stack([f[0] for f in gt_fields]).astype(np.float32)
        v_gt = np.stack([f[1] for f in gt_fields]).astype(np.float32)
        w_gt = np.stack([f[2] for f in gt_fields]).astype(np.float32)
        rng = np.random.RandomState(2025)
        obs_mask = (rng.rand(N, N, N) < data_frac).astype(np.float32)
        p["obs_u"] = tf.constant(u_gt)
        p["obs_v"] = tf.constant(v_gt)
        p["obs_w"] = tf.constant(w_gt)
        p["obs_mask"] = tf.constant(obs_mask)

    trainer = Trainer(make_forward(batch, p, data_frac, lam),
                      p["forward"].trainable_variables,
                      tf.keras.optimizers.Adam(2e-3), jit=True)
    dummy = build_dataset(tf.zeros((1, 1)), batch_size=1,
                          shuffle=False).repeat()
    history = trainer.fit(dummy, steps=steps, log_every=steps)
    print(f"[{PROBLEM}] steps={steps} data_frac={data_frac} lam={lam}: "
          f"nelbo {history['nelbo'][0]:.2f} -> {history['nelbo'][-1]:.2f}")

    mean_flat, _ = p["forward"].Map([tf.zeros((1, Z_DIM))])
    learned = np.transpose(mean_flat.numpy().reshape(N, N, N, NT, 4),
                           (3, 0, 1, 2, 4))
    gt_fields = tgv_ground_truth() if PROBLEM == "tgv" else es_ground_truth()

    os.makedirs("artifacts", exist_ok=True)
    tag = f"artifacts/ns3d_{PROBLEM}"
    kz = N // 2

    if PROBLEM == "tgv":
        e_learned = [kinetic_energy(learned[k, ..., 0], learned[k, ..., 1],
                                    learned[k, ..., 2]) for k in range(NT)]
        e_gt = [kinetic_energy(*gt_fields[k]) for k in range(NT)]
        k_gt, s_gt = energy_spectrum(*gt_fields[-1], L=L)
        k_le, s_le = energy_spectrum(learned[-1, ..., 0],
                                     learned[-1, ..., 1],
                                     learned[-1, ..., 2], L=L)
        nk = min(len(s_gt), len(s_le))
        spec_err = float(np.sqrt(np.mean((s_le[1:nk] - s_gt[1:nk]) ** 2)))
        print(f"[tgv] spectrum L2 error: {spec_err:.3e}")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(p["t"].numpy(), e_learned, "o-", label="PDDLVM")
        axes[0].plot(p["t"].numpy(), e_gt, "s--", label="spectral GT")
        axes[0].set_xlabel("t"); axes[0].set_ylabel("E(t)"); axes[0].legend()
        axes[1].loglog(k_gt[1:], s_gt[1:], "s--", label="GT")
        axes[1].loglog(k_le[1:], s_le[1:], "o-", label="PDDLVM")
        axes[1].set_xlabel("k"); axes[1].set_ylabel("E(k)"); axes[1].legend()
        for ax in axes:
            ax.grid(True, which="both", alpha=0.3)
        save_or_show(fig, f"{tag}_energy_spectrum.png", show=SHOW)

        vmax = max(np.max(np.abs(learned[..., 0])),
                   np.max(np.abs(np.stack([f[0] for f in gt_fields]))))
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        frames = []
        for k in range(NT):
            art = [axes[0].imshow(learned[k, kz, ..., 0], origin="lower",
                                  cmap="RdBu_r", vmin=-vmax, vmax=vmax),
                   axes[1].imshow(gt_fields[k][0][kz], origin="lower",
                                  cmap="RdBu_r", vmin=-vmax, vmax=vmax)]
            axes[0].set_title(f"PDDLVM u_x slice, t={p['t'].numpy()[k]:.2f}")
            axes[1].set_title("GT u_x slice")
            frames.append(art)
        ani = animation.ArtistAnimation(fig, frames, interval=500,
                                        blit=False)
        ani.save(f"{tag}_animation.gif",
                 writer=animation.PillowWriter(fps=2))
        plt.close(fig)
    else:
        errs = [float(np.sqrt(np.mean((learned[k, ..., 0]
                                       - gt_fields[k][0]) ** 2)))
                for k in range(NT)]
        gt_stack = np.stack([f[0] for f in gt_fields])
        rel = float(np.sqrt(np.mean((learned[..., 0] - gt_stack) ** 2)
                            / np.mean(gt_stack ** 2)))
        print("[es] per-slice L2: " + "  ".join(f"{e:.3f}" for e in errs))
        print(f"[es] relative L2: {rel:.4f}")

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(p["t"].numpy(), errs, "o-")
        ax.set_xlabel("t"); ax.set_ylabel("L2 vs analytic ES")
        ax.grid(True)
        save_or_show(fig, f"{tag}_error.png", show=SHOW)

        vmax = max(np.max(np.abs(learned[..., 0])),
                   np.max(np.abs(gt_stack)))
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        frames = []
        for k in range(NT):
            art = [axes[0].imshow(learned[k, kz, ..., 0], origin="lower",
                                  cmap="RdBu_r", vmin=-vmax, vmax=vmax),
                   axes[1].imshow(gt_fields[k][0][kz], origin="lower",
                                  cmap="RdBu_r", vmin=-vmax, vmax=vmax),
                   axes[2].imshow(np.abs(learned[k, kz, ..., 0]
                                         - gt_fields[k][0][kz]),
                                  origin="lower", cmap="inferno")]
            axes[0].set_title(f"PDDLVM u, t={p['t'].numpy()[k]:.2f}")
            axes[1].set_title("GT u")
            axes[2].set_title("|error|")
            frames.append(art)
        ani = animation.ArtistAnimation(fig, frames, interval=500,
                                        blit=False)
        ani.save(f"{tag}_animation.gif",
                 writer=animation.PillowWriter(fps=2))
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["step"], history["nelbo"], label="NELBO")
    ax.legend(); ax.grid(True)
    save_or_show(fig, f"{tag}_loss.png", show=SHOW)
    print(f"[{PROBLEM}] artifacts under {tag}_* (animation: "
          f"{tag}_animation.gif)")


if __name__ == "__main__":
    main()
