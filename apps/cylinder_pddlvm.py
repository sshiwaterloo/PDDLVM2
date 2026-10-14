"""cylinder_pddlvm.py — obstacle-flow PDDLVM (2D cylinder, periodic box).

Physics-driven deep latent-variable model of flow past a circular cylinder
embedded in a periodic box via a Brinkman penalty (the same system
characterized with the spectral solver).  The decoder (``ConvFieldNet2D``)
maps a latent ``z`` to a space-time (u, v, p) field; the physics likelihood is
the obstacle NS residual from ``pddlvm.numerics.ns2d_obstacle`` (momentum with
the ``+ lam u`` Brinkman term + continuity + IC), so training minimizes the
residual (physics) with the latent prior/entropy terms of the legacy ELBO.

The mean-flow pinning that drives the flow (``u += U_target - mean(u)``,
``v -= mean(v)``) is applied to the decoded field exactly as the spectral
solver does, so the decoder only learns the fluctuating field and the
prediction always carries the exact target mean.

Ground truth is ``spectral_solve_2d`` on the same grid, sampled at the same
NT times; comparison is on (u, v) / vorticity (the solver eliminates p, so
pressure is an unobservable auxiliary the net uses to satisfy continuity).

LIMITATION: the residual is backward-Euler in time + central differences
in space, i.e. O(dt + dx^2) accurate, so an exact match to the fine
spectral ground truth is not expected; the demo's target is residual
reduction + qualitative structure.

Usage:
    python apps/cylinder_pddlvm.py --re 100 --quick      # pipeline check
    python apps/cylinder_pddlvm.py --re 100              # full modest demo
"""

import _bootstrap  # noqa: F401  (repo root on sys.path)

import argparse
import json
import os

import numpy as np
import tensorflow as tf

# Deterministic training (eliminates the run-to-run variance that otherwise
# swamps the rate signal): fixed seeds are not enough, GPU reduce/concat ops
# are non-deterministic by default.
tf.config.experimental.enable_op_determinism()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from pddlvm.NeuralNet import ConvFieldNet2D
from pddlvm.numerics.ns2d_obstacle import (
    ns2d_obstacle_residual,
    ns2d_obstacle_residual_tf,
)
from pddlvm.numerics import ns2d_spectral as S
from pddlvm.numerics import obstacles as O
from pddlvm.probabilistic import (
    Trainer,
    build_dataset,
    combine_terms,
    gaussian_residual_log_likelihood,
    reparameterize,
)
from pddlvm.plotting import save_or_show

# --------------------------------------------------------------------------
# Defaults: L=8, D=1 (blockage 12.5%), nu=1/Re, U=1.
# --------------------------------------------------------------------------
L = 8.0
D = 1.0
U_TARGET = 1.0
EPS = 1e-2
WIDTH = 0.1
SIGMA_R = 0.1
DT_FINE = 0.008  # spectral ground-truth integration step


def ic_np(n, re):
    """Initial condition: uniform flow + tiny asymmetric nudge."""
    cx = cy = L / 2
    x = (np.arange(n) + 0.5) * L / n
    X, Y = np.meshgrid(x, x)
    u0 = np.full((n, n), U_TARGET, dtype=np.float32)
    v0 = (0.05 * np.sin(2 * np.pi * X / L) * np.exp(
        -((X - cx) ** 2 + (Y - cy) ** 2) / 0.2)).astype(np.float32)
    return u0, v0


def block_weights(n, nt, nu, lam, x, t, ic, seed=0):
    """Calibrated block-diagonal weights for the physics residual.

    Uses ``separate_penalty`` so the Brinkman term is its own block.  The
    momentum / continuity / IC blocks are weighted by inverse block variance
    (relative calibration), while the PENALTY block gets a strong fixed weight:
    the solver drives the solid velocity to zero, so the penalty is a near-hard
    constraint and must NOT be down-weighted by its (large) random-field scale.
    """
    rng = np.random.RandomState(seed)
    u = rng.randn(nt, n, n)
    v = rng.randn(nt, n, n)
    p = rng.randn(nt, n, n)
    r = ns2d_obstacle_residual(u, v, p, x, t, nu, lam, ic=ic,
                               separate_penalty=True)
    n_mom = 2 * (nt - 1) * n * n
    n_pen = 2 * (nt - 1) * n * n
    n_cont = nt * n * n
    r_mom = r[:n_mom]
    r_pen = r[n_mom:n_mom + n_pen]
    r_cont = r[n_mom + n_pen:n_mom + n_pen + n_cont]
    r_ic = r[n_mom + n_pen + n_cont:]

    def inv_var(b):
        s = np.sqrt(np.mean(b ** 2)) + 1e-8
        return 1.0 / (s * s)

    w_pen = 10.0  # strong: enforce solid suppression (near-hard constraint)
    return [inv_var(r_mom), w_pen, inv_var(r_cont), inv_var(r_ic)]


def _block_phys_loss(r, n, nt, w):
    """Block-weighted physics loss for a [batch, M] residual.

    ``r`` has the ``separate_penalty`` layout [mom_x; mom_y; pen_x; pen_y;
    cont; ic].
    """
    n_mom = 2 * (nt - 1) * n * n
    n_pen = 2 * (nt - 1) * n * n
    n_cont = nt * n * n
    r_mom = r[:, :n_mom]
    r_pen = r[:, n_mom:n_mom + n_pen]
    r_cont = r[:, n_mom + n_pen:n_mom + n_pen + n_cont]
    r_ic = r[:, n_mom + n_pen + n_cont:]
    w_mom, w_pen, w_cont, w_ic = w
    return (w_mom * tf.reduce_mean(r_mom ** 2)
            + w_pen * tf.reduce_mean(r_pen ** 2)
            + w_cont * tf.reduce_mean(r_cont ** 2)
            + w_ic * tf.reduce_mean(r_ic ** 2))


def build_problem(n, nt, re, z_dim, dt_res, base_filters=16, seed=2025):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    nu = 1.0 / re
    t_max = dt_res * (nt - 1)
    x = tf.constant(np.linspace(0.0, L, n, endpoint=False), tf.float32)
    t = tf.constant(np.linspace(0.0, t_max, nt), tf.float32)
    lam, mask = O.cylinder_field(n, n, L, L, L / 2, L / 2, D, WIDTH, EPS)
    lam_tf = tf.constant(lam.astype(np.float32))
    u0, v0 = ic_np(n, re)
    ic = (u0, v0)

    forward_net = ConvFieldNet2D("Cyl2D", z_dim=z_dim, field_shape=(n, n),
                                 channels=3 * nt, base_filters=base_filters,
                                 init_seed=seed)
    forward_net.Map([tf.zeros((1, z_dim))])  # build -> trainable_variables
    return {"x": x, "t": t, "t_max": t_max, "nu": nu, "lam": lam_tf,
            "mask": mask, "ic": ic, "forward": forward_net}


def make_forward(batch, n, nt, z_dim, p, data_frac=0.0, lam_phys=0.1,
                 v_weight=1.0):
    def forward(_):
        if data_frac > 0:
            # Data-assimilation mode: the latent is DEGENERATE for a single
            # observed field (it collapses to the prior mean z=0), so train
            # DETERMINISTICALLY at z=0 with the mean field (no reparameterize
            # noise).  Sampling z~N(0,I) + reparameterizing adds per-step noise
            # the decoder must average out, which makes the landscape harder to
            # optimize.  Prediction already uses z=0/mean, so this also makes
            # training match inference.
            z = tf.zeros([batch, z_dim])
        else:
            z = tf.random.normal([batch, z_dim])
        mean_flat, logvar_flat = p["forward"].Map([z])  # [b, n, n, 3*nt]
        mean_f = tf.transpose(
            tf.reshape(mean_flat, [batch, n, n, nt, 3]), [0, 3, 1, 2, 4])
        if data_frac > 0:
            f = mean_f
        else:
            logvar_f = tf.transpose(
                tf.reshape(logvar_flat, [batch, n, n, nt, 3]),
                [0, 3, 1, 2, 4])
            logvar_f = tf.clip_by_value(logvar_f, -10.0, 10.0)  # guard exp blowup
            f = reparameterize(mean_f, logvar_f)

        # mean-flow pinning (exact, as the spectral solver does)
        u = f[..., 0] + (U_TARGET - tf.reduce_mean(f[..., 0], axis=[2, 3],
                                                   keepdims=True))
        v = f[..., 1] - tf.reduce_mean(f[..., 1], axis=[2, 3], keepdims=True)
        f = tf.stack([u, v, f[..., 2]], axis=-1)

        def one(uvp):
            return ns2d_obstacle_residual_tf(
                uvp[..., 0], uvp[..., 1], uvp[..., 2],
                p["x"], p["t"], p["nu"], p["lam"], ic=p["ic"],
                separate_penalty=("block_w" in p))

        r = tf.vectorized_map(one, f)
        if "block_w" in p:
            phys_loss = _block_phys_loss(r, n, nt, p["block_w"])
        else:
            phys_loss = tf.reduce_mean(r ** 2)
        if data_frac > 0:
            # data-assimilation mode: fit the observed GT points, regularized
            # by the physics residual (the latent + physics fill the unobserved
            # points -- this is the original PDDLVM "infer-from-data" usage).
            du = (u - p["obs_u"][None]) * p["obs_mask"][None, None]
            dv = (v - p["obs_v"][None]) * p["obs_mask"][None, None]
            # v_weight > 1 up-weights the transverse velocity in the data term
            # (the v amplitude is small relative to u, so up-weight it).
            data_loss = tf.reduce_mean(du ** 2 + v_weight * dv ** 2)
            total = data_loss + lam_phys * phys_loss
        else:
            total = phys_loss
        return combine_terms(-total, 0.0, 0.0)
    return forward


def ground_truth(n, nt, re, t_max):
    """Spectral solve on the same grid, NT snapshots at the residual's times."""
    nu = 1.0 / re
    lam, mask = O.cylinder_field(n, n, L, L, L / 2, L / 2, D, WIDTH, EPS)
    u0, v0 = ic_np(n, re)
    dt_res = t_max / (nt - 1)
    snap_every = max(1, int(round(dt_res / DT_FINE)))
    n_steps = (nt - 1) * snap_every  # last snapshot lands on t = t_max
    _, _, snaps, _ = S.spectral_solve_2d(
        u0, v0, nu=nu, dt=DT_FINE, n_steps=n_steps, Lx=L, Ly=L,
        U_target=U_TARGET, penalty=lam, diameter=D,
        snapshot_every=snap_every)
    fields = [(us, vs) for _, (us, vs) in snaps[:nt]]
    while len(fields) < nt:
        fields.append((fields[-1][0], fields[-1][1]))
    return fields, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--re", type=int, default=100)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--nt", type=int, default=16)
    ap.add_argument("--z-dim", type=int, default=4)
    ap.add_argument("--base-filters", type=int, default=16,
                    help="decoder base_filters (capacity knob for eps_amort)")
    ap.add_argument("--dt-res", type=float, default=0.04)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--data-frac", type=float, default=0.0,
                    help="fraction of GT points observed (0 = residual-only; "
                         ">0 = data-assimilation PDDLVM)")
    ap.add_argument("--lam", type=float, default=0.1,
                    help="physics residual weight in the data mode")
    ap.add_argument("--v-weight", type=float, default=1.0,
                    help="weight of the v (transverse velocity) term in the "
                         "data fit (up-weight the small-magnitude transverse velocity)")
    ap.add_argument("--block-gamma", action="store_true",
                    help="use calibrated block-diagonal weights (per-block "
                         "inverse variance) instead of a uniform residual norm")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.n = min(args.n, 32)
        args.nt = min(args.nt, 8)
        args.steps = 120

    n, nt, re, z_dim = args.n, args.nt, args.re, args.z_dim
    p = build_problem(n, nt, re, z_dim, args.dt_res, args.base_filters)
    if args.block_gamma:
        x_np = p["x"].numpy(); t_np = p["t"].numpy()
        lam_np = p["lam"].numpy()
        p["block_w"] = block_weights(n, nt, 1.0 / re, lam_np, x_np, t_np,
                                     p["ic"])
    if args.data_frac > 0:
        gt_fields, mask = ground_truth(n, nt, re, p["t_max"])
        u_gt_all = np.stack([f[0] for f in gt_fields]).astype(np.float32)
        v_gt_all = np.stack([f[1] for f in gt_fields]).astype(np.float32)
        rng = np.random.RandomState(2025)
        obs_mask = (rng.rand(n, n) < args.data_frac).astype(np.float32)
        p["obs_u"] = tf.constant(u_gt_all)
        p["obs_v"] = tf.constant(v_gt_all)
        p["obs_mask"] = tf.constant(obs_mask)
    print(f"[cylinder-pddlvm] Re={re} n={n} nt={nt} z_dim={z_dim} "
          f"dt_res={args.dt_res} t_max={p['t_max']:.3f} "
          f"steps={args.steps} batch={args.batch} lr={args.lr} "
          f"data_frac={args.data_frac} lam={args.lam}")

    trainer = Trainer(make_forward(args.batch, n, nt, z_dim, p,
                                   args.data_frac, args.lam,
                                   v_weight=args.v_weight),
                      p["forward"].trainable_variables,
                      tf.keras.optimizers.Adam(args.lr), jit=True)
    dummy = build_dataset(tf.zeros((1, 1)), batch_size=1,
                          shuffle=False).repeat()
    log_every = max(args.steps // 100, 1)
    history = trainer.fit(dummy, steps=args.steps, log_every=log_every)
    print(f"[cylinder-pddlvm] nelbo {history['nelbo'][0]:.3f} -> "
          f"{history['nelbo'][-1]:.3f}")

    # ---- MAP latent: optimize z to minimize the objective (data + physics).
    # Several restarts make the non-convex MAP search robust.
    def pinned_loss(z_in):
        mean_flat, _ = p["forward"].Map([z_in])
        mean_f = tf.transpose(
            tf.reshape(mean_flat, [1, n, n, nt, 3]), [0, 3, 1, 2, 4])
        uu = mean_f[..., 0] + (U_TARGET - tf.reduce_mean(
            mean_f[..., 0], axis=[2, 3], keepdims=True))
        vv = mean_f[..., 1] - tf.reduce_mean(
            mean_f[..., 1], axis=[2, 3], keepdims=True)
        ff = tf.stack([uu, vv, mean_f[..., 2]], axis=-1)
        r = ns2d_obstacle_residual_tf(
            ff[0, ..., 0], ff[0, ..., 1], ff[0, ..., 2],
            p["x"], p["t"], p["nu"], p["lam"], ic=p["ic"])
        phys_loss = tf.reduce_mean(r ** 2)
        if args.data_frac > 0:
            du = (uu - p["obs_u"][None]) * p["obs_mask"][None, None]
            dv = (vv - p["obs_v"][None]) * p["obs_mask"][None, None]
            return tf.reduce_mean(du ** 2 + dv ** 2) + args.lam * phys_loss
        return phys_loss

    if args.data_frac > 0:
        # data mode: the decoder fits the observed GT for z~N(0,I), so z=0
        # (the prior mean) is already a good reconstruction -- deterministic,
        # no MAP variance (the latent collapses toward the single data field).
        best_z = np.zeros((1, z_dim), dtype=np.float32)
    else:
        best_loss, best_z = None, None
        for restart in range(3):
            init = (tf.zeros([1, z_dim]) if restart == 0
                    else tf.random.normal([1, z_dim]))
            z_var = tf.Variable(init, dtype=tf.float32)
            opt_z = tf.keras.optimizers.Adam(0.05)
            for _ in range(2000):
                with tf.GradientTape() as tape:
                    loss = pinned_loss(z_var)
                g = tape.gradient(loss, [z_var])
                opt_z.apply_gradients(zip(g, [z_var]))
            lv = float(loss)
            if best_loss is None or lv < best_loss:
                best_loss, best_z = lv, z_var.numpy().copy()
    mean_flat, _ = p["forward"].Map([tf.constant(best_z, tf.float32)])
    learned = tf.transpose(
        tf.reshape(mean_flat, [n, n, nt, 3]), [2, 0, 1, 3]).numpy()
    u_pred = learned[..., 0] + (U_TARGET - learned[..., 0].mean(axis=(1, 2),
                                                                keepdims=True))
    v_pred = learned[..., 1] - learned[..., 1].mean(axis=(1, 2), keepdims=True)

    gt_fields, mask = ground_truth(n, nt, re, p["t_max"])
    cx = cy = L / 2

    # ---- metrics ---------------------------------------------------------
    u_gt = np.stack([f[0] for f in gt_fields])
    v_gt = np.stack([f[1] for f in gt_fields])
    rel_u = float(np.sqrt(np.mean((u_pred - u_gt) ** 2) / np.mean(u_gt ** 2)))
    rel_v = float(np.sqrt(np.mean((v_pred - v_gt) ** 2) / np.mean(v_gt ** 2)))
    solid_max = float(np.abs(u_pred[-1, mask > 0.9]).max())
    # solid-velocity of the spectral GT (reference for how well the penalty
    # is satisfied by the solver itself), measured on the last snapshot (the
    # t=0 level is pinned to the uniform IC, where the solid is at u=1).
    gt_solid_max = float(np.abs(u_gt[-1, mask > 0.9]).max())
    print(f"[cylinder-pddlvm] rel L2 u={rel_u:.3f} v={rel_v:.3f} "
          f"solid_max_u={solid_max:.2e} (GT solid={gt_solid_max:.2e})")

    os.makedirs("artifacts", exist_ok=True)
    tag = f"artifacts/cylinder_pddlvm_re{re}_n{n}_z{z_dim}_bf{args.base_filters}"
    if args.data_frac > 0:
        tag += f"_data{args.data_frac:g}_lam{args.lam:g}"
    if abs(args.v_weight - 1.0) > 1e-9:
        tag += f"_vw{args.v_weight:g}"
    if args.block_gamma:
        tag += "_bg"

    # ---- durable metrics + field dump ------------------------------------
    metrics = {
        "setup": {"re": re, "n": n, "nt": nt, "z_dim": z_dim,
                  "dt_res": args.dt_res, "t_max": float(p["t_max"]),
                  "steps": args.steps, "batch": args.batch, "lr": args.lr,
                  "L": L, "D": D, "eps": EPS, "width": WIDTH,
                  "sigma_r": SIGMA_R},
        "nelbo": {"init": float(history["nelbo"][0]),
                  "final": float(history["nelbo"][-1])},
        "error": {"rel_u": rel_u, "rel_v": rel_v,
                  "solid_max_u": solid_max, "gt_solid_max_u": gt_solid_max},
    }
    with open(f"{tag}_metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    np.save(f"{tag}_pred.npy", np.stack([u_pred, v_pred]))
    np.save(f"{tag}_gt.npy", np.stack([u_gt, v_gt]))

    # ---- intermediate training data (kept in full) -----------------------
    hist = {k: [float(x) for x in v]
            for k, v in history.items()
            if k in ("step", "log_likelihood", "log_prior", "log_q",
                     "elbo", "nelbo", "lr")}
    with open(f"{tag}_history.json", "w") as fh:
        json.dump(hist, fh)
    # MAP latent (the learned z*) + trained decoder weights
    np.save(f"{tag}_zmap.npy", best_z.astype(np.float32))
    tf.train.Checkpoint(forward=p["forward"]).save(f"{tag}_ckpt")

    # ---- loss curve ------------------------------------------------------
    nelbo_init = float(history["nelbo"][0])
    nelbo_final = float(history["nelbo"][-1])
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot(history["step"], history["nelbo"], label="NELBO")
    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("NELBO  =  -residual log-likelihood")
    ax.set_title(f"Re={re} physics residual: {nelbo_init:.2e} → {nelbo_final:.2e}"
                 f"  ({nelbo_init / max(nelbo_final, 1e-12):.0f}× reduction)")
    ax.grid(True, which="both", alpha=0.3)
    save_or_show(fig, f"{tag}_loss.png", show=args.show)
    plt.close(fig)

    # ---- vorticity comparison (PDDLVM vs spectral GT) --------------------
    k_last = nt - 1
    w_pred = S.vorticity(u_pred[k_last], v_pred[k_last], Lx=L, Ly=L)
    w_gt = S.vorticity(u_gt[k_last], v_gt[k_last], Lx=L, Ly=L)
    vmax = max(abs(w_gt).max(), abs(w_pred).max())
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    for ax, w, title in [(axes[0], w_gt, "spectral GT (reference)"),
                         (axes[1], w_pred, "PDDLVM (learned)"),
                         (axes[2], w_pred - w_gt, "difference")]:
        im = ax.imshow(w, origin="lower", extent=[0, L, 0, L], cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax)
        ax.add_patch(plt.Circle((cx, cy), 0.5, color="0.3", zorder=3))
        ax.set_title(title)
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01, shrink=0.85)
    cb.set_label("vorticity  ω")
    fig.suptitle(f"Re={re}, t={p['t_max']:.2f}: decoder reproduces the "
                 f"cylinder wake vortex sheets (rel u L2={rel_u:.2f})")
    save_or_show(fig, f"{tag}_vorticity.png", show=args.show)
    plt.close(fig)

    # ---- u-velocity slice comparison ------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    for ax, uu, title in [(axes[0], u_gt[k_last], "spectral GT u"),
                          (axes[1], u_pred[k_last], "PDDLVM u")]:
        im = ax.imshow(uu, origin="lower", extent=[0, L, 0, L], cmap="RdBu_r",
                       vmin=u_gt[k_last].min(), vmax=u_gt[k_last].max())
        ax.add_patch(plt.Circle((cx, cy), 0.5, color="0.3", zorder=3))
        ax.set_title(title)
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01, shrink=0.85)
    cb.set_label("streamwise velocity  u")
    fig.suptitle(f"Re={re}, t={p['t_max']:.2f}: solid interior damped to "
                 f"|u|≈{solid_max:.3f} (GT {gt_solid_max:.3f}); wake deficit "
                 f"behind the cylinder")
    save_or_show(fig, f"{tag}_u.png", show=args.show)
    plt.close(fig)

    # ---- vorticity animation (GT vs PDDLVM over the time window) ---------
    w_all = np.stack([S.vorticity(u_pred[k], v_pred[k], Lx=L, Ly=L)
                      for k in range(nt)])
    wg_all = np.stack([S.vorticity(u_gt[k], v_gt[k], Lx=L, Ly=L)
                       for k in range(nt)])
    vmax_anim = max(abs(wg_all).max(), abs(w_all).max())
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    axes[0].set_title("spectral GT")
    axes[1].set_title("PDDLVM (learned)")
    for ax in axes:
        ax.add_patch(plt.Circle((cx, cy), 0.5, color="0.3", zorder=3))
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    im0 = axes[0].imshow(wg_all[0], origin="lower", extent=[0, L, 0, L],
                         cmap="RdBu_r", vmin=-vmax_anim, vmax=vmax_anim)
    im1 = axes[1].imshow(w_all[0], origin="lower", extent=[0, L, 0, L],
                         cmap="RdBu_r", vmin=-vmax_anim, vmax=vmax_anim)
    cb = fig.colorbar(im1, ax=axes, fraction=0.02, pad=0.01, shrink=0.85)
    cb.set_label("vorticity  ω")
    ttl = fig.suptitle("")
    ts = p["t"].numpy()

    def update(k):
        im0.set_data(wg_all[k])
        im1.set_data(w_all[k])
        ttl.set_text(f"Re={re}, t={ts[k]:.3f}   (left = spectral GT, "
                     f"right = PDDLVM learned)")
        return (im0, im1, ttl)

    anim = animation.FuncAnimation(fig, update, frames=nt, interval=300,
                                   blit=False)
    anim.save(f"{tag}_vorticity.gif", writer=animation.PillowWriter(fps=3))
    plt.close(fig)

    print(f"[cylinder-pddlvm] artifacts under {tag}_*")


if __name__ == "__main__":
    main()
