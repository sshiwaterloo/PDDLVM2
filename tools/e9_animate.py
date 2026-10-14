"""Snapshot renderer: u,v snapshots -> vortex-shedding animation GIF.

Reads the outputs of ``tools/e9_batch.py --snap`` (``.tmp/e9_<tag>_snaps.npy``
plus the probe series and JSON) and renders a four-panel GIF:
vorticity, |u| (with the solid cylinder marked), and the Cd / Cl traces with a
progress marker.  Pure local post-processing (numpy + matplotlib), no cloud
compute.

Usage:
    python tools/e9_animate.py TAG [--frames N] [--fps F] [--out PATH]
Output: artifacts/<tag>_anim.gif (default)
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pddlvm.numerics import ns2d_spectral as S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--frames", type=int, default=240,
                    help="frames in the GIF (evenly sampled from snapshots)")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    tag = args.tag

    with open(f".tmp/e9_{tag}.json") as f:
        meta = json.load(f)
    L, n = meta["L"], meta["n"]
    # snapshot time = snapshot_index * snap_every * dt (probe series is 5-step
    # sampling, a different clock); fall back to inference for older JSONs
    snap_every = meta.get("snap_every") or int(round(meta["n_steps"] / max(
        np.load(f".tmp/e9_{tag}_snaps.npy", mmap_mode="r").shape[0], 1)))
    snaps = np.load(f".tmp/e9_{tag}_snaps.npy")  # [nf, 2, n, n] float32
    t = np.load(f".tmp/e9_{tag}_t.npy")
    cd = np.load(f".tmp/e9_{tag}_cd.npy")
    cl = np.load(f".tmp/e9_{tag}_cl.npy")
    nf = snaps.shape[0]
    snap_t = np.arange(nf) * snap_every * meta["dt"]
    print(f"{tag}: {nf} snapshots, L={L}, n={n}, "
          f"t_end={snap_t[-1]:.1f}, snap_every={snap_every}")

    idx = np.linspace(0, nf - 1, min(args.frames, nf)).astype(int)
    idx = np.unique(idx)
    w_all = []
    for i in idx:
        u, v = snaps[i, 0], snaps[i, 1]
        w_all.append(S.vorticity(u, v, Lx=L, Ly=L))
    w_max = max(float(np.abs(w).max()) for w in w_all)
    vmax = max(float(np.sqrt(snaps[idx, 0] ** 2 + snaps[idx, 1] ** 2).max()), 1e-6)

    fig, axs = plt.subplots(2, 2, figsize=(11.5, 9))
    fig.suptitle(f"{tag}: vorticity / speed / force traces "
                 f"(L={L}, Re={meta['re']}, T={meta['T']})", fontsize=11)
    x = np.linspace(0, L, n)
    theta = np.linspace(0, 2 * np.pi, 200)
    cx = cy = L / 2
    r = 0.5

    ims = []
    for k, i in enumerate(idx):
        frame = []
        im = axs[0, 0].imshow(w_all[k].T, origin="lower",
                              extent=[0, L, 0, L], cmap="RdBu_r",
                              vmin=-w_max, vmax=w_max)
        frame.append(im)
        frame.append(axs[0, 0].plot(cx + r * np.cos(theta),
                                    cy + r * np.sin(theta), "k-", lw=1)[0])
        axs[0, 0].set_title("vorticity")
        axs[0, 0].set_xlabel("x"); axs[0, 0].set_ylabel("y")

        speed = np.sqrt(snaps[i, 0] ** 2 + snaps[i, 1] ** 2)
        im2 = axs[0, 1].imshow(speed.T, origin="lower",
                               extent=[0, L, 0, L], cmap="viridis",
                               vmin=0, vmax=vmax)
        frame.append(im2)
        frame.append(axs[0, 1].plot(cx + r * np.cos(theta),
                                    cy + r * np.sin(theta), "w-", lw=1)[0])
        axs[0, 1].set_title("|u| (solid region should be ~0)")
        axs[0, 1].set_xlabel("x"); axs[0, 1].set_ylabel("y")

        ti = snap_t[i]
        (ln1,) = axs[1, 0].plot(t, cd, "C0", lw=1)
        (ln2,) = axs[1, 0].plot([ti, ti], [cd.min(), cd.max()], "r--", lw=1)
        axs[1, 0].set_title(f"Cd (t={ti:.1f})")
        axs[1, 0].set_xlabel("t"); axs[1, 0].set_ylabel("Cd")
        axs[1, 0].set_xlim(0, t[-1]); axs[1, 0].set_ylim(cd.min(), cd.max())

        (ln3,) = axs[1, 1].plot(t, cl, "C1", lw=1)
        (ln4,) = axs[1, 1].plot([ti, ti], [cl.min(), cl.max()], "r--", lw=1)
        axs[1, 1].set_title("Cl")
        axs[1, 1].set_xlabel("t"); axs[1, 1].set_ylabel("Cl")
        axs[1, 1].set_xlim(0, t[-1]); axs[1, 1].set_ylim(cl.min(), cl.max())

        frame.extend([ln1, ln2, ln3, ln4])
        ims.append(frame)
        if k == 0:
            fig.colorbar(im, ax=axs[0, 0], fraction=0.046, pad=0.04)
            fig.colorbar(im2, ax=axs[0, 1], fraction=0.046, pad=0.04)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = args.out or f"artifacts/{tag}_anim.gif"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    from matplotlib.animation import ArtistAnimation, PillowWriter
    ani = ArtistAnimation(fig, ims, interval=1000 / args.fps, blit=False)
    ani.save(out, writer=PillowWriter(fps=args.fps), dpi=110)
    plt.close(fig)
    print(f"GIF written: {out} ({len(idx)} frames @ {args.fps} fps)")


if __name__ == "__main__":
    main()
