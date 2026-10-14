"""Batch runner: parametric cylinder-flow diagnostics (periodic box + penalty).

Deterministic diagnostic harness for the mode-selection question.  One
process = one run = one tag.  Results print as a single ``RESULT <tag> ...``
line and save to ``.tmp/e9_<tag>.{json,t,cd,cl}.npy``.  Designed for parallel
launch via ``tools/e9_batch_launch.sh`` (one core per run); no randomness, so
reruns with identical arguments are bit-for-bit reproducible.

Protocol (locked):
- u0 = 1 everywhere; single k=1 (box fundamental) transverse perturbation
  ``v0 = amp * sin(2 pi x / L) * exp(-((x-cx)^2 + (y-cy)^2) / 0.2)``
- smooth tanh mask, Brinkman penalty ``eps=0.01, width=0.0625`` (physical
  units; cylinder diameter D=1 centered in the box)
- driving: mean pinning ``U_target=1`` (Re = U D / nu exact) or constant
  body force ``--force``; Cd/Cl from the IBM penalty formulas
- dt chosen for CFL = U dt / dx ~ 0.5 at n=512: 0.008 for L=8, 0.016 otherwise
- f_peak: Hann-windowed, zero-padded rfft of Cl over ``t >= max(60, T/4)``

Usage:
    python tools/e9_batch.py TAG L RE T [N] [--force] [--amp AMP]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from pddlvm.numerics import ns2d_spectral as S
from pddlvm.numerics import obstacles as O


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("L", type=float)
    ap.add_argument("re", type=float)
    ap.add_argument("T", type=float)
    ap.add_argument("N", type=int, nargs="?", default=512)
    ap.add_argument("--force", action="store_true",
                    help="constant-force driving instead of mean pinning")
    ap.add_argument("--amp", type=float, default=0.05,
                    help="initial perturbation amplitude")
    ap.add_argument("--snap", type=int, default=0,
                    help="save u,v snapshots every N steps (float32, "
                         ".tmp/e9_<tag>_snaps.npy); 0 = off")
    ap.add_argument("--dt", type=float, default=None,
                    help="override dt (default: 0.008 for L<=8 else 0.016; "
                         "use 0.008 for Re>=200 or force-driving to stay "
                         "inside the advective stability region)")
    args = ap.parse_args()

    L, re, T, n = args.L, args.re, args.T, args.N
    nu = 1.0 / re
    dt = args.dt if args.dt is not None else (0.008 if L <= 8.0 else 0.016)
    n_steps = int(round(T / dt))
    T = n_steps * dt  # exact T actually integrated
    cx = cy = L / 2
    eps, width = 0.01, 0.0625
    x = (np.arange(n) + 0.5) * L / n
    X, Y = np.meshgrid(x, x)
    u0 = np.full((n, n), 1.0)
    v0 = args.amp * np.sin(2 * np.pi * X / L) * np.exp(
        -((X - cx) ** 2 + (Y - cy) ** 2) / 0.2)
    lam, mask = O.cylinder_field(n, n, L, L, cx, cy, 1.0, width, eps)
    u, v, snapshots, probes = S.spectral_solve_2d(
        u0, v0, nu=nu, dt=dt, n_steps=n_steps, Lx=L, Ly=L,
        U_target=None if args.force else 1.0,
        force_x=1.0 if args.force else None,
        penalty=lam, diameter=1.0, probe_every=5,
        snapshot_every=(args.snap if args.snap > 0 else None))
    t = np.array([p[0] * dt for p in probes])
    cd = np.array([p[1] for p in probes])
    cl = np.array([p[2] for p in probes])

    lines = []
    for t0, t1 in [(0.25 * T, 0.5 * T), (0.5 * T, 0.75 * T), (0.75 * T, T)]:
        sel = (t >= t0) & (t < t1)
        lines.append(f"t {t0:.0f}-{t1:.0f}: Cd={cd[sel].mean():.3f} "
                     f"Cl_rms={np.sqrt(np.mean((cl[sel] - cl[sel].mean()) ** 2)):.4f}")
    sel = t >= max(60.0, 0.25 * T)
    clw = (cl[sel] - cl[sel].mean()) * np.hanning(sel.sum())
    spec = np.abs(np.fft.rfft(clw, n=1 << 20)) ** 2
    freq = np.fft.rfftfreq(1 << 20, d=dt)
    f_peak = freq[np.argmax(spec)]
    cd_late = cd[sel].mean()
    for ln in lines:
        print(ln)
    print(f"RESULT {args.tag}: f_peak={f_peak:.4f} (St={f_peak:.4f}) "
          f"Cd_late={cd_late:.3f} [L={L} Re={re} T={T} n={n} "
          f"drive={'force' if args.force else 'pin'} amp={args.amp}]")

    out = {"tag": args.tag, "L": L, "re": re, "T": T, "n": n, "dt": dt,
           "n_steps": n_steps, "drive": "force" if args.force else "pin",
           "amp": args.amp, "eps": eps, "width": width,
           "snap_every": args.snap, "f_peak": float(f_peak),
           "cd_late": float(cd_late), "windows": lines}
    os.makedirs(".tmp", exist_ok=True)
    with open(f".tmp/e9_{args.tag}.json", "w") as f:
        json.dump(out, f, indent=2)
    np.save(f".tmp/e9_{args.tag}_t.npy", t)
    np.save(f".tmp/e9_{args.tag}_cd.npy", cd)
    np.save(f".tmp/e9_{args.tag}_cl.npy", cl)
    if snapshots:
        snaps = np.stack([[np.asarray(s[1][0], dtype=np.float32),
                           np.asarray(s[1][1], dtype=np.float32)]
                          for s in snapshots])
        np.save(f".tmp/e9_{args.tag}_snaps.npy", snaps)
        print(f"SNAPS {args.tag}: {snaps.shape} float32, "
              f"{snaps.nbytes / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
