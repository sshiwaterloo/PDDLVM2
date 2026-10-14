"""Pillar driver: 3D flow past a vertical circular cylinder (pillar).

Deterministic diagnostics + snapshot capture for genuine-3D rendering.  One
process = one run = one tag; results print as a ``RESULT <tag> ...`` line and
save to ``.tmp/e9p_<tag>.{json,t,cd,cly,clz}[,_snaps].npy``.

Protocol (matches e9_sphere.py): u0 = 1 everywhere; a transverse + spanwise
perturbation seeds the wake (and, because the perturbation varies along the
pillar axis, allows the cylinder-wake 3D transition to develop); smooth tanh
mask + Brinkman penalty (eps=0.01, width=0.25); mean-flow pinning U_target=1
(Re exact) or constant force; IBM probes with the cylinder frontal-area
convention ``Cd = 2 Fx / (D Lz U^2)`` (frontal area = D * Lz, the projected
face of the pillar).

``--snap STEP`` additionally saves, every ``STEP`` steps, two float32 scalar
fields per frame -- Q-criterion and |u| -- as ``.tmp/e9p_<tag>_snaps.npy``
of shape (n_frames, 2, n, n, n), the input for the PyVista/Plotly renderer.

Usage:
    python tools/e9_pillar.py TAG N L RE T DT [--force] [--amp A] [--snap STEP]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from pddlvm.numerics import ns3d_spectral as S
from pddlvm.numerics import obstacles as O


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("N", type=int)
    ap.add_argument("L", type=float)
    ap.add_argument("re", type=float)
    ap.add_argument("T", type=float)
    ap.add_argument("dt", type=float)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--amp", type=float, default=0.05)
    ap.add_argument("--eps", type=float, default=0.01)
    ap.add_argument("--width", type=float, default=0.25)
    ap.add_argument("--snap", type=int, default=0,
                    help="save Q-criterion + |u| every N steps (0 = off)")
    ap.add_argument("--vframes", type=int, default=6,
                    help="save full velocity (u,v,w) for N evenly-spaced "
                         "snapshots (streamline rendering; 0 = off)")
    args = ap.parse_args()

    n, L, re, T, dt = args.N, args.L, args.re, args.T, args.dt
    nu = 1.0 / re
    n_steps = int(round(T / dt))
    T = n_steps * dt
    cx = cy = L / 2
    # explicit axis convention matching ns3d_spectral._wave_numbers:
    # axis 0 = z (pillar axis), axis 1 = y, axis 2 = x (flow direction)
    zc = ((np.arange(n) + 0.5) * L / n)[:, None, None]
    yc = ((np.arange(n) + 0.5) * L / n)[None, :, None]
    xc = ((np.arange(n) + 0.5) * L / n)[None, None, :]
    g = np.exp(-((xc - cx) ** 2 + (yc - cy) ** 2) / 0.2)
    u0 = np.full((n, n, n), 1.0)
    v0 = args.amp * np.sin(2 * np.pi * zc / L) * g   # spanwise perturbation
    w0 = args.amp * np.sin(2 * np.pi * xc / L) * g   # transverse perturbation
    lam, mask = O.cylinder3d_field(n, n, n, L, L, L, cx, cy, 1.0,
                                   args.width, args.eps)
    u, v, w, snapshots, probes = S.spectral_solve_obstacle(
        u0, v0, w0, nu=nu, dt=dt, n_steps=n_steps, L=L,
        U_target=None if args.force else 1.0,
        force_x=1.0 if args.force else None,
        penalty=lam, diameter=1.0, frontal_area=1.0 * L,
        probe_every=5, snapshot_every=(args.snap if args.snap > 0 else None))
    t = np.array([p[0] * dt for p in probes])
    cd = np.array([p[1] for p in probes])
    cly = np.array([p[2] for p in probes])
    clz = np.array([p[3] for p in probes])

    lines = []
    for t0, t1 in [(0.25 * T, 0.5 * T), (0.5 * T, 0.75 * T), (0.75 * T, T)]:
        sel = (t >= t0) & (t < t1)
        lines.append(f"t {t0:.0f}-{t1:.0f}: Cd={cd[sel].mean():.3f} "
                     f"Cl_rms={np.sqrt(np.mean((cly[sel] - cly[sel].mean()) ** 2)):.4f}")

    def fpeak(sig):
        sel = t >= max(30.0, 0.25 * T)
        sw = (sig[sel] - sig[sel].mean()) * np.hanning(sel.sum())
        spec = np.abs(np.fft.rfft(sw, n=1 << 18)) ** 2
        freq = np.fft.rfftfreq(1 << 18, d=dt)
        return freq[np.argmax(spec)]

    f_y = fpeak(cly)
    f_z = fpeak(clz)
    sel = t >= max(30.0, 0.25 * T)
    cd_late = cd[sel].mean()
    solid = mask > 0.9
    solid_max = float(np.sqrt(u ** 2 + v ** 2 + w ** 2)[solid].max())
    for ln in lines:
        print(ln)
    print(f"RESULT {args.tag}: f_peak_y={f_y:.4f} (St={f_y:.4f}) "
          f"f_peak_z={f_z:.4f} Cd_late={cd_late:.3f} solid_max_u={solid_max:.2e} "
          f"[n={n} L={L} Re={re} T={T} dt={dt} "
          f"drive={'force' if args.force else 'pin'} amp={args.amp}]")

    out = {"tag": args.tag, "n": n, "L": L, "re": re, "T": T, "dt": dt,
           "n_steps": n_steps, "drive": "force" if args.force else "pin",
           "amp": args.amp, "eps": args.eps, "width": args.width,
           "f_peak_y": float(f_y), "f_peak_z": float(f_z),
           "cd_late": float(cd_late), "solid_max_u": solid_max,
           "snap": args.snap, "windows": lines}
    os.makedirs(".tmp", exist_ok=True)
    with open(f".tmp/e9p_{args.tag}.json", "w") as f:
        json.dump(out, f, indent=2)
    np.save(f".tmp/e9p_{args.tag}_t.npy", t)
    np.save(f".tmp/e9p_{args.tag}_cd.npy", cd)
    np.save(f".tmp/e9p_{args.tag}_cly.npy", cly)
    np.save(f".tmp/e9p_{args.tag}_clz.npy", clz)
    if snapshots:
        qs, umags = [], []
        for _, (uu, vv, ww) in snapshots:
            qs.append(S.q_criterion_3d(uu, vv, ww, L=L))
            umags.append(np.sqrt(uu ** 2 + vv ** 2 + ww ** 2))
        snaps = np.stack([np.asarray(qs, dtype=np.float32),
                          np.asarray(umags, dtype=np.float32)], axis=1)
        np.save(f".tmp/e9p_{args.tag}_snaps.npy", snaps)
        print(f"SNAPS {args.tag}: {snaps.shape} float32, "
              f"{snaps.nbytes / 1e6:.1f} MB")
        # full velocity (u,v,w) at a few instants for 3D streamline rendering
        if args.vframes:
            nsnap = len(snapshots)
            vidx = np.unique(np.linspace(0, nsnap - 1,
                                         min(args.vframes, nsnap)).astype(int))
            vel = np.stack([
                np.stack([np.asarray(snapshots[j][1][0], dtype=np.float32),
                          np.asarray(snapshots[j][1][1], dtype=np.float32),
                          np.asarray(snapshots[j][1][2], dtype=np.float32)])
                for j in vidx])
            vtimes = np.array([snapshots[j][0] * dt for j in vidx])
            np.save(f".tmp/e9p_{args.tag}_vel.npy", vel)
            np.save(f".tmp/e9p_{args.tag}_vel_t.npy", vtimes)
            print(f"VEL {args.tag}: {vel.shape} float32 at t={vtimes.round(1)}")


if __name__ == "__main__":
    main()
