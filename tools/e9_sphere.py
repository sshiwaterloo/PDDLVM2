"""Sphere driver: flow past a sphere in a periodic box (Brinkman penalty).

Deterministic, batch-style diagnostics for the 3D obstacle solver
(``pddlvm.numerics.ns3d_spectral.spectral_solve_obstacle``).  One process =
one run = one tag; results print as a ``RESULT <tag> ...`` line and save to
``.tmp/e9s_<tag>.{json,t,cd,cly,clz}.npy``.

Protocol (matches the 2D cylinder batch): u0 = 1 everywhere, single box-
fundamental transverse perturbation; smooth tanh mask with Brinkman penalty
(eps=0.01, width=0.25 physical units); mean-flow pinning U_target=1 (Re exact)
or constant force; IBM probes with the sphere frontal-area convention
Cd = 8 Fx / (pi D^2 U^2).  Literature values (Re=300: St~0.136, Cd~0.66) are
the unbounded-flow REFERENCE context, not pass/fail.

Usage:
    python tools/e9_sphere.py TAG N L RE T DT [--force] [--amp A]
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
    args = ap.parse_args()

    n, L, re, T, dt = args.N, args.L, args.re, args.T, args.dt
    nu = 1.0 / re
    n_steps = int(round(T / dt))
    T = n_steps * dt
    cx = cy = cz = L / 2
    x = (np.arange(n) + 0.5) * L / n
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    u0 = np.full((n, n, n), 1.0)
    v0 = args.amp * np.sin(2 * np.pi * X / L) * np.exp(
        -((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2) / 0.2)
    w0 = args.amp * np.sin(2 * np.pi * Y / L) * np.exp(
        -((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2) / 0.2)
    lam, mask = O.sphere_field(n, n, n, L, L, L, cx, cy, cz, 1.0,
                               args.width, args.eps)
    u, v, w, _, probes = S.spectral_solve_obstacle(
        u0, v0, w0, nu=nu, dt=dt, n_steps=n_steps, L=L,
        U_target=None if args.force else 1.0,
        force_x=1.0 if args.force else None,
        penalty=lam, diameter=1.0, probe_every=5)
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
           "windows": lines}
    os.makedirs(".tmp", exist_ok=True)
    with open(f".tmp/e9s_{args.tag}.json", "w") as f:
        json.dump(out, f, indent=2)
    np.save(f".tmp/e9s_{args.tag}_t.npy", t)
    np.save(f".tmp/e9s_{args.tag}_cd.npy", cd)
    np.save(f".tmp/e9s_{args.tag}_cly.npy", cly)
    np.save(f".tmp/e9s_{args.tag}_clz.npy", clz)


if __name__ == "__main__":
    main()
