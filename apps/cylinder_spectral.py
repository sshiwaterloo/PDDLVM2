"""Flow past a circular cylinder in a periodic box (spectral + IBM).

Drives the flow by pinning the mean velocity to U=1 (exact Re), reads drag and
lift off the Brinkman penalty, and reports St / Cd / Cl' with the literature values as unbounded-flow reference:

    Re=100: St ~ 0.165, Cd ~ 1.34, Cl' ~ 0.33   (unbounded flow)
    Re=300: St ~ 0.21,  Cd ~ 1.35, Cl' ~ 0.66

Usage:
    python apps/cylinder_spectral.py --re 100            # full validation run
    python apps/cylinder_spectral.py --re 100 --quick    # short pipeline check
    python apps/cylinder_spectral.py --re 300 --show     # pop figures

Outputs under artifacts/cylinder_<re>/: Cd/Cl time series, vorticity
snapshots, vortex-shedding animation, metrics.json with the reference comparison.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from pddlvm.numerics import ns2d_spectral as S  # noqa: E402
from pddlvm.numerics import obstacles as O  # noqa: E402
from pddlvm.plotting import save_or_show  # noqa: E402

LITERATURE = {
    100: {"St": 0.165, "Cd": 1.34, "Cl": 0.33},
    300: {"St": 0.21, "Cd": 1.35, "Cl": 0.66},
}


def run_cylinder(re, n, dt, n_steps, eps, width, outdir, show=False):
    L = 8.0                      # domain side, D = 1 -> blockage 12.5%
    nu = 1.0 / re
    U = 1.0
    cx = cy = L / 2
    lam, mask = O.cylinder_field(n, n, L, L, cx, cy, 1.0, width, eps)

    # initial condition: uniform flow + tiny asymmetric nudge to seed shedding
    x = (np.arange(n) + 0.5) * L / n
    X, Y = np.meshgrid(x, x)
    u0 = np.full((n, n), U)
    v0 = 0.05 * np.sin(2 * np.pi * X / L) * np.exp(
        -((X - cx) ** 2 + (Y - cy) ** 2) / 0.2)

    probe_every = max(1, int(round(0.08 / dt)))
    snap_every = max(1, n_steps // 40)
    print(f"[cylinder] Re={re} n={n} dt={dt} steps={n_steps} "
          f"eps={eps} width={width}")
    u, v, snaps, probes = S.spectral_solve_2d(
        u0, v0, nu=nu, dt=dt, n_steps=n_steps, Lx=L, Ly=L,
        U_target=U, penalty=lam, diameter=1.0,
        snapshot_every=snap_every, probe_every=probe_every)

    t = np.array([p[0] * dt for p in probes])
    cd = np.array([p[1] for p in probes])
    cl = np.array([p[2] for p in probes])
    os.makedirs(outdir, exist_ok=True)

    # --- statistics over the developed (second-half) phase ----------------
    half = len(t) // 2
    cd_mean = float(np.mean(cd[half:]))
    cl_rms = float(np.sqrt(np.mean((cl[half:] - np.mean(cl[half:])) ** 2)))
    # Strouhal: FFT peak of Cl on the oscillatory phase (Hann window)
    cl_w = (cl - np.mean(cl)) * np.hanning(len(cl))
    nfft = 1 << 18
    spec = np.abs(np.fft.rfft(cl_w, n=nfft)) ** 2
    freq = np.fft.rfftfreq(nfft, d=dt)
    f_peak = float(freq[np.argmax(spec)])
    st = f_peak * 1.0 / U

    # --- literature comparison (informational; NOT a pass/fail verdict) ---
    # The periodic-box + penalty system has its own natural modes; literature
    # values are the *unbounded-flow* reference, reported for context only.
    lit = LITERATURE[re]
    reference = {}
    for key, name in [("St", "Strouhal"), ("Cd", "mean drag"), ("Cl", "lift rms")]:
        meas = {"St": st, "Cd": cd_mean, "Cl": cl_rms}[key]
        tol = 0.10 * lit[key] if key != "Cl" else 0.10
        reference[name] = {
            "measured": meas, "literature": lit[key],
            "rel_err": (meas - lit[key]) / lit[key] if key != "Cl"
            else (meas - lit[key]),
            "within_10pct": abs((meas - lit[key]) / lit[key]) <= 0.10
            if key != "Cl" else abs(meas - lit[key]) <= 0.10,
        }

    metrics = {
        "setup": {"re": re, "n": n, "dt": dt, "n_steps": n_steps,
                  "L": L, "D": 1.0, "eps": eps, "width": width,
                  "blockage": 1.0 / L},
        "measured": {"St": st, "Cd_mean": cd_mean, "Cl_rms": cl_rms,
                     "mean_u": float(u.mean())},
        "solid_max_u": float(np.abs(u[mask > 0.9]).max()),
        "reference": reference,
    }
    with open(os.path.join(outdir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    for name, val in metrics["reference"].items():
        print(f"  {name:10s} measured={val['measured']:.4f} "
              f"lit={val['literature']:.4f} rel_err={val['rel_err']:+.1%} "
              f"(unbounded-flow reference, not pass/fail)")
    print(f"  solid max|u| = {metrics['solid_max_u']:.2e}")

    # --- figures -----------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    w = S.vorticity(u, v, Lx=L, Ly=L)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(w, origin="lower", extent=[0, L, 0, L], cmap="RdBu_r")
    ax.add_patch(plt.Circle((cx, cy), 0.5, color="0.3", zorder=3))
    ax.set_title(f"Re={re} vorticity, t={n_steps * dt:.1f}")
    fig.colorbar(im)
    fig.tight_layout()
    save_or_show(fig, os.path.join(outdir, "vorticity_final.png"), show=show)
    plt.close(fig)

    fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    ax[0].plot(t, cd)
    ax[0].axhline(cd_mean, color="k", ls="--", lw=0.8)
    ax[0].axhline(lit["Cd"], color="g", ls=":", lw=0.8, label="literature")
    ax[0].set_ylabel("Cd")
    ax[0].legend()
    ax[1].plot(t, cl)
    ax[1].set_ylabel("Cl")
    ax[1].set_xlabel("t")
    fig.suptitle(f"Re={re} force coefficients")
    fig.tight_layout()
    save_or_show(fig, os.path.join(outdir, "forces.png"), show=show)
    plt.close(fig)

    frames = []
    for step, (us, vs) in snaps:
        ws = S.vorticity(us, vs, Lx=L, Ly=L)
        frames.append(ws)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.add_patch(plt.Circle((cx, cy), 0.5, color="0.3", zorder=3))
    im = ax.imshow(frames[0], origin="lower", extent=[0, L, 0, L],
                   cmap="RdBu_r", vmin=-2.5, vmax=2.5)
    ax.set_title(f"Re={re} vortex shedding")
    fig.colorbar(im)
    fig.tight_layout()

    from matplotlib.animation import FuncAnimation

    def update(i):
        im.set_data(frames[i])
        return (im,)

    anim = FuncAnimation(fig, update, frames=len(frames), interval=120)
    anim.save(os.path.join(outdir, "vortex_shedding.gif"), writer="pillow",
              fps=8)
    if show:
        plt.show()
    plt.close(fig)
    print(f"[cylinder] artifacts in {outdir}")
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--re", type=int, default=100, choices=[100, 300])
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--dt", type=float, default=0.008)
    ap.add_argument("--steps", type=int, default=12500)
    ap.add_argument("--eps", type=float, default=1e-2)
    ap.add_argument("--width", type=float, default=0.1)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.n = min(args.n, 128)
        args.steps = 2000
    outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
        __file__))), "artifacts", f"cylinder_{args.re}")
    run_cylinder(args.re, args.n, args.dt, args.steps, args.eps, args.width,
                 outdir, show=args.show)


if __name__ == "__main__":
    main()
