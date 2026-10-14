"""Streamline renderer: 3D streamlines through the pillar wake.

Integrates the saved velocity snapshots (``.tmp/e9p_<tag>_vel.npy``, shape
(n_inst, 3, n, n, n)) into instantaneous 3D streamlines seeded upstream of the
pillar, and renders them (coloured by local speed) together with the pillar and
a flow arrow in a rotatable Plotly HTML -- showing *how the fluid actually
moves* (splits around the pillar, then recirculates/swirls in the wake).

Usage:
    python tools/e9_streamlines.py TAG [--ds 1] [--nseed 12] [--steps 400]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np


def _interp(vel, pts, L):
    """Trilinear interpolation of a (3,n,n,n) velocity field on a periodic,
    cell-centred grid, at an array of points pts (k,3).  Returns (k,3)."""
    n = vel.shape[1]
    s = pts / L * n - 0.5            # continuous grid index
    s = s % n                         # periodic wrap
    i0 = np.floor(s).astype(int)
    i1 = (i0 + 1) % n
    f = s - i0
    out = np.zeros_like(pts)
    for c in range(3):                # velocity component
        v = vel[c]
        for (wx, ix) in ((1 - f[:, 0], i0[:, 0]), (f[:, 0], i1[:, 0])):
            for (wy, iy) in ((1 - f[:, 1], i0[:, 1]), (f[:, 1], i1[:, 1])):
                for (wz, iz) in ((1 - f[:, 2], i0[:, 2]), (f[:, 2], i1[:, 2])):
                    out[:, c] += wx * wy * wz * v[iz, iy, ix]
    return out


def trace(vel, seeds, L, steps, dt=0.04):
    """Advect seed points through the velocity field (RK2), returning a list
    of (k, steps+1, 3) streamline point arrays (one per seed)."""
    lines = []
    for p0 in seeds:
        p = p0.copy()
        pts = [p.copy()]
        dead = False
        for _ in range(steps):
            k1 = _interp(vel, p[None, :], L)[0]
            k2 = _interp(vel, (p + 0.5 * dt * k1)[None, :], L)[0]
            p = p + dt * k2
            p = p % L
            r = np.hypot(p[0] - L / 2, p[1] - L / 2)
            if r < 0.5:               # hit the pillar
                dead = True
                break
            pts.append(p.copy())
        if dead and len(pts) < 10:    # dropped straight into the pillar
            continue
        lines.append(np.array(pts))
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--nseed", type=int, default=13)
    ap.add_argument("--steps", type=int, default=400)
    args = ap.parse_args()
    tag = args.tag

    import plotly.graph_objects as go

    with open(f".tmp/e9p_{tag}.json") as f:
        meta = json.load(f)
    L, n = meta["L"], meta["n"]
    vel = np.load(f".tmp/e9p_{tag}_vel.npy")       # (ni, 3, n, n, n)
    vtimes = np.load(f".tmp/e9p_{tag}_vel_t.npy")
    ni = vel.shape[0]

    # seeds: a vertical line upstream of the pillar, spanning its wake shadow
    ys = np.linspace(L / 2 - 1.0, L / 2 + 1.0, args.nseed)
    seeds = [np.array([0.4, y, L / 2]) for y in ys]

    th = np.linspace(0, 2 * np.pi, 40)
    zz = np.linspace(0, L, 16)
    TH, ZZ = np.meshgrid(th, zz)

    def pillar_trace():
        return go.Surface(x=L / 2 + 0.5 * np.cos(TH), y=L / 2 + 0.5 * np.sin(TH),
                          z=ZZ, showscale=False, opacity=0.95,
                          colorscale=[[0, "rgb(125,125,125)"],
                                      [1, "rgb(125,125,125)"]],
                          name="pillar", hoverinfo="skip")

    frames = []
    sliders = []
    for k in range(ni):
        lines = trace(vel[k], seeds, L, args.steps)
        tr = [pillar_trace()]
        for ln in lines:
            sp = _interp(vel[k], ln, L)
            speed = np.linalg.norm(sp, axis=1)
            tr.append(go.Scatter3d(
                x=ln[:, 0], y=ln[:, 1], z=ln[:, 2], mode="lines",
                line=dict(width=3, color=speed, colorscale="Turbo",
                          cmin=0.5, cmax=2.0),
                showlegend=False, hoverinfo="skip"))
        if k == 0:
            fig = go.Figure(data=tr)
        else:
            frames.append(go.Frame(data=tr, name=f"t={vtimes[k]:.1f}"))
        sliders.append(dict(method="animate",
                            args=[[f"t={vtimes[k]:.1f}"],
                                  {"mode": "immediate",
                                   "frame": {"duration": 0, "redraw": True}}],
                            label=f"{vtimes[k]:.1f}"))
    fig.frames = frames
    fig.update_layout(
        title=f"{tag}: streamlines through the cylinder wake (coloured by "
              f"speed), Re={meta['re']:.0f} — note the recirculation behind "
              f"the pillar",
        scene=dict(aspectmode="cube", xaxis_title="x (flow)",
                   yaxis_title="y", zaxis_title="z"),
        sliders=[dict(active=0, steps=sliders, x=0, y=-0.05)],
        width=1000, height=800)
    os.makedirs("artifacts", exist_ok=True)
    out = f"artifacts/{tag}_streamlines.html"
    fig.write_html(out, include_plotlyjs=True, full_html=True)
    print(f"wrote {out} ({ni} instants, {args.nseed} seeds)")


if __name__ == "__main__":
    main()
