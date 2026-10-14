"""Pillar 3D renderer: Q-criterion vortex tubes + |u| wake, rotatable.

Turns the snapshot dump of ``tools/e9_pillar.py --snap`` (``.tmp/e9p_<tag>_snaps.npy``,
shape (n_frames, 2, n, n, n): channel 0 = Q-criterion, channel 1 = |u|) into a
self-contained, browser-rotatable HTML with a time slider (Plotly, embedded JS,
no server needed).

Usage:
    python tools/e9_render3d.py TAG [--ds 2] [--frames 15] [--qfrac 0.2]

Output: ``artifacts/<tag>_3d.html``
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--ds", type=int, default=2,
                    help="downsample stride (2 => 96^3 -> 48^3)")
    ap.add_argument("--frames", type=int, default=15,
                    help="number of time frames in the slider")
    ap.add_argument("--qfrac", type=float, default=None,
                    help="Q isosurface level as a fraction of max positive Q "
                         "(default: auto = 95th percentile of positive Q)")
    args = ap.parse_args()
    tag = args.tag

    import plotly.graph_objects as go

    with open(f".tmp/e9p_{tag}.json") as f:
        meta = json.load(f)
    L, n = meta["L"], meta["n"]
    snaps = np.load(f".tmp/e9p_{tag}_snaps.npy")  # (nf, 2, n, n, n)
    nf = snaps.shape[0]
    snap_every = meta.get("snap", 0) or 1
    snap_t = np.arange(nf) * snap_every * meta["dt"]

    # downsample
    q = snaps[:, 0, ::args.ds, ::args.ds, ::args.ds]
    umag = snaps[:, 1, ::args.ds, ::args.ds, ::args.ds]
    nn = q.shape[1]
    x = np.arange(nn) * (L / nn) + L / nn / 2
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    xf, yf, zf = X.ravel(), Y.ravel(), Z.ravel()
    # mask the solid pillar (radius 0.5 + a margin for the smeared mask
    # transition) so the penalty's artificial gradients don't appear as fake
    # vortex tubes that cut through the obstacle
    r = np.sqrt((X - L / 2) ** 2 + (Y - L / 2) ** 2)
    solid = r < 0.65
    q = q.copy()
    q[:, solid] = 0.0
    # mid-span (z = L/2) speed slice for flow-field context
    umid = umag[:, nn // 2, :, :]  # (nf, nn, nn)

    # threshold: auto = 95th percentile of positive Q (robust vortex-core
    # level, much lower than the max which is a few-cell outlier)
    qpos = q[q > 0]
    qmax = float(qpos.max()) if qpos.size else 0.0
    if args.qfrac is not None:
        iso = args.qfrac * qmax
    else:
        iso = float(np.percentile(qpos, 95)) if qpos.size else 0.0
    # keep only frames that actually have vortex structure above the threshold
    good = [i for i in range(nf) if float(q[i].max()) > iso]
    if not good:
        print(f"no frames above threshold iso={iso:.3g}; lower --qfrac")
        return
    idx = np.unique(np.linspace(0, len(good) - 1,
                                min(args.frames, len(good))).astype(int))
    idx = [good[i] for i in idx]

    # pillar (solid cylinder) + flow arrow for context
    th = np.linspace(0, 2 * np.pi, 40)
    zz = np.linspace(0, L, 16)
    TH, ZZ = np.meshgrid(th, zz)

    def pillar_trace():
        return go.Surface(
            x=L / 2 + 0.5 * np.cos(TH), y=L / 2 + 0.5 * np.sin(TH), z=ZZ,
            showscale=False, opacity=0.95,
            colorscale=[[0, "rgb(125,125,125)"], [1, "rgb(125,125,125)"]],
            name="pillar", hoverinfo="skip")

    def flow_trace():
        return go.Cone(x=[1.7], y=[L / 2], z=[L / 2], u=[1.0], v=[0.0], w=[0.0],
                       sizemode="absolute", sizeref=1.3, showscale=False,
                       anchor="tip",
                       colorscale=[[0, "rgb(20,20,20)"], [1, "rgb(20,20,20)"]],
                       name="flow (+x)", hoverinfo="skip")

    def speed_slice(i):
        # mid-span horizontal slice, coloured by flow speed (context: the
        # fluid fills the whole box; the wake slows down behind the pillar)
        return go.Surface(
            x=x, y=x, z=np.full((nn, nn), L / 2), surfacecolor=umid[i].T,
            colorscale="Viridis", opacity=0.85, showscale=True,
            colorbar=dict(title="|u|", thickness=15, len=0.5),
            name="flow speed (mid-span slice)", hoverinfo="skip")

    frames = []
    sliders_steps = []
    for k, i in enumerate(idx):
        traces = [speed_slice(i), pillar_trace(), flow_trace(),
                  go.Isosurface(
                      x=xf, y=yf, z=zf, value=q[i].ravel(),
                      isomin=iso, isomax=iso, opacity=0.9,
                      surface_count=1, colorscale="Reds",
                      caps=dict(x_show=False, y_show=False, z_show=False),
                      name="Q (vortex cores)", showscale=False)]
        if k == 0:
            fig = go.Figure(data=traces)
        else:
            frames.append(go.Frame(data=traces, name=f"t={snap_t[i]:.1f}"))
        sliders_steps.append(dict(
            method="animate",
            args=[[f"t={snap_t[i]:.1f}"],
                  {"mode": "immediate",
                   "frame": {"duration": 0, "redraw": True}}],
            label=f"{snap_t[i]:.1f}"))

    fig.frames = frames
    fig.update_layout(
        title=f"Cylinder wake, Re={meta['re']:.0f} — gray pillar, coloured "
              f"plane = flow speed, red = vortex cores (spinning fluid), "
              f"flow +x →",
        scene=dict(aspectmode="cube", xaxis_title="x (flow)",
                   yaxis_title="y", zaxis_title="z"),
        sliders=[dict(active=0, steps=sliders_steps, x=0, y=-0.05)],
        width=1000, height=800)
    os.makedirs("artifacts", exist_ok=True)
    out = f"artifacts/{tag}_3d.html"
    fig.write_html(out, include_plotlyjs=True, full_html=True)
    print(f"wrote {out} ({len(idx)} frames from {nf}, ds={args.ds}, "
          f"q_iso={iso:.3g})")


if __name__ == "__main__":
    main()
