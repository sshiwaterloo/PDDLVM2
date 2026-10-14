"""Batch post-processing plots: turn .tmp/e9_*.{json,npy} into figures.

Local-only (numpy + matplotlib, no cloud).  Produces the standard figure set
for a batch run; run after pulling .tmp from the cloud:

    python tools/e9_plot.py            # all figures into artifacts/
    python tools/e9_plot.py --only mode_map
    python tools/e9_plot.py --formats png,svg

Figures (artifacts/e9_*; default png + svg vector versions):
- mode_map.png/.svg : St vs Re for each box size (L=8 / L=16); the literature
                      line is a REFERENCE for unbounded flow, NOT ground truth
                      -- every series in this figure set is our simulation
- force_traces.png  : Cd(t) and Cl(t) per run (grid, one panel per run)
- spectra.png       : |FFT(Cl)|^2 per run with the f_peak marked
- wake_sheet.png    : vorticity contact sheet at 4 instants (from the --snap
                      run; requires .tmp/e9_<tag>_snaps.npy)
- e9_batch_summary.csv : flat table of all runs (for re-plotting elsewhere,
                      e.g. MATLAB -> save as .fig there)

Note: matplotlib cannot write MATLAB .fig; use the .svg files for vector
editing, or the CSV to re-plot in MATLAB.
"""
import argparse
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pddlvm.numerics import ns2d_spectral as S

LIT_ST = {100: 0.165, 200: 0.19, 300: 0.21}  # literature Strouhal (unbounded)


def _save(fig, out, fmts):
    """Save a figure in every requested format (png default + vector extras)."""
    base, _ = os.path.splitext(out)
    for fmt in fmts:
        fig.savefig(f"{base}.{fmt}", dpi=130)
    print("wrote", ", ".join(f"{base}.{f}" for f in fmts))


def load_runs():
    runs = {}
    for p in sorted(glob.glob(".tmp/e9_*.json")):
        with open(p) as f:
            meta = json.load(f)
        tag = meta["tag"]
        base = f".tmp/e9_{tag}"
        runs[tag] = {
            "meta": meta,
            "t": np.load(f"{base}_t.npy"),
            "cd": np.load(f"{base}_cd.npy"),
            "cl": np.load(f"{base}_cl.npy"),
        }
    return runs


def spectrum(cl, t, dt):
    sel = t >= max(60.0, 0.25 * t[-1])
    clw = (cl[sel] - cl[sel].mean()) * np.hanning(sel.sum())
    spec = np.abs(np.fft.rfft(clw, n=1 << 20)) ** 2
    freq = np.fft.rfftfreq(1 << 20, d=dt)
    return freq, spec


def mode_map(runs, fmts, out="artifacts/e9_mode_map"):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for L, color, marker in [(8.0, "C0", "o"), (16.0, "C1", "s")]:
        rs, ss = [], []
        for tag, r in runs.items():
            m = r["meta"]
            if m["L"] == L and m["drive"] == "pin" and not tag.startswith("VIZ"):
                rs.append(m["re"])
                ss.append(m["f_peak"])
        order = np.argsort(rs)
        ax.plot(np.array(rs)[order], np.array(ss)[order], f"-{marker}",
                color=color, lw=1.6, ms=7,
                label=f"simulation: L={L:.0f} periodic box + penalty")
        for re, st in zip(np.array(rs)[order], np.array(ss)[order]):
            ax.annotate(f"{st:.3f}", (re, st), textcoords="offset points",
                        xytext=(6, 4), fontsize=8, color=color)
    re_lit = sorted(LIT_ST)
    ax.plot(re_lit, [LIT_ST[r] for r in re_lit], "--k", lw=1.2, marker="D",
            mfc="none", ms=6,
            label="reference: literature St, unbounded flow\n"
                  "(NOT ground truth of this system)")
    for re, st in LIT_ST.items():
        ax.annotate(f"{st:.3f}", (re, st), textcoords="offset points",
                    xytext=(6, -12), fontsize=8, color="k")
    ax.set_xlabel("Re")
    ax.set_ylabel("St (f_peak from Cl spectrum)")
    ax.set_title("St-Re mode map -- all series are this work's simulations; "
                 "literature line is the unbounded-flow reference")
    ax.text(0.02, 0.02,
            "periodic-box + penalty system; unbounded-flow literature shown "
            "for reference",
            transform=ax.transAxes, fontsize=7.5, style="italic", color="0.35")
    ax.legend(fontsize=8.5); ax.grid(alpha=0.3)
    fig.tight_layout(); _save(fig, out, fmts); plt.close(fig)


def force_traces(runs, fmts, out="artifacts/e9_force_traces"):
    tags = [t for t in runs if not t.startswith("VIZ")]
    n = len(tags)
    cols = 2
    rows = (n + 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(11, 2.1 * rows + 0.5),
                            squeeze=False)
    for k, tag in enumerate(tags):
        r = runs[tag]
        m = r["meta"]
        ax = axs[k // cols, k % cols]
        ax.plot(r["t"], r["cd"], "C0", lw=0.8, label="Cd")
        ax.set_ylabel("Cd", color="C0")
        ax.tick_params(axis="y", labelcolor="C0")
        ax2 = ax.twinx()
        ax2.plot(r["t"], r["cl"], "C1", lw=0.5, alpha=0.8, label="Cl")
        ax2.set_ylabel("Cl", color="C1")
        ax2.tick_params(axis="y", labelcolor="C1")
        ax.set_title(f"{tag}  (St={m['f_peak']:.3f}, Cd_late={m['cd_late']:.3f})",
                     fontsize=9)
        ax.set_xlabel("t"); ax.grid(alpha=0.25)
    for k in range(n, rows * cols):
        axs[k // cols, k % cols].axis("off")
    fig.suptitle("Cd/Cl force traces -- all series are this work's simulations",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, out, fmts); plt.close(fig)


def spectra_plot(runs, fmts, out="artifacts/e9_spectra"):
    tags = [t for t in runs if not t.startswith("VIZ")]
    n = len(tags)
    cols = 2
    rows = (n + 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(11, 2.1 * rows + 0.5),
                            squeeze=False)
    for k, tag in enumerate(tags):
        r = runs[tag]
        m = r["meta"]
        freq, spec = spectrum(r["cl"], r["t"], m["dt"])
        ax = axs[k // cols, k % cols]
        ax.semilogy(freq[freq > 0], spec[freq > 0], "C2", lw=0.7)
        ax.axvline(m["f_peak"], color="r", ls="--", lw=1,
                   label=f"f_peak={m['f_peak']:.3f}")
        ax.set_title(f"{tag}", fontsize=9)
        ax.set_xlabel("f"); ax.set_ylabel("|FFT(Cl)|^2")
        ax.legend(fontsize=7); ax.grid(alpha=0.25)
        ax.set_xlim(0, min(2.5, freq.max()))
    for k in range(n, rows * cols):
        axs[k // cols, k % cols].axis("off")
    fig.suptitle("Cl power spectra (f_peak marked; all simulation)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, out, fmts); plt.close(fig)


def wake_sheet(runs, fmts, out="artifacts/e9_wake_sheet",
               times=(80, 160, 240, 320)):
    tag = None
    for t in runs:
        if t.startswith("VIZ"):
            tag = t
            break
    if tag is None:
        print("no --snap run found, skipping wake sheet")
        return
    r = runs[tag]
    m = r["meta"]
    snaps = np.load(f".tmp/e9_{tag}_snaps.npy")  # [nf, 2, n, n]
    snap_every = m.get("snap_every") or int(round(m["n_steps"] / snaps.shape[0]))
    snap_t = np.arange(snaps.shape[0]) * snap_every * m["dt"]
    idx = [int(np.argmin(np.abs(snap_t - tt))) for tt in times]
    wmax = max(float(np.abs(S.vorticity(snaps[i, 0], snaps[i, 1],
                                        Lx=m["L"], Ly=m["L"])).max())
               for i in idx)
    fig, axs = plt.subplots(1, len(times), figsize=(4.1 * len(times), 4.4),
                            constrained_layout=True)
    x = np.linspace(0, m["L"], m["n"])
    theta = np.linspace(0, 2 * np.pi, 200)
    cx = cy = m["L"] / 2
    for ax, tt, i in zip(axs, times, idx):
        w = S.vorticity(snaps[i, 0], snaps[i, 1], Lx=m["L"], Ly=m["L"])
        im = ax.imshow(w.T, origin="lower", extent=[0, m["L"], 0, m["L"]],
                       cmap="RdBu_r", vmin=-wmax, vmax=wmax)
        ax.plot(cx + 0.5 * np.cos(theta), cy + 0.5 * np.sin(theta), "k-", lw=1)
        ax.set_title(f"t = {snap_t[i]:.0f}")
        ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.colorbar(im, ax=axs, fraction=0.03, pad=0.02)
    fig.suptitle(f"{tag}: vorticity wake at selected instants (simulation)")
    _save(fig, out, fmts); plt.close(fig)


def write_csv(runs, out="artifacts/e9_batch_summary.csv"):
    fields = ["tag", "L", "re", "T", "dt", "n_steps", "drive", "amp",
              "St_f_peak", "Cd_late"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for tag, r in sorted(runs.items()):
            m = r["meta"]
            w.writerow({
                "tag": tag, "L": m["L"], "re": m["re"], "T": m["T"],
                "dt": m["dt"], "n_steps": m["n_steps"], "drive": m["drive"],
                "amp": m["amp"], "St_f_peak": m["f_peak"],
                "Cd_late": m["cd_late"],
            })
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None,
                    choices=["mode_map", "force_traces", "spectra",
                             "wake_sheet", "sphere", "pillar"])
    ap.add_argument("--formats", default="png,svg",
                    help="comma-separated output formats (matplotlib cannot "
                         "write MATLAB .fig; svg is the vector/editable "
                         "equivalent)")
    args = ap.parse_args()
    fmts = [f.strip() for f in args.formats.split(",") if f.strip()]
    runs = load_runs()
    if not runs:
        print("no .tmp/e9_*.json found; pull from cloud first")
        sys.exit(1)
    os.makedirs("artifacts", exist_ok=True)
    if args.only is None or args.only == "mode_map":
        mode_map(runs, fmts)
    if args.only is None or args.only == "force_traces":
        force_traces(runs, fmts)
    if args.only is None or args.only == "spectra":
        spectra_plot(runs, fmts)
    if args.only is None or args.only == "wake_sheet":
        wake_sheet(runs, fmts)
    if args.only is None or args.only == "sphere":
        sphere_plots(fmts)
    if args.only is None or args.only == "pillar":
        pillar_plots(fmts)
    if args.only is None:
        write_csv(runs)


def obstacle_plots(prefix, outbase, fmts, suptitle_forces, suptitle_spectra):
    """Generic 3D-obstacle figures (force traces + Cl spectra) for files
    ``.tmp/<prefix><tag>.{json,_t,_cd,_cly,_clz}.npy``."""
    tags = sorted({os.path.basename(p)[len(prefix):-len(".json")]
                   for p in glob.glob(f".tmp/{prefix}*.json")
                   if not os.path.basename(p).startswith(f"{prefix}SMOKE")})
    if not tags:
        print(f"no .tmp/{prefix}*.json found; pull from cloud first")
        return
    # force traces
    n = len(tags)
    cols, rows = 2, (n + 1) // 2
    fig, axs = plt.subplots(rows, cols, figsize=(11, 2.6 * rows + 0.5),
                            squeeze=False)
    for k, tag in enumerate(tags):
        with open(f".tmp/{prefix}{tag}.json") as f:
            m = json.load(f)
        t = np.load(f".tmp/{prefix}{tag}_t.npy")
        cd = np.load(f".tmp/{prefix}{tag}_cd.npy")
        cly = np.load(f".tmp/{prefix}{tag}_cly.npy")
        clz = np.load(f".tmp/{prefix}{tag}_clz.npy")
        ax = axs[k // cols, k % cols]
        ax.plot(t, cd, "C0", lw=0.8, label="Cd")
        ax.set_ylabel("Cd", color="C0")
        ax.tick_params(axis="y", labelcolor="C0")
        ax2 = ax.twinx()
        ax2.plot(t, cly, "C1", lw=0.5, alpha=0.8, label="Cl_y")
        ax2.plot(t, clz, "C2", lw=0.5, alpha=0.8, label="Cl_z")
        ax2.set_ylabel("Cl", color="C1")
        ax.set_title(f"{tag}  St_y={m['f_peak_y']:.3f} St_z={m['f_peak_z']:.3f} "
                     f"Cd_late={m['cd_late']:.3f}", fontsize=9)
        ax.set_xlabel("t"); ax.grid(alpha=0.25)
    for k in range(n, rows * cols):
        axs[k // cols, k % cols].axis("off")
    fig.suptitle(suptitle_forces, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, f"artifacts/{outbase}_force_traces", fmts)
    plt.close(fig)

    # spectra
    fig, axs = plt.subplots(rows, cols, figsize=(11, 2.6 * rows + 0.5),
                            squeeze=False)
    for k, tag in enumerate(tags):
        with open(f".tmp/{prefix}{tag}.json") as f:
            m = json.load(f)
        t = np.load(f".tmp/{prefix}{tag}_t.npy")
        cly = np.load(f".tmp/{prefix}{tag}_cly.npy")
        clz = np.load(f".tmp/{prefix}{tag}_clz.npy")
        ax = axs[k // cols, k % cols]
        for sig, c, lab, fp in [(cly, "C1", "Cl_y", m["f_peak_y"]),
                                (clz, "C2", "Cl_z", m["f_peak_z"])]:
            sel = t >= max(30.0, 0.25 * t[-1])
            sw = (sig[sel] - sig[sel].mean()) * np.hanning(sel.sum())
            spec = np.abs(np.fft.rfft(sw, n=1 << 18)) ** 2
            freq = np.fft.rfftfreq(1 << 18, d=m["dt"])
            ax.semilogy(freq[freq > 0], spec[freq > 0], c, lw=0.7, label=lab)
        ax.axvline(m["f_peak_y"], color="r", ls="--", lw=0.8,
                   label=f"St_y={m['f_peak_y']:.3f}")
        ax.set_title(tag, fontsize=9)
        ax.set_xlabel("f"); ax.set_ylabel("|FFT(Cl)|^2")
        ax.legend(fontsize=7); ax.grid(alpha=0.25)
        ax.set_xlim(0, min(2.0, freq.max()))
    for k in range(n, rows * cols):
        axs[k // cols, k % cols].axis("off")
    fig.suptitle(suptitle_spectra, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, f"artifacts/{outbase}_spectra", fmts)
    plt.close(fig)
    print(f"{outbase} plots done for", tags)


def sphere_plots(fmts):
    obstacle_plots(
        "e9s_", "e9s", fmts,
        "Sphere: force traces (all series are simulation; "
        "literature St~0.136/Cd~0.66 is unbounded-flow reference)",
        "Sphere: Cl spectra (f_peak marked)")


def pillar_plots(fmts):
    obstacle_plots(
        "e9p_", "e9p", fmts,
        "Pillar: force traces (vertical cylinder; all simulation; "
        "note St_y vs St_z anisotropy)",
        "Pillar: Cl spectra (f_peak marked)")


if __name__ == "__main__":
    main()
