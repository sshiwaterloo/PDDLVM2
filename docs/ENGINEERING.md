# PDDLVM — Engineering Overview

[English](ENGINEERING.md) · [中文](ENGINEERING.zh.md)

PDDLVM (Physics-Driven Deep Latent Variable Models) solves PDE forward and
inverse problems by embedding a parametric PDE solver inside a hierarchical
latent-variable model with amortized variational inference. Latent parameters
(conductivity, forcing, boundary conditions) generate a solution field through
the physics, and a variational posterior infers them from observations.

This document is an index of the package: its directory layout, the numerical
and probabilistic layers, the experiments that were run (and on what
hardware), and the testing/development workflow. For the original paper and
code release, see `reference/`.

## Directory layout

```
PDDLVM2/
├── README.md                         getting started, quickstart, demos
├── CHANGELOG.md                      release notes (1.1.0 → 2.0.0)
├── LICENSE                           MIT
├── environment.yml                   pinned conda environment (TF/TFP)
├── pyproject.toml                    package metadata / tool config
├── run.py                            experiment runner (typed JSON configs)
├── Dockerfile / .dockerignore        container build
├── .github/workflows/ci.yml          CI (ruff / mypy / pytest / coverage)
├── configs/
│   └── ve_prior_default.json         default variational-EM prior config
├── assets/                           demo GIFs and figures (referenced by docs)
│   ├── cylinder_vortex_street.gif    cylinder vortex shedding (pseudo-spectral solver)
│   ├── cylinder_pddlvm_data_vorticity.gif data assimilation: learned vs GT vorticity
│   └── cylinder_pddlvm_data_u.png    data assimilation: solid-region suppression
├── pddlvm/
│   ├── __init__.py                   package entry, __version__
│   ├── VEModel.py                    variational-EM model (public API)
│   ├── NeuralNet.py                  networks: NeuralN, ConvFieldNet(2D/3D)
│   ├── experiment.py                 experiment system (configs, artifacts)
│   ├── plotting.py                   opt-in publication-style plotting
│   ├── utilities.py                  shared utilities
│   ├── poisson1DPDEState.py          1D Poisson state (delegates to numerics)
│   ├── poisson1DSolver.py            1D Poisson solver (delegates to numerics)
│   ├── numerics/                     numerical solvers and residual operators
│   │   ├── __init__.py
│   │   ├── fem.py                    vectorized 1D FEM assembly (dense/banded)
│   │   ├── chebyshev.py              Chebyshev basis/transforms + spectral Poisson
│   │   ├── fd.py                     stable Burgers (θ-IMEX, damped Newton)
│   │   ├── mms.py                    manufactured solutions, error norms, rates
│   │   ├── heat3d.py                 3D heat-equation solver
│   │   ├── ns3d.py                   3D Navier–Stokes residual (NumPy + TF)
│   │   ├── ns2d_obstacle.py          2D obstacle NS residual (NumPy + TF)
│   │   ├── ns3d_spectral.py          3D pseudo-spectral NS (FFT, RK4)
│   │   ├── ns2d_spectral.py          2D pseudo-spectral NS + obstacle (exact exponential integrators)
│   │   ├── turbulence.py             Taylor–Green vortex, energy/enstrophy spectra
│   │   └── obstacles.py              cylinder/sphere SDF, masks, penalty fields
│   ├── probabilistic/                variational inference and training
│   │   ├── __init__.py
│   │   ├── distributions.py          diagonal Gaussian, residual likelihood
│   │   ├── elbo.py                   per-component ELBO records
│   │   ├── trainer.py                tf.data + objective-agnostic Trainer
│   │   └── checkpointing.py          Checkpointer (seamless resume)
│   └── tests/                        unit tests (85 tests)
│       ├── __init__.py
│       ├── test_numerics.py          FEM/Chebyshev/Burgers MMS convergence
│       ├── test_obstacles.py         mask/penalty/SDF properties
│       ├── test_ns3d.py              3D NS residual (TF↔NumPy, MMS)
│       ├── test_ns3d_obstacles.py    obstacle NS checks
│       ├── test_ns2d_obstacle.py     2D obstacle NS checks
│       ├── test_probabilistic.py     distributions, ELBO arithmetic
│       ├── test_experiment.py        experiment system
│       ├── test_calibration.py       simulation-based calibration
│       ├── test_golden.py            golden-value regression locks
│       ├── test_heat3d.py            3D heat checks
│       ├── test_neuraln.py           network shapes / initialization
│       ├── test_side_effects.py      import side-effect guards
│       └── testBasic.ipynb           original notebook demo
├── apps/                             demo drivers (each writes artifacts/)
│   ├── burgers_fd.py                 Burgers finite-difference reference
│   ├── burgers_spectral.py           Burgers pseudo-spectral reference
│   ├── burgers_pddlvm.py             time-dependent Burgers PDDLVM
│   ├── burgers_pinnpddlvm.py         Burgers PDDLVM (PINN-style)
│   ├── ns_spectral.py                Navier–Stokes pseudo-spectral reference
│   ├── ns_pddlvm.py                  steady Navier–Stokes PDDLVM
│   ├── sh_spectral.py                Swift–Hohenberg pseudo-spectral reference
│   ├── sh_pddlvm.py                  Swift–Hohenberg PDDLVM
│   ├── heat3d_pddlvm.py              3D heat PDDLVM
│   ├── ns3d_pddlvm.py                3D Navier–Stokes PDDLVM (TGV / ES)
│   ├── cylinder_spectral.py          cylinder-flow spectral reference (Re 100/300)
│   └── cylinder_pddlvm.py            cylinder PDDLVM (decoder + residual + data)
├── tools/                            dev + experiment orchestration
│   ├── build_docs.py                 offline API doc generation
│   ├── coverage.py                   offline line-coverage estimate
│   ├── e9_batch.py / e9_plot.py / e9_animate.py
│   └── e9_L16.py / e9_pillar.py / e9_sphere.py / e9_render3d.py / e9_streamlines.py
├── docs/
│   ├── ENGINEERING.md                this file (English)
│   ├── ENGINEERING.zh.md             Chinese version
│   └── api/                          generated API docs (tools/build_docs.py)
└── reference/
    ├── PDDLVMPaper/                  original paper source (TeX)
    └── PDDLVM-main/                  original code release snapshot
```

## Numerical layer (`pddlvm.numerics`)

- `fem.py` — vectorized 1D linear finite-element assembly (dense and O(n)
  banded) with Dirichlet elimination.
- `chebyshev.py` — Chebyshev basis/transforms and a spectral Poisson solver.
- `fd.py` — stable Burgers solvers: linearized θ-IMEX time stepping and a
  damped-Newton steady-state iteration.
- `mms.py` — manufactured-solution framework (error norms, convergence-rate
  estimation), including the Ethier–Steinman 3D Navier–Stokes analytical
  solution.
- `heat3d.py` — 3D heat-equation solver.
- `ns3d.py` / `ns2d_obstacle.py` — 3D and 2D-obstacle Navier–Stokes residual
  operators (NumPy + TensorFlow).
- `ns3d_spectral.py` / `ns2d_spectral.py` — pseudo-spectral Navier–Stokes
  solvers (FFT, 2/3 dealiasing, divergence-free projection, exact exponential
  integration factors); the 2D solver supports immersed obstacles with mean-flow
  pinning and drag/lift probes.
- `turbulence.py` — Taylor–Green vortex initial conditions and
  energy/enstrophy/shell-averaged spectra.
- `obstacles.py` — cylinder/sphere signed-distance fields, smoothed masks, and
  penalty fields.

## Probabilistic layer (`pddlvm.probabilistic`)

- `distributions.py` — diagonal-Gaussian log-probability / reparameterization,
  and a Gaussian residual likelihood.
- `elbo.py` — per-component ELBO records (log-likelihood / log-prior / log-q /
  ELBO / NELBO) plus the physics Gaussian likelihood.
- `trainer.py` — `tf.data` dataset builder and an objective-agnostic `Trainer`
  (gradient tape, component metrics, LR schedules, early stopping,
  checkpointing, optional `tf.function`).
- `checkpointing.py` — `Checkpointer` with seamless resume.

## Experiment system

`pddlvm.experiment` + `run.py` + typed JSON configs (`configs/`) provide a
reproducible pipeline: every run writes `metrics.json`, figures, checkpoints and
training history under `artifacts/`, keyed by run tag.

## Applications (`apps/`)

| App | Description |
|---|---|
| `burgers_pddlvm.py` | time-dependent Burgers PDDLVM vs a transient reference |
| `burgers_pinnpddlvm.py` | Burgers PDDLVM (PINN-style) |
| `burgers_fd.py` / `burgers_spectral.py` | Burgers finite-difference / pseudo-spectral references |
| `ns_pddlvm.py` / `ns_spectral.py` | steady Navier–Stokes PDDLVM / spectral reference |
| `sh_pddlvm.py` / `sh_spectral.py` | Swift–Hohenberg PDDLVM / spectral reference |
| `heat3d_pddlvm.py` | 3D heat PDDLVM (GIF + PNGs) |
| `ns3d_pddlvm.py` | 3D Navier–Stokes PDDLVM (Taylor–Green / Ethier–Steinman) |
| `cylinder_spectral.py` | cylinder-flow spectral reference (Re = 100 / 300, St/Cd/Cl) |
| `cylinder_pddlvm.py` | 2D cylinder PDDLVM: neural decoder + physics residual, plus data assimilation |

## Experiments and results

All runs are reproducible via `run.py` / the `apps/` drivers and the `tools/`
batch scripts; outputs land in `artifacts/`.

### Hardware

| Label | Configuration |
|---|---|
| **Local** | Apple Silicon macOS, CPU (TensorFlow via `tensorflow-macos`; Python 3.9, TF 2.19, TFP 0.25) |
| **Cloud** | Linux cloud instance, NVIDIA RTX 3080 Ti (12 GB) + 80 vCPU, TF 2.19.1 + TFP 0.25 |
| **GPU workstation** | NVIDIA RTX 4090D (24 GB) |

### Solver verification (manufactured solutions)

| Experiment | Setting | Hardware | Result |
|---|---|---|---|
| FEM MMS | 1D Poisson / Burgers, h-refinement | Local | energy rate 1.0, L² rate 2.0 (matches theory) |
| Chebyshev spectral MMS | 1D, p-refinement | Local | spectral convergence to round-off (7.8e-16) |
| Ethier–Steinman MMS | 3D Navier–Stokes, [−1,1]³ | Local | truncation scaling 0.145 → 0.044 (O(dt + dx²)) |
| Inviscid TGV energy | 3D pseudo-spectral, RK4 | Local | energy drift 2.55e-15 (machine precision) |
| Viscous TGV decay | 2D pseudo-spectral | Local | matches analytic decay to machine precision |

### Flow around a cylinder

| Experiment | Setting | Hardware | Result |
|---|---|---|---|
| Cylinder flow (spectral reference) | Re = 100/200/300, 256²/512² grids, periodic box + penalization | Cloud | vortex shedding; drag/lift (Cd/Cl) and Strouhal-number measurement |
| Cylinder PDDLVM (residual-driven) | Re = 100/300, grids n = 32–256, nt = 16, 12k steps | RTX 4090D | residual down 442×–18,600×; solid-region \|u\| → 0.034 (GT 0.028); wake velocity deficit 0.80 (GT 0.78) |
| Cylinder PDDLVM (data assimilation) | sparse → full observations, deterministic decoder | RTX 4090D | 15/15 seeds reconstruct the full field (rel. err. 0.025, corr. 0.97) |

### Deep latent-variable demos

| Demo | Equation | Hardware |
|---|---|---|
| `heat3d_pddlvm.py` | 3D heat | Local |
| `ns3d_pddlvm.py` | 3D Navier–Stokes (Taylor–Green / Ethier–Steinman) | GPU |
| `burgers_pddlvm.py` | time-dependent Burgers | Local |

## Testing

`python -m unittest discover -s pddlvm/tests -v` runs the full suite (85 tests),
covering manufactured-solution convergence, energy conservation, side-effect
guards, reparameterization-gradient unbiasedness, simulation-based calibration,
golden-value regression locks, and same-seed reproducibility.

## Development

```bash
ruff check .              # lint
mypy pddlvm               # type check
pytest                    # tests
python tools/coverage.py  # offline line-coverage estimate
python tools/build_docs.py  # regenerate docs/api/
```

## Installation

```bash
conda env create -f environment.yml
conda activate pddlvm2
pip install -e .
```

For development tooling (pytest, ruff, mypy), add the dev extra or clone the
pinned environment as described in `README.md`.

## Docker

```bash
docker build -t pddlvm:2.0.0 .
docker run --rm -v "$PWD/artifacts:/work/artifacts" \
    pddlvm:2.0.0 python run.py configs/ve_prior_default.json --steps 200
```

## License

MIT — see `LICENSE`.
