# Changelog

## [2.0.0] — 2026-10-14

Re-founding release: the original research prototype (1.1.0) is re-architected
into a layered, verified package. See `docs/ENGINEERING.md` for the layout and
the experiments that were run.

### Added

- `pddlvm.numerics`: vectorized 1D FEM assembly (dense + O(n) banded),
  Chebyshev transforms and a spectral Poisson solver, stable steady/transient
  Burgers solvers (linearized θ-IMEX, damped Newton), 3D and 2D-obstacle
  Navier–Stokes residual operators (NumPy + TensorFlow), 2D/3D pseudo-spectral
  Navier–Stokes solvers (FFT, dealiasing, exact exponential integration),
  obstacle masks/penalties, Taylor–Green vortex utilities, and a
  manufactured-solution (MMS) framework.
- `pddlvm.probabilistic`: diagonal-Gaussian utilities, per-component ELBO
  records, an objective-agnostic `Trainer` (metrics, LR schedules, early
  stopping, checkpoints), and a `Checkpointer`.
- Experiment system: typed JSON configs (`configs/`), `run.py`, reproducible
  artifact pipelines (`metrics.json`, figures, checkpoints) — `pddlvm.experiment`.
- Test suite (85 tests): MMS convergence for every solver, energy-conservation
  checks, side-effect guards, reparameterization-gradient unbiasedness,
  simulation-based calibration, golden-value regression locks, same-seed
  reproducibility.
- Apps: 3D heat PDDLVM, 3D Navier–Stokes PDDLVM (Taylor–Green /
  Ethier–Steinman), cylinder-flow spectral reference, and cylinder PDDLVM
  (neural decoder + physics residual + data assimilation).
- Tooling: CI skeleton (ruff / mypy / pytest / coverage gate), offline coverage
  estimator, cloud orchestration scripts, `docs/ENGINEERING.md`, demo assets
  (`assets/`).

### Changed

- `VEModel` internals delegate to the probabilistic layer (public API
  unchanged; configuration errors now fail loudly).
- `apps/burgers_pddlvm.py` is now a genuine time-dependent PDDLVM compared
  against a transient reference under the same physics.
- All demo figures save to `artifacts/` instead of blocking on `plt.show()`
  (opt into interactive windows with `--show`).

### Removed

- Import-time matplotlib side effects (paper style moved to the opt-in
  `pddlvm.plotting.setup_paper_style`).
- The unconditionally unstable forward-Euler Burgers demo solver.
- `setup.py` and build artifacts (`*.egg-info`, `__pycache__`) dropped from
  version control.

## [1.1.0] — 2025-04-28

Original research prototype (as first published): neural networks
(`NeuralNet.py`) and a variational-EM model (`VEModel.py`) with 1D Poisson
solvers, plus Burgers / Navier–Stokes / Swift–Hohenberg demo scripts (`apps/`).
