# PDDLVM

[English](README.md) · [中文](README.zh.md)

Physics-Driven Deep Latent Variable Models (PDDLVM) for PDE inverse problems:
Burgers, Navier–Stokes, and Swift–Hohenberg equations.

PDDLVM embeds a PDE forward solver inside a hierarchical latent-variable model:
latent parameters (conductivity, forcing, boundary conditions) generate a
solution field through the physics, and an amortized variational posterior
infers the latent parameters from observations. See
`reference/PDDLVMPaper/paper_arxiv.tex` for the original paper and
`reference/PDDLVM-main/` for the original code release.

> The package is organized in independently tested layers (numerics,
> probabilistic, experiment). See `docs/ENGINEERING.md` for a full overview.

## Layout

```
pddlvm/                 core package (numerics, probabilistic, networks, VE model)
  tests/                unit tests
apps/                   demo scripts (Burgers / NS / Swift-Hohenberg / heat / cylinder)
docs/                   ENGINEERING.md / ENGINEERING.zh.md — package overview
reference/              original paper source (tex) and original code release
```

## Installation

The training environment is a pinned conda environment:

```bash
conda env create -f environment.yml
conda activate pddlvm2
pip install -e .
```

For development tooling (pytest, ruff, mypy) add the dev extra, or clone the
pinned env and add the tools:

```bash
conda create -n pddlvm2-dev --clone pddlvm2
conda activate pddlvm2-dev
pip install pytest ruff mypy
pip install -e ".[dev]"
```

> Note for Apple Silicon: the TensorFlow build used here comes from the
> conda channel (`tensorflow-macos` / `tensorflow-metal`); installing plain
> `tensorflow` from PyPI does not provide arm64 wheels.

## Quickstart

Run the unit tests:

```bash
python -m unittest discover -s pddlvm/tests -v
```

Run a demo (interactive plotting; `--show` pops figures, otherwise they
are saved under `artifacts/`):

```bash
python apps/heat3d_pddlvm.py --quick   # 3D heat PDDLVM (GIF + pngs)
python apps/burgers_pddlvm.py
```

> Demos must run with the `pddlvm2` environment (TensorFlow is not
> installed system-wide).  `python apps/foo.py` works from the repo root
> without installing the package (apps bootstrap the import path), but
> `pip install -e .` is the supported setup.

## Demos

Cylinder vortex shedding (pseudo-spectral solver, Re = 100):

![Cylinder vortex street](assets/cylinder_vortex_street.gif)

Cylinder PDDLVM — data assimilation (observations → full field): learned vs
ground-truth vorticity:

![Cylinder PDDLVM data assimilation](assets/cylinder_pddlvm_data_vorticity.gif)

## Development

```bash
ruff check .          # lint
mypy pddlvm           # type check
pytest                # tests
python tools/coverage.py   # offline line-coverage estimate
```

## API docs

Zero-dependency doc generation (stdlib pydoc):

```bash
python tools/build_docs.py   # writes docs/api/*.html
```

## Docker

```bash
docker build -t pddlvm:2.0.0 .
docker run --rm -v "$PWD/artifacts:/work/artifacts" \
    pddlvm:2.0.0 python run.py configs/ve_prior_default.json --steps 200
```

## Release notes

See `CHANGELOG.md`; for a package overview see `docs/ENGINEERING.md`.

## License

MIT — see `LICENSE`.
