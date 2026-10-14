"""Numerics layer for PDDLVM.

Pure, side-effect-free numerical routines (no matplotlib imports, no graph
state).  Imports are LAZY (PEP 562 ``__getattr__``): modules that pull in
TensorFlow (``chebyshev``, ``heat3d``, ``ns3d``) are only imported on demand,
so the pure-NumPy subpackage (``fem`` / ``fd`` / ``mms`` / ``obstacles`` /
``ns2d_spectral`` / ``ns3d_spectral`` / ``turbulence``) works in TF-free
environments (e.g. a CPU-only cloud solver environment).

- ``chebyshev`` : Chebyshev transforms and a spectral Poisson solver (TF)
- ``fem``       : vectorized 1D finite-element assembly for -d/dx(k du/dx) = f
- ``fd``        : stable steady/transient 1D Burgers solvers (NumPy reference)
- ``heat3d``    : 3D heat-equation residual operators (NumPy + TF)
- ``mms``       : manufactured solutions, error norms, convergence rates
- ``obstacles`` : SDF / smooth-mask / Brinkman-penalty fields
- ``ns2d_spectral`` : 2D pseudospectral NS + obstacles
- ``ns3d_spectral`` : 3D pseudospectral NS + obstacles
- ``turbulence``    : TGV initial data and spectra
"""

import importlib

_LAZY = {
    # modules
    "chebyshev": ("pddlvm.numerics.chebyshev", None),
    "fem": ("pddlvm.numerics.fem", None),
    "fd": ("pddlvm.numerics.fd", None),
    "heat3d": ("pddlvm.numerics.heat3d", None),
    "mms": ("pddlvm.numerics.mms", None),
    "obstacles": ("pddlvm.numerics.obstacles", None),
    "ns2d_spectral": ("pddlvm.numerics.ns2d_spectral", None),
    "ns3d": ("pddlvm.numerics.ns3d", None),
    "ns3d_spectral": ("pddlvm.numerics.ns3d_spectral", None),
    "turbulence": ("pddlvm.numerics.turbulence", None),
    # chebyshev (TF)
    "cheb_T_mat": ("pddlvm.numerics.chebyshev", "cheb_T_mat"),
    "cheb_eval": ("pddlvm.numerics.chebyshev", "cheb_eval"),
    "cheb_fit": ("pddlvm.numerics.chebyshev", "cheb_fit"),
    "cheb_integrate": ("pddlvm.numerics.chebyshev", "cheb_integrate"),
    "poisson_const_kappa": ("pddlvm.numerics.chebyshev", "poisson_const_kappa"),
    "poisson_const_kappa_values": ("pddlvm.numerics.chebyshev",
                                   "poisson_const_kappa_values"),
    # fem
    "assemble_poisson_1d": ("pddlvm.numerics.fem", "assemble_poisson_1d"),
    "assemble_poisson_1d_banded": ("pddlvm.numerics.fem",
                                   "assemble_poisson_1d_banded"),
    "apply_dirichlet": ("pddlvm.numerics.fem", "apply_dirichlet"),
    "residual_poisson_1d": ("pddlvm.numerics.fem", "residual_poisson_1d"),
    "solve_poisson_1d": ("pddlvm.numerics.fem", "solve_poisson_1d"),
    # fd
    "solve_burgers_steady": ("pddlvm.numerics.fd", "solve_burgers_steady"),
    "solve_burgers_transient": ("pddlvm.numerics.fd", "solve_burgers_transient"),
    "burgers_space_time_residual": ("pddlvm.numerics.fd",
                                    "burgers_space_time_residual"),
    "burgers_space_time_residual_tf": ("pddlvm.numerics.fd",
                                       "burgers_space_time_residual_tf"),
    # heat3d (TF for the _tf variant)
    "heat3d_residual": ("pddlvm.numerics.heat3d", "heat3d_residual"),
    "heat3d_residual_tf": ("pddlvm.numerics.heat3d", "heat3d_residual_tf"),
    # ns3d (TF for the _tf variant)
    "ns3d_residual": ("pddlvm.numerics.ns3d", "ns3d_residual"),
    "ns3d_residual_tf": ("pddlvm.numerics.ns3d", "ns3d_residual_tf"),
    # turbulence
    "tgv_initial": ("pddlvm.numerics.turbulence", "tgv_initial"),
    "kinetic_energy": ("pddlvm.numerics.turbulence", "kinetic_energy"),
    "enstrophy": ("pddlvm.numerics.turbulence", "enstrophy"),
    "energy_spectrum": ("pddlvm.numerics.turbulence", "energy_spectrum"),
}


def __getattr__(name):
    if name in _LAZY:
        mod_name, attr = _LAZY[name]
        module = importlib.import_module(mod_name)
        return module if attr is None else getattr(module, attr)
    raise AttributeError(f"module 'pddlvm.numerics' has no attribute "
                         f"{name!r}")


__all__ = list(_LAZY.keys())
