"""Solid-obstacle representation for spectral solvers.

Obstacles enter the Fourier (periodic-domain) solvers through a smooth
indicator mask built from the signed distance to the solid surface::

    mask(x) = 0.5 * (1 - tanh(sdf(x) / width))     # 1 inside the solid

and a Brinkman penalty field ``lambda = mask / eps`` that drives the fluid
velocity to zero inside the solid.  ``width`` is the mask transition width in
physical units (a few grid spacings controls Gibbs ringing); ``eps`` is the
penalty relaxation time (small => stiffer, but the solvers integrate it
semi-implicitly so stability does not depend on ``eps``).
"""

import numpy as np


def sdf_cylinder(X, Y, cx, cy, D):
    """Signed distance to a cylinder of diameter ``D`` (positive outside).

    ``X, Y`` are coordinate grids (2D numpy arrays, matching shape).
    """
    return np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) - 0.5 * D


def sdf_sphere(X, Y, Z, cx, cy, cz, D):
    """Signed distance to a sphere of diameter ``D`` (positive outside)."""
    return np.sqrt((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2) - 0.5 * D


def smooth_mask(sdf, width):
    """Smooth solid indicator in [0, 1]; 1 well inside the solid."""
    return 0.5 * (1.0 - np.tanh(sdf / width))


def brinkman_penalty(mask, eps):
    """Brinkman penalty field ``lambda(x) = mask(x) / eps``."""
    return mask / eps


def grid_coords_2d(nx, ny, Lx, Ly):
    """Cell-centered coordinate grids of shape (ny, nx) for a periodic box."""
    x = (np.arange(nx) + 0.5) * Lx / nx
    y = (np.arange(ny) + 0.5) * Ly / ny
    X, Y = np.meshgrid(x, y, indexing="xy")
    return X, Y


def grid_coords_3d(nx, ny, nz, Lx, Ly, Lz):
    """Cell-centered coordinate grids of shape (nz, ny, nx)."""
    x = (np.arange(nx) + 0.5) * Lx / nx
    y = (np.arange(ny) + 0.5) * Ly / ny
    z = (np.arange(nz) + 0.5) * Lz / nz
    X, Y, Z = np.meshgrid(x, y, z, indexing="xy")
    return X, Y, Z


def cylinder_field(nx, ny, Lx, Ly, cx, cy, D, width, eps):
    """Full cylinder penalty field ``lambda`` of shape (ny, nx)."""
    X, Y = grid_coords_2d(nx, ny, Lx, Ly)
    mask = smooth_mask(sdf_cylinder(X, Y, cx, cy, D), width)
    return brinkman_penalty(mask, eps), mask


def sphere_field(nx, ny, nz, Lx, Ly, Lz, cx, cy, cz, D, width, eps):
    """Full sphere penalty field ``lambda`` of shape (nz, ny, nx)."""
    X, Y, Z = grid_coords_3d(nx, ny, nz, Lx, Ly, Lz)
    mask = smooth_mask(sdf_sphere(X, Y, Z, cx, cy, cz, D), width)
    return brinkman_penalty(mask, eps), mask


def cylinder3d_field(nx, ny, nz, Lx, Ly, Lz, cx, cy, D, width, eps):
    """Full finite vertical cylinder (pillar) penalty field of shape
    (nz, ny, nx).

    The pillar axis is along the solver's ``z`` direction (array **axis 0**,
    matching ``ns3d_spectral._wave_numbers``: axis 0 = z, axis 1 = y,
    axis 2 = x).  The cross-section is a disk of diameter ``D`` centred at
    ``(cx, cy)`` in the (x, y) plane.  Coordinates are built with explicit
    broadcasting rather than ``meshgrid`` to pin the axis convention down
    unambiguously.
    """
    x = ((np.arange(nx) + 0.5) * Lx / nx)[None, None, :]  # axis 2 = x
    y = ((np.arange(ny) + 0.5) * Ly / ny)[None, :, None]  # axis 1 = y
    sdf = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - 0.5 * D  # (1, ny, nx)
    mask = smooth_mask(sdf, width)
    mask = np.broadcast_to(mask, (nz, ny, nx)).copy()  # extrude along axis 0
    return brinkman_penalty(mask, eps), mask
