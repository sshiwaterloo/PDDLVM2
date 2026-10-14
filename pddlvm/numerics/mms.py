"""Manufactured solutions (MMS) and error/convergence utilities.

Every solver in this package must be verified against a manufactured
solution: pick an analytic field u*, substitute it into the PDE, and derive
the source term that makes u* exact.  A correct solver then reproduces u*
with a computable error, whose decay rate under refinement is the solver's
verified convergence order.

The reference analytics live in NumPy (the reference solvers in this package
are NumPy-based); TensorFlow callers convert the arrays themselves.
"""

import numpy as np


# --------------------------------------------------------------------------
# Poisson MMS: -d/dx(kappa du/dx) = f on [-1, 1]
# u*(x) = sin(pi x), kappa(x) = 2 + x  (affine -> midpoint sampling is exact)
# f* = -(kappa' u' + kappa u'') = -pi cos(pi x) + pi^2 (2 + x) sin(pi x)
# --------------------------------------------------------------------------

def poisson_mms_u(x):
    return np.sin(np.pi * x)


def poisson_mms_u_x(x):
    return np.pi * np.cos(np.pi * x)


def poisson_mms_kappa(x):
    return 2.0 + x


def poisson_mms_source(x):
    return -np.pi * np.cos(np.pi * x) + np.pi ** 2 * (2.0 + x) * np.sin(np.pi * x)


# --------------------------------------------------------------------------
# Burgers MMS (transient): u_t + u u_x - nu u_xx = S
# u*(x, t) = exp(-t) sin(2 pi x) on x in [0, 1]
# --------------------------------------------------------------------------

def burgers_mms_u_transient(x, t, nu=0.1):
    return np.exp(-t) * np.sin(2.0 * np.pi * x)


def burgers_mms_source_transient(x, t, nu=0.1):
    u = np.exp(-t) * np.sin(2.0 * np.pi * x)
    u_t = -u
    u_x = 2.0 * np.pi * np.exp(-t) * np.cos(2.0 * np.pi * x)
    u_xx = -4.0 * np.pi ** 2 * u
    return u_t + u * u_x - nu * u_xx


# --------------------------------------------------------------------------
# Burgers MMS (steady): u u_x - nu u_xx = S
# u*(x) = 1 + 0.3 x^2 + 0.1 x^4 on x in [0, 1]   (BCs: u(0)=1, u(1)=1.4)
# (a pure quadratic would be reproduced exactly by central differences,
# making the MMS rate degenerate; the quartic term restores a real O(dx^2)
# truncation error with a small constant)
# --------------------------------------------------------------------------

def burgers_mms_u_steady(x, nu=0.1):
    return 1.0 + 0.3 * x ** 2 + 0.1 * x ** 4


def burgers_mms_source_steady(x, nu=0.1):
    u = 1.0 + 0.3 * x ** 2 + 0.1 * x ** 4
    u_x = 0.6 * x + 0.4 * x ** 3
    u_xx = 0.6 + 1.2 * x ** 2
    return u * u_x - nu * u_xx


# --------------------------------------------------------------------------
# 3D heat equation MMS: u_t - nu Lap(u) = f on [0,1]^3, homogeneous Dirichlet
# u*(x,y,z,t) = sin(pi x) sin(2 pi y) sin(3 pi z) e^{-t}
#             + 0.5 sin(2 pi x) sin(pi y) sin(2 pi z) e^{-2t}
# (both modes vanish on the boundary, so the BCs are homogeneous)
# --------------------------------------------------------------------------

_HEAT_NU = 0.05
_HEAT_L1 = np.pi ** 2 * (1 + 4 + 9)   # mode 1 eigenvalue
_HEAT_L2 = np.pi ** 2 * (4 + 1 + 4)   # mode 2 eigenvalue


def heat3d_mms_u(x, y, z, t, nu=_HEAT_NU):
    m1 = (np.sin(np.pi * x) * np.sin(2 * np.pi * y) * np.sin(3 * np.pi * z)
          * np.exp(-t))
    m2 = 0.5 * (np.sin(2 * np.pi * x) * np.sin(np.pi * y)
                * np.sin(2 * np.pi * z) * np.exp(-2 * t))
    return m1 + m2


def heat3d_mms_source(x, y, z, t, nu=_HEAT_NU):
    """f* = u*_t - nu Lap(u*): per-mode analytic evaluation."""
    m1 = ((nu * _HEAT_L1 - 1.0)
          * np.sin(np.pi * x) * np.sin(2 * np.pi * y) * np.sin(3 * np.pi * z)
          * np.exp(-t))
    m2 = (0.5 * (nu * _HEAT_L2 - 2.0)
          * np.sin(2 * np.pi * x) * np.sin(np.pi * y)
          * np.sin(2 * np.pi * z) * np.exp(-2 * t))
    return m1 + m2


# --------------------------------------------------------------------------
# 3D Navier-Stokes MMS: the Ethier-Steinman exact solution (1994)
#
# An unsteady, incompressible, zero-forcing exact solution of the 3D NS
# equations on [-1,1]^3 with parameters a and d.  All three velocity
# components and the pressure are analytic; the residual of the FD
# operator applied to it is O(dt + dx^2) (verified in test_ns3d.py).
#
#   u = -a [ e^{ax} sin(ay+dz) + e^{az} cos(ax+dy) ] e^{-nu d^2 t}
#   v = -a [ e^{ay} sin(az+dx) + e^{ax} cos(ay+dz) ] e^{-nu d^2 t}
#   w = -a [ e^{az} sin(ax+dy) + e^{ay} cos(az+dx) ] e^{-nu d^2 t}
#   p = -(a^2/2) [ e^{2ax} + e^{2ay} + e^{2az}
#        + 2 sin(ax+dy) cos(az+dx) e^{a(y+z)}
#        + 2 sin(ay+dz) cos(ax+dy) e^{a(z+x)}
#        + 2 sin(az+dx) cos(ay+dz) e^{a(x+y)} ] e^{-2 nu d^2 t}
#
# The solution is NOT periodic: use Dirichlet BCs taken from the fields.
# --------------------------------------------------------------------------

_ES_A = np.pi / 4.0
_ES_D = np.pi / 2.0


def es3d_fields(x, y, z, t, nu=0.1, a=_ES_A, d=_ES_D):
    """Ethier-Steinman velocity (u, v, w) and pressure p.

    Inputs may be scalars or broadcastable arrays.  Returns a tuple
    (u, v, w, p).
    """
    e = np.exp(-nu * d ** 2 * t)
    ax, ay, az = a * x, a * y, a * z
    ex, ey, ez = np.exp(ax), np.exp(ay), np.exp(az)
    dx_, dy_, dz_ = d * x, d * y, d * z

    u = -a * (ex * np.sin(ay + dz_) + ez * np.cos(ax + dy_)) * e
    v = -a * (ey * np.sin(az + dx_) + ex * np.cos(ay + dz_)) * e
    w = -a * (ez * np.sin(ax + dy_) + ey * np.cos(az + dx_)) * e

    p = -0.5 * a ** 2 * (
        np.exp(2 * ax) + np.exp(2 * ay) + np.exp(2 * az)
        + 2.0 * np.sin(ax + dy_) * np.cos(az + dx_) * np.exp(a * (y + z))
        + 2.0 * np.sin(ay + dz_) * np.cos(ax + dy_) * np.exp(a * (z + x))
        + 2.0 * np.sin(az + dx_) * np.cos(ay + dz_) * np.exp(a * (x + y))
    ) * np.exp(-2.0 * nu * d ** 2 * t)
    return u, v, w, p


def es3d_grid(x, t, nu=0.1):
    """Ethier-Steinman fields on a 4D space-time grid.

    x : [n] coordinates for all three axes; t : [nt].  Returns
    (u, v, w, p) each of shape [nt, n, n, n].
    """
    x4 = x[None, None, None, :]   # varies on the last axis
    y4 = x[None, None, :, None]
    z4 = x[None, :, None, None]
    t4 = t[:, None, None, None]   # time on the first axis
    return es3d_fields(x4, y4, z4, t4, nu=nu)


# --------------------------------------------------------------------------
# Error norms and convergence rates
# --------------------------------------------------------------------------

def l2_error(u_num, u_exact, x):
    """Discrete L2 error via trapezoidal quadrature."""
    err = np.asarray(u_num) - np.asarray(u_exact)
    return np.sqrt(np.trapz(err ** 2, x))


def h1_semi_error(u_num, u_exact, x):
    """Discrete H1 seminorm error of the derivative (central differences)."""
    u_n = np.asarray(u_num)
    u_e = np.asarray(u_exact)
    du = np.diff(u_n) / np.diff(x) - np.diff(u_e) / np.diff(x)
    xm = 0.5 * (x[:-1] + x[1:])
    return np.sqrt(np.trapz(du ** 2, xm))


def convergence_rates(errors, hs):
    """Per-level rates: log(err[k]/err[k+1]) / log(h[k]/h[k+1])."""
    errors = np.asarray(errors, dtype=float)
    hs = np.asarray(hs, dtype=float)
    return np.log(errors[:-1] / errors[1:]) / np.log(hs[:-1] / hs[1:])
