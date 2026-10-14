"""Chebyshev polynomial utilities.

Pure TensorFlow ops, no graph side effects, no imports of pddlvm.utilities
(that module mutates matplotlib rcParams at import time).

Conventions
-----------
- ``coeffs`` tensors have shape ``[n, numFuncs]``: ``n`` Chebyshev coefficients
  along axis 0, one or more functions along axis 1.
- ``x`` is a rank-1 tensor of evaluation points in the *physical* domain
  ``[lo, hi]`` (default [-1, 1]).
- Matrices ``T`` have shape ``[n, len(x)]`` so that
  ``values = matmul(T, coeffs, transpose_a=True)`` gives shape
  ``[len(x), numFuncs]``, matching the legacy ``PDE_State`` convention.
"""

import numpy as np
import tensorflow as tf


def cheb_T_mat(n: int, x: tf.Tensor, lo: float = -1.0, hi: float = 1.0,
               dtype: tf.DType = tf.float32) -> tf.Tensor:
    """T_{j}(xi) matrix, shape [n, len(x)].

    ``xi = -1 + 2 (x - lo) / (hi - lo)`` maps the physical domain onto [-1, 1],
    where the Chebyshev polynomials are defined via the recurrence
    ``T_{j+1} = 2 xi T_j - T_{j-1}``.
    """
    x = tf.convert_to_tensor(x, dtype=dtype)
    # convert_to_tensor (not tf.cast): python floats must not round-trip
    # through float32 on the way to float64
    lo = tf.convert_to_tensor(lo, dtype=dtype)
    hi = tf.convert_to_tensor(hi, dtype=dtype)
    xi = -1.0 + 2.0 * (x - lo) / (hi - lo)
    rows = [tf.ones_like(xi)]
    if n > 1:
        rows.append(xi)
        for _ in range(2, n):
            rows.append(2.0 * xi * rows[-1] - rows[-2])
    return tf.stack(rows[:n], axis=0)


def cheb_eval(coeffs: tf.Tensor, x: tf.Tensor, lo: float = -1.0, hi: float = 1.0,
              dtype: tf.DType = tf.float32) -> tf.Tensor:
    """Evaluate Chebyshev series. coeffs [n, numFuncs] -> values [len(x), numFuncs]."""
    coeffs = tf.convert_to_tensor(coeffs, dtype=dtype)
    T = cheb_T_mat(tf.shape(coeffs)[0], x, lo, hi, dtype)
    return tf.matmul(T, coeffs, transpose_a=True)


def cheb_fit(x: tf.Tensor, values: tf.Tensor, n: int,
             dtype: tf.DType = tf.float32) -> tf.Tensor:
    """Least-squares Chebyshev fit of values at points x -> coeffs [n, numFuncs].

    ``values`` may be rank 1 (a single function) or rank 2
    ([len(x), numFuncs]); the returned coefficients are squeezed to rank 1
    for single-function input.
    """
    x = tf.convert_to_tensor(x, dtype=dtype)
    values = tf.convert_to_tensor(values, dtype=dtype)
    if values.shape.rank == 1:
        values = values[:, None]
    T = cheb_T_mat(n, x, dtype=dtype)
    # solve T coeffs = values  <=>  T^T coeffs^T = values^T;  the system
    # matrix for lstsq is [len(x), n] = transpose(T)
    return tf.squeeze(tf.linalg.lstsq(tf.transpose(T), values, fast=True),
                      axis=-1)


def cheb_integrate(coeffs: tf.Tensor, dtype: tf.DType = tf.float32) -> tf.Tensor:
    """Integrate a Chebyshev series (single function, coeffs [n]) -> [n + 1].

    The free integration constant is set to zero; callers fix it through
    boundary conditions. Uses int T_0 = T_1, int T_1 = (T_2 + T_0)/4 and
    int T_j = T_{j+1}/(2(j+1)) - T_{j-1}/(2(j-1)) for j >= 2.
    """
    coeffs = tf.convert_to_tensor(coeffs, dtype=dtype)
    n = int(coeffs.shape[0])
    rows: list = []
    cols: list = []
    vals: list = []
    # int T_0 = T_1
    rows += [1]
    cols += [0]
    vals += [1.0]
    if n > 1:
        # int T_1 = (T_2 + T_0) / 4
        rows += [0, 2]
        cols += [1, 1]
        vals += [0.25, 0.25]
    for j in range(2, n):
        rows += [j - 1, j + 1]
        cols += [j, j]
        vals += [-1.0 / (2.0 * (j - 1)), 1.0 / (2.0 * (j + 1))]
    out = tf.zeros([n + 1], dtype=dtype)
    if rows:
        idx = tf.stack([tf.constant(rows)], axis=1)  # [k, 1] for 1D output
        # tf.constant(..., dtype=dtype) converts python floats directly;
        # a tf.cast around a default (float32) constant would corrupt the
        # 1/(2j) coefficients at the 1e-8 level
        out = tf.tensor_scatter_nd_add(
            out, idx,
            tf.constant(vals, dtype=dtype) * tf.gather(coeffs, cols))
    return out


def poisson_const_kappa(f_coeffs: tf.Tensor, kappa: float,
                        bc_left: float, bc_right: float,
                        lo: float = -1.0, hi: float = 1.0,
                        dtype: tf.DType = tf.float64) -> tf.Tensor:
    """Solve -kappa u'' = f with Dirichlet BCs, in Chebyshev coefficient space.

    ``f_coeffs``: Chebyshev coefficients of the source on the physical domain
    [lo, hi]. The solution is obtained by double integration in coefficient
    space (spectral accuracy for smooth f); the two integration constants are
    fixed analytically from the boundary values. Returns solution coefficients
    with the same degree as the input.
    """
    f_coeffs = tf.convert_to_tensor(f_coeffs, dtype=dtype)
    kappa = tf.convert_to_tensor(kappa, dtype=dtype)
    scale = tf.convert_to_tensor((hi - lo) ** 2 / 4.0, dtype=dtype)
    # d2u/dx2 = (4/L^2) d2u/dxi2  =>  u'' coefficients on [-1, 1]:
    c2 = -scale / kappa * f_coeffs  # coefficients of u''
    c1 = cheb_integrate(c2, dtype)  # u', free constant at index 0
    c0 = cheb_integrate(c1, dtype)  # u, two free constants

    # Contributions of the two free constants: c0[0] = C0 (T_0), c1[0] = C1
    # (integrating C1 T_0 gives C1 T_1 = C1 xi).  So
    #   u(-1) = base(-1) + C0 - C1,  u(+1) = base(+1) + C0 + C1.
    # T_j(-1) = (-1)^j and T_j(+1) = 1.
    def value_at_xi(c, xi):
        signs = tf.cast([xi ** j for j in range(c.shape[0])], dtype)
        return tf.reduce_sum(c * signs)

    base = tf.identity(c0)
    base = tf.tensor_scatter_nd_update(base, [[0]], [tf.cast(0.0, dtype)])
    b_m = value_at_xi(base, -1.0)
    b_p = value_at_xi(base, +1.0)
    C0 = (bc_left + bc_right - b_m - b_p) / 2.0
    C1 = (bc_right - bc_left - b_p + b_m) / 2.0
    out = tf.tensor_scatter_nd_update(base, [[0], [1]],
                                      tf.stack([C0, base[1] + C1]))
    return out


def poisson_const_kappa_values(f_coeffs: tf.Tensor, x: tf.Tensor, kappa: float,
                               bc_left: float, bc_right: float,
                               lo: float = -1.0, hi: float = 1.0,
                               dtype: tf.DType = tf.float64) -> tf.Tensor:
    """Solve -kappa u'' = f and evaluate the solution at points x."""
    u_coeffs = poisson_const_kappa(f_coeffs, kappa, bc_left, bc_right,
                                   lo, hi, dtype)
    return tf.squeeze(cheb_eval(u_coeffs[:, None], x, lo, hi, dtype))
