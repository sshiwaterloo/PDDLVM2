"""Vectorized 1D linear finite-element assembly for -d/dx(kappa du/dx) = f.

The legacy ``poisson1DSolver.assemble_tf_internal`` looped over elements with
``tf.tensor_scatter_nd_add`` inside a Python ``for`` loop.  This module builds
the same operators with a single ``tf.scatter_nd`` per operator (duplicate
indices accumulate), so the whole assembly is one vectorized graph op that is
differentiable and ``tf.function``-friendly.

Semantics (matching the legacy solver exactly)
----------------------------------------------
- Element stiffness: ``kappa_e / h_e * [[1, -1], [-1, 1]]`` scattered to the
  four element-node pairs.
- Consistent load vector: ``h_e * (f_l/3 + f_r/6)`` at the left node and
  ``h_e * (f_l/6 + f_r/3)`` at the right node.
- Dirichlet elimination at both ends: ``f -= bc_l * A[:, 0]`` and
  ``f -= bc_r * A[:, -1]`` using the *uneliminated* columns, then
  ``f[0] = bc_l, f[-1] = bc_r``, rows/columns 0 and N-1 zeroed, and unit
  diagonals restored at the boundary nodes.
"""

import tensorflow as tf


def assemble_poisson_1d(x, kappa_e, f_nodes, bc_left, bc_right,
                        dtype=tf.float32):
    """Assemble (A, f) for -d/dx(kappa du/dx) = f with Dirichlet BCs.

    Parameters
    ----------
    x : tensor [N]
        Mesh nodes (monotonic, any spacing).
    kappa_e : tensor [N-1]
        Diffusivity at element centers (or any per-element effective value;
        nonlinear stress-strain laws are applied by the caller).
    f_nodes : tensor [N]
        Source sampled at mesh nodes.
    bc_left, bc_right : scalars
        Dirichlet values at x[0] and x[-1].

    Returns
    -------
    (A, f) : tensor [N, N], tensor [N]
        The Dirichlet-eliminated system ``A u = f``.
    """
    x = tf.convert_to_tensor(x, dtype=dtype)
    kappa_e = tf.convert_to_tensor(kappa_e, dtype=dtype)
    f_nodes = tf.convert_to_tensor(f_nodes, dtype=dtype)
    # convert_to_tensor (not tf.cast) so python scalars are converted
    # directly to the requested dtype instead of via float32
    bc_left = tf.convert_to_tensor(bc_left, dtype=dtype)
    bc_right = tf.convert_to_tensor(bc_right, dtype=dtype)

    n = tf.shape(x)[0]
    e = n - 1
    idx = tf.range(e)
    h = x[1:] - x[:-1]
    k_h = kappa_e / h

    # ---- stiffness: single scatter over 4 contributions per element ----
    rows = tf.concat([idx, idx + 1, idx, idx + 1], axis=0)
    cols = tf.concat([idx, idx, idx + 1, idx + 1], axis=0)
    vals = tf.concat([k_h, -k_h, -k_h, k_h], axis=0)
    a_mat = tf.scatter_nd(tf.stack([rows, cols], axis=1), vals, [n, n])

    # ---- consistent load vector ----
    f_l, f_r = f_nodes[:-1], f_nodes[1:]
    load_l = h * (f_l / 3.0 + f_r / 6.0)
    load_r = h * (f_l / 6.0 + f_r / 3.0)
    node_idx = tf.reshape(tf.concat([idx, idx + 1], axis=0), [-1, 1])
    f_vec = tf.scatter_nd(node_idx, tf.concat([load_l, load_r], axis=0), [n])

    return apply_dirichlet(a_mat, f_vec, bc_left, bc_right, dtype)


def apply_dirichlet(a_mat, f_vec, bc_left, bc_right, dtype=tf.float32):
    """Dirichlet elimination at both endpoints (see module docstring)."""
    n = tf.shape(a_mat)[0]
    zero_row = tf.zeros([2, n], dtype=dtype)
    one_diag = tf.ones([2], dtype=dtype)

    # f -= bc_l * A[:, 0]; f -= bc_r * A[:, -1]  (uneliminated columns)
    f_vec = f_vec - bc_left * a_mat[:, 0] - bc_right * a_mat[:, n - 1]
    f_vec = tf.tensor_scatter_nd_update(
        f_vec, [[0], [n - 1]], tf.stack([bc_left, bc_right]))

    a_mat = tf.tensor_scatter_nd_update(a_mat, [[0], [n - 1]], zero_row)
    a_mat = tf.transpose(
        tf.tensor_scatter_nd_update(tf.transpose(a_mat), [[0], [n - 1]],
                                    zero_row))
    a_mat = tf.tensor_scatter_nd_update(
        a_mat, [[0, 0], [n - 1, n - 1]], one_diag)
    return a_mat, f_vec


def assemble_poisson_1d_banded(x, kappa_e, f_nodes, bc_left, bc_right,
                                dtype=tf.float32):
    """Tridiagonal (banded) assembly of the same system, O(n) memory.

    For 1D linear elements the Dirichlet-eliminated stiffness matrix is
    tridiagonal, so large meshes should never materialize a dense matrix
    (``assemble_poisson_1d`` returns a dense [N, N] matrix and is intended
    for small systems / the training graph, where N ~ 20).

    Returns
    -------
    (sub, diag, sup, f_vec) : banded storage in natural indexing
        (``sub[i] = A[i, i-1]``, ``sup[i] = A[i, i+1]``).  For
        ``scipy.linalg.solve_banded((1, 1), ab, f)`` build
        ``ab = [pad(sup), diag, pad(sub)]``: scipy row 0 stores
        ``A[j-1, j]`` (left-shifted, leading pad) and row 2 stores
        ``A[j+1, j]`` (right-shifted, trailing pad).
    """
    x = tf.convert_to_tensor(x, dtype=dtype)
    kappa_e = tf.convert_to_tensor(kappa_e, dtype=dtype)
    f_nodes = tf.convert_to_tensor(f_nodes, dtype=dtype)
    # see note in assemble_poisson_1d about tf.cast vs convert_to_tensor
    bc_left = tf.convert_to_tensor(bc_left, dtype=dtype)
    bc_right = tf.convert_to_tensor(bc_right, dtype=dtype)

    h = x[1:] - x[:-1]
    k_h = kappa_e / h  # [E], k_h[e] = kappa_e / h_e
    n = tf.shape(x)[0]

    # eliminated tridiagonal structure (see module docstring):
    #   sub[i] = -k_h[i-1] for i = 2..N-2, zero elsewhere (cols 0/N-1 removed)
    #   diag[0] = diag[N-1] = 1, interior diag = k_h[i-1] + k_h[i]
    #   sup[i] = -k_h[i] for i = 1..N-3, zero at 0 and N-2 (cols 0/N-1 removed)
    sub = tf.concat([tf.zeros([2], dtype), -k_h[1:-1], tf.zeros([1], dtype)],
                    axis=0)
    interior_diag = k_h[:-1] + k_h[1:]  # nodes 1..N-2
    diag = tf.concat([tf.ones([1], dtype), interior_diag,
                      tf.ones([1], dtype)], axis=0)
    sup = tf.concat([tf.zeros([1], dtype), -k_h[1:-1],
                     tf.zeros([2], dtype)], axis=0)

    # consistent load vector
    f_l, f_r = f_nodes[:-1], f_nodes[1:]
    load_l = h * (f_l / 3.0 + f_r / 6.0)
    load_r = h * (f_l / 6.0 + f_r / 3.0)
    f_mid = load_r[:-1] + load_l[1:]  # nodes 1..N-2
    f_vec = tf.concat([tf.zeros([1], dtype), f_mid, tf.zeros([1], dtype)],
                      axis=0)

    # Dirichlet elimination: absorb boundary columns into the load, then
    # overwrite the boundary rows with the Dirichlet values.
    f_vec = tf.tensor_scatter_nd_add(f_vec, [[1]], [bc_left * k_h[0]])
    f_vec = tf.tensor_scatter_nd_add(f_vec, [[n - 2]],
                                     [bc_right * k_h[-1]])
    f_vec = tf.tensor_scatter_nd_update(
        f_vec, [[0], [n - 1]], tf.stack([bc_left, bc_right]))
    return sub, diag, sup, f_vec


def residual_poisson_1d(a_mat, f_vec, u, dtype=tf.float32):
    """Residual A u - f for an assembled system."""
    u = tf.convert_to_tensor(u, dtype=dtype)
    return tf.squeeze(a_mat @ u[:, None] - f_vec[:, None])


def solve_poisson_1d(x, kappa_e, f_nodes, bc_left, bc_right,
                     dtype=tf.float32):
    """Assemble and solve the Dirichlet Poisson problem. Returns u at nodes."""
    a_mat, f_vec = assemble_poisson_1d(x, kappa_e, f_nodes, bc_left, bc_right,
                                       dtype)
    return tf.squeeze(tf.linalg.solve(a_mat, f_vec[:, None]))
