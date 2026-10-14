"""Taylor-Green vortex and turbulence diagnostics.

The Taylor-Green vortex is the canonical three-dimensional
"turbulence-like" benchmark: a smooth periodic initial condition whose
nonlinear evolution transfers energy from large to small scales.  It has
no closed-form solution, so ground truth comes from a converged spectral
solver (pddlvm.numerics.ns3d_spectral) validated by exact conservation
laws.
"""

import numpy as np


def tgv_initial(n, L=2.0 * np.pi):
    """Taylor-Green initial condition u = (sin x cos y cos z,
    -cos x sin y cos z, 0) on [0, L]^3.  Returns (u0, v0, w0) [n, n, n]."""
    x = np.linspace(0.0, L, n, endpoint=False)
    xg = x[None, None, :]
    yg = x[None, :, None]
    zg = x[:, None, None]
    u0 = np.sin(xg) * np.cos(yg) * np.cos(zg)
    v0 = -np.cos(xg) * np.sin(yg) * np.cos(zg)
    w0 = np.zeros_like(u0)
    return u0, v0, w0


def kinetic_energy(u, v, w):
    """E = 0.5 * <|u|^2> (volume mean)."""
    return 0.5 * np.mean(u ** 2 + v ** 2 + w ** 2)


def enstrophy(u, v, w, L=2.0 * np.pi):
    """Z = 0.5 * <|curl u|^2>, computed spectrally (exact for band-limited
    fields)."""
    n = u.shape[0]
    k = np.fft.fftfreq(n, d=L / n) * 2.0 * np.pi
    kx = k[None, None, :]
    ky = k[None, :, None]
    kz = k[:, None, None]
    fu = np.fft.fftn(u)
    fv = np.fft.fftn(v)
    fw = np.fft.fftn(w)
    # curl in Fourier space
    wx = 1j * (ky * fw - kz * fv)
    wy = 1j * (kz * fu - kx * fw)
    wz = 1j * (kx * fv - ky * fu)
    return 0.5 * np.mean(
        np.abs(np.fft.ifftn(wx)) ** 2 + np.abs(np.fft.ifftn(wy)) ** 2
        + np.abs(np.fft.ifftn(wz)) ** 2)


def energy_spectrum(u, v, w, L=2.0 * np.pi):
    """Shell-averaged kinetic energy spectrum E(k).

    Returns (k, E) where k are integer shell radii and
    sum_k E(k) dk ~ E_total.
    """
    n = u.shape[0]
    fu = np.fft.fftn(u) / n ** 3
    fv = np.fft.fftn(v) / n ** 3
    fw = np.fft.fftn(w) / n ** 3
    kf = np.fft.fftfreq(n, d=1.0 / n)
    kx = kf[None, None, :]
    ky = kf[None, :, None]
    kz = kf[:, None, None]
    krad = np.sqrt((kx ** 2 + ky ** 2 + kz ** 2)).astype(int)
    e = 0.5 * (np.abs(fu) ** 2 + np.abs(fv) ** 2 + np.abs(fw) ** 2)
    kmax = krad.max()
    spec = np.zeros(kmax + 1)
    for k in range(1, kmax + 1):
        spec[k] = e[krad == k].sum()
    return np.arange(kmax + 1), spec
