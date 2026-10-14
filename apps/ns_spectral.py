# ns_spectral.py
# --------------------------------------------------------
# A pseudo-spectral solver for 2D Navier–Stokes in vorticity form on [0,1]^2, periodic BC.
# Allow to specify a time snapshot t_snap (<= tmax), so we return w(x,y,t_snap) and u(x,y,t_snap).

import _bootstrap  # noqa: F401  (repo root on sys.path)

import numpy as np

def ns_spectral_solve(nx=32, ny=32, nu=0.01, tmax=1.0, dt=0.001,
                      t_snap=None,
                      w0=None,
                      forcing=lambda x, y: 0.1*np.sin(2.*np.pi*(x+y)) + np.cos(2.*np.pi*(x+y)),
                      return_velocity=False):
    """
    Pseudo-spectral solver for the 2D vorticity eq on [0,1]^2 (periodic).
    PDE: w_t + u·∇w = nu ∆ w + f(x),   ∇·u=0,   w(.,0)= w0
    We'll do a 2-step (Heun) method in time.
    We only store and return the solution at t=t_snap (or at tmax if t_snap is None or > tmax).

    Args:
      nx, ny: grid size in x,y
      nu: viscosity
      tmax: final integration time
      dt: time step
      t_snap: the time at which to output w(x,y,t_snap). If t_snap is None or > tmax, we do tmax.
      w0: initial vorticity array, shape=(ny,nx), or None => zero
      forcing: function f(x,y)
      return_velocity: if True, also compute velocity (u_x, u_y) at that snapshot.

    Returns:
      w_snapshot: shape=(ny,nx), the vorticity at t_snap (or tmax).
      if return_velocity=True, also returns (ux_snapshot, uy_snapshot) each shape=(ny,nx).
    """

    if t_snap is None or t_snap > tmax:
        t_snap = tmax

    x_1d = np.linspace(0., 1., nx, endpoint=False)
    y_1d = np.linspace(0., 1., ny, endpoint=False)
    X, Y = np.meshgrid(x_1d, y_1d)

    if w0 is None:
        w0 = np.zeros((ny, nx), dtype=np.float64)
    w = w0.copy()

    # wave numbers
    kx_1d = np.fft.fftfreq(nx, d=1./nx)
    ky_1d = np.fft.fftfreq(ny, d=1./ny)
    kx, ky = np.meshgrid(kx_1d, ky_1d)
    k2 = kx**2 + ky**2
    k2[0,0] = 1e-14  # avoid dividing by zero

    # real-space forcing
    fxy = forcing(X, Y)

    def velocity_from_vorticity(w_hat):
        # w = ∆ψ => w_hat = -k^2 ψ_hat => => ψ_hat = - w_hat / k^2
        psi_hat = - w_hat / k2
        # velocity => u= ∇^⊥ ψ => ( -∂ψ/∂y, ∂ψ/∂x )
        # => in Fourier => uhat_x= i ky psi_hat, uhat_y= -i kx psi_hat
        uhat_x = 1j* ky * psi_hat
        uhat_y = -1j* kx * psi_hat
        ux = np.fft.ifft2(uhat_x).real
        uy = np.fft.ifft2(uhat_y).real
        return ux, uy

    def compute_rhs(w_):
        """
        PDE => w_t = - (u·∇w) + nu ∆ w + f
        We'll do a 2-step (Heun).
        """
        w_hat_ = np.fft.fft2(w_)
        ux_, uy_ = velocity_from_vorticity(w_hat_)
        # advection => - u·∇w
        dw_dx_ = np.gradient(w_, axis=1)*nx
        dw_dy_ = np.gradient(w_, axis=0)*ny
        adv_   = - (ux_*dw_dx_ + uy_*dw_dy_)

        # diffusion => -nu k^2 w_hat_
        diff_hat= - nu*k2*w_hat_
        diff_ = np.fft.ifft2(diff_hat).real

        return adv_ + diff_ + fxy

    # integrate in time, but only up to t_snap
    steps = int(np.ceil(t_snap/dt))
    # if sets t_snap < tmax, we only do partial integration
    # or if t_snap==tmax, we do full integration
    current_time= 0.
    for _ in range(steps):
        # Heun's method:
        k1= compute_rhs(w)
        w1= w + dt*k1
        k2= compute_rhs(w1)
        w = w + 0.5*dt*(k1 + k2)
        current_time += dt

    w_snapshot = w
    if not return_velocity:
        return w_snapshot
    else:
        w_hat_snap = np.fft.fft2(w_snapshot)
        ux_snap, uy_snap = velocity_from_vorticity(w_hat_snap)
        return w_snapshot, ux_snap, uy_snap


if __name__=="__main__":
    # quick test => Nx=32, tmax=1.0, t_snap=0.2
    Nx=32
    Ny=32
    nu=0.01
    dt=0.001
    tmax=1.0
    t_snap=0.2
    w_final, ux_final, uy_final= ns_spectral_solve(
        nx=Nx, ny=Ny,
        nu=nu,
        tmax=tmax,
        dt=dt,
        t_snap=t_snap,
        w0=None,
        return_velocity=True
    )
    import matplotlib.pyplot as plt
    plt.contourf(w_final, 50, cmap='jet')
    plt.colorbar()
    plt.title(f"ns_spectral test run w(x,y) at t={t_snap}")
    from pddlvm.plotting import save_or_show, show_requested
    save_or_show(plt.gcf(), "artifacts/ns_spectral.png",
                 show=show_requested())