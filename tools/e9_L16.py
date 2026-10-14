"""L=16 domain: does doubling the box change the selected mode? (recycling test)"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from pddlvm.numerics import ns2d_spectral as S
from pddlvm.numerics import obstacles as O

n = 512
L = 16.0
nu = 1.0 / 100.0
dt = 0.016
cx = cy = L / 2
eps, width = 0.01, 0.0625
x = (np.arange(n) + 0.5) * L / n
X, Y = np.meshgrid(x, x)
u0 = np.full((n, n), 1.0)
v0 = 0.05 * np.sin(2 * np.pi * X / L) * np.exp(
    -((X - cx) ** 2 + (Y - cy) ** 2) / 0.2)
lam, mask = O.cylinder_field(n, n, L, L, cx, cy, 1.0, width, eps)
u, v, _, probes = S.spectral_solve_2d(
    u0, v0, nu=nu, dt=dt, n_steps=10000, Lx=L, Ly=L,
    U_target=1.0, penalty=lam, diameter=1.0, probe_every=5)
t = np.array([p[0] * dt for p in probes])
cl = np.array([p[2] for p in probes])
cd = np.array([p[1] for p in probes])
for t0, t1 in [(40, 80), (80, 120), (120, 160)]:
    sel = (t >= t0) & (t < t1)
    print(f"t {t0}-{t1}: Cd={cd[sel].mean():.3f} "
          f"Cl_rms={np.sqrt(np.mean((cl[sel]-cl[sel].mean())**2)):.4f}")
sel = t >= 60
clw = (cl[sel] - cl[sel].mean()) * np.hanning(sel.sum())
spec = np.abs(np.fft.rfft(clw, n=1 << 20)) ** 2
freq = np.fft.rfftfreq(1 << 20, d=dt)
f_peak = freq[np.argmax(spec)]
print(f"f_peak={f_peak:.4f} (St={f_peak:.4f}) Cd_late={cd[t>=60].mean():.3f}")
np.save(".tmp/e9_L16_cl.npy", cl)
np.save(".tmp/e9_L16_t.npy", t)
