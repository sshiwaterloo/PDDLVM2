"""
burgers_pinnpddlvm.py

Blocks:
    1..8: PDE-only training (similar to a PINN concept, but with random latent variables)
    9..12: Visualization and parameter sweeps
    13..16: Data-driven inverse approach with synthetic sensor data
    17..19: Parameter sweeps, final predictions, PDE solutions at predicted parameters
"""

##########################
# Block 1: Environment Check
##########################
import _bootstrap  # noqa: F401  (repo root on sys.path)
import sys
print(sys.executable)
print(sys.path)

import tensorflow as tf
import tensorflow_probability as tfp
print(tf.__version__)

##########################
# Block 2: GPU + Imports
##########################
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['text.usetex'] = False
mpl.rcParams['axes.unicode_minus'] = False

from pddlvm.utilities import tf as tfpddl, floatf, tffloat
from pddlvm.VEModel import VE
from pddlvm.NeuralNet import NeuralN
from pddlvm.plotting import save_or_show

SHOW = "--show" in sys.argv

tf.random.set_seed(42)

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.set_logical_device_configuration(
        physical_devices[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=6000)]
    )
except:
    pass
print(tf.config.list_physical_devices())

##########################
# Block 3: PDE dims
##########################
dimZ = 1
dimDomain = 2
dimU = 1

##########################
# Block 4: Domain Setup
##########################
nx = 100
nt = 200
xmin, xmax = -1.0, 1.0
tmin, tmax = 0.0, 1.0

tmp_x = tf.linspace(xmin, xmax, nx)
x_vals = tf.cast(tmp_x, tf.float32)
tmp_t = tf.linspace(tmin, tmax, nt)
t_vals = tf.cast(tmp_t, tf.float32)

x_grid, t_grid = tf.meshgrid(x_vals, t_vals)
xm = tf.reshape(x_grid, [-1, 1])
tm = tf.reshape(t_grid, [-1, 1])

icIndices = tf.range(0, nx, delta=1, dtype=tf.int32)

LatentPriorZ = tfp.distributions.Uniform(
    low=tf.constant([0.005], dtype=tf.float32),
    high=tf.constant([0.02], dtype=tf.float32)
)

def init_condition(x):
    """
    Returns the initial condition for Burgers PDE:
    u(x,0) = -sin(pi*x).

    Parameters
    ----------
    x : tf.Tensor
        1D spatial coordinate(s).

    Returns
    -------
    tf.Tensor
        Shape matches x, giving -sin(pi*x).
    """
    return -tf.math.sin(np.pi * x)

##########################
# Block 5: Networks
##########################
U_of_XTZ = NeuralN('U_of_XTZ',
    layout=np.array([dimZ + dimDomain, 100, 100, 100, 100, dimU])
)
K_of_U   = NeuralN('K_of_U',
    layout=np.array([nx*nt, 50, dimZ])
)

##########################
# Block 6: PDE residual
##########################
@tf.function
def reparameterize(mean, logvar):
    """
    Reparameterizes a Gaussian distribution to create a sample.

    Parameters
    ----------
    mean : tf.Tensor
        Mean of shape (...).
    logvar : tf.Tensor
        Log-variance of shape (...).

    Returns
    -------
    tf.Tensor
        Sampled values from the normal distribution with given mean, variance=exp(logvar).
    """
    eps = tf.random.normal(shape=mean.shape, dtype=tf.float32)
    return mean + tf.exp(0.5*logvar)*eps

log2pi = tf.math.log(tf.constant(2.0*np.pi, tf.float32))
def log_normal_pdf(sample, mean, logvar, raxis=1):
    """
    Evaluates log N(sample; mean, exp(logvar)) under a diagonal normal.

    Parameters
    ----------
    sample : tf.Tensor
        Sampled values.
    mean : tf.Tensor
        Means for each entry in sample.
    logvar : tf.Tensor
        Log-variance for each entry in sample.
    raxis : int
        Axis over which to sum.

    Returns
    -------
    tf.Tensor
        Log probability of shape determined by the omitted axis.
    """
    return tf.reduce_sum(
        -0.5*((sample - mean)**2 * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis
    )

@tf.function
def PDE_Residual(z_i):
    """
    Computes PDE residual for 1D Burgers equation with time:
    PDE => u_t + u u_x = nu * u_xx (set by z_i).
    Also applies initial condition at t=0 => u(x,0)=-sin(pi*x).

    Parameters
    ----------
    z_i : tf.Tensor
        Shape (1, dimZ). PDE param in [0.005..0.02].

    Returns
    -------
    us : tf.Tensor
        Shape (nx*nt,1). The reparameterized PDE solution from the forward net.
    residual : tf.Tensor
        Shape (nx*nt + nx,1). PDE residual plus initial condition residual.
    logP_alpha_u : tf.Tensor
        Log probability under the forward net's normal for us.
    """
    with tf.GradientTape(watch_accessed_variables=False) as tape1:
        tape1.watch(xm)
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape2:
            tape2.watch(xm)
            tape2.watch(tm)

            mean_alpha, logvar_alpha = U_of_XTZ.Map([
                xm,
                tm,
                tf.tile(z_i, [xm.shape[0], 1])
            ])
            us = tf.reshape(reparameterize(mean_alpha, logvar_alpha), [-1,1])

        grad_t_u = tape2.gradient(us, tm)
        grad_x_u = tape2.gradient(us, xm)
        grad_x_x_u = tape1.gradient(grad_x_u, xm)

        residualBody = grad_t_u + us * grad_x_u - z_i[0,0]*grad_x_x_u

    us_IC = tf.gather(us, icIndices)
    x_IC  = tf.gather(xm, icIndices)
    resIC = us_IC - init_condition(x_IC)

    residual = tf.concat([residualBody, resIC], axis=0)
    logP_alpha_u = log_normal_pdf(us, mean_alpha, logvar_alpha, raxis=0)
    return us, residual, logP_alpha_u

eps_r = tf.Variable(1e-2, dtype=tf.float32, trainable=False)

##########################
# Block 7: PDE-Only ELBO
##########################
@tf.function
def PDE_ELBO():
    """
    Constructs the evidence lower bound for PDE-only training:
    1) Samples z from a uniform prior in [0.005..0.02].
    2) Reparameterizes PDE solution from forward net => us.
    3) PDE residual => log-likelihood log p(r=0 | us).
    4) Inverse net => log p(z|us).
    5) Summation => PDE-only objective.

    Returns
    -------
    tf.Tensor
        Scalar float representing the PDE-only ELBO.
    """
    z_i = tf.reshape(LatentPriorZ.sample(), [1, dimZ])
    us, residual, logP_alpha_u = PDE_Residual(z_i)

    body_size = nx*nt
    ic_size   = nx
    logVarBody = tf.repeat(2.*tf.math.log(eps_r), body_size)[:,None]
    logVarIC   = tf.repeat(2.*tf.math.log(eps_r/10.), ic_size)[:,None]
    logVarAll  = tf.concat([logVarBody, logVarIC], axis=0)

    logP_r = log_normal_pdf(residual, tf.zeros_like(residual), logVarAll, raxis=0)

    # Inverse net => p_beta(z|u)
    us_flat = tf.reshape(us, [1, -1])
    mean_beta, logvar_beta = K_of_U.Map(us_flat)
    logP_beta_z_u = log_normal_pdf(z_i, mean_beta, logvar_beta, raxis=1)

    elbo = tf.reduce_mean(logP_r + logP_beta_z_u - logP_alpha_u)
    return elbo

##########################
# Block 8: PDE-Only Training
##########################
train_vars_pde = U_of_XTZ.trainable_variables + K_of_U.trainable_variables
optimizer_pde  = tf.keras.optimizers.Adam(1e-3)

@tf.function
def train_step_pde():
    """
    Single PDE-only optimization step:
    Minimizes negative PDE_ELBO.

    Returns
    -------
    tf.Tensor
        Scalar representing the current loss value.
    """
    with tf.GradientTape() as tape:
        loss = -PDE_ELBO()
    grads = tape.gradient(loss, train_vars_pde)
    optimizer_pde.apply_gradients(zip(grads, train_vars_pde))
    return loss

num_iterations_pde = 1000
pde_elbo_log = []
for i in range(num_iterations_pde):
    loss_val = train_step_pde()
    if i % 100 == 0:
        curr_elbo = -loss_val.numpy()
        pde_elbo_log.append(curr_elbo)
        print(f"[PDE-Only] Iter={i}, ELBO={curr_elbo:.4f}")

plt.figure()
plt.plot(np.arange(len(pde_elbo_log))*100, pde_elbo_log, marker='o')
plt.xlabel("Iteration")
plt.ylabel("PDE-Only ELBO")
plt.title("PDE-Only PINN-PDDLVM (Blocks 1..8) Training")
save_or_show(plt.gcf(), "artifacts/burgers_pinnpddlvm_train1.png", show=SHOW)

##########################
# Block 9: Plot PDE solution (Mean, +/-2 sigma) at t=0.5
##########################
z_test = tf.constant([[0.01]], dtype=tf.float32)
t_index = 0.5

x_slice = tf.reshape(x_vals, [-1,1])
t_slice = tf.fill([nx,1], tf.cast(t_index, tf.float32))

mean_slice, logvar_slice = U_of_XTZ.Map([
    x_slice,
    t_slice,
    tf.tile(z_test, [nx,1])
])
us_mean = mean_slice.numpy().flatten()
us_std  = np.exp(0.5*logvar_slice.numpy().flatten())

plt.figure()
plt.plot(x_vals.numpy(), us_mean, 'r-', label='mean')
plt.fill_between(
    x_vals.numpy(),
    us_mean + 2*us_std,
    us_mean - 2*us_std,
    color='r', alpha=0.2,
    label='PINN +/- 2 sigma'
)
plt.xlabel('x')
plt.ylabel(f'u(x,{t_index})')
plt.title("Block 9: PDE-Only solution slice at t=0.5 (ASCII label)")
plt.legend()
save_or_show(plt.gcf(), "artifacts/burgers_pinnpddlvm_slice.png", show=SHOW)

##########################
# Block 10: Save model weights
##########################
saveModelWeights = False
if saveModelWeights:
    U_of_XTZ.NN.save_weights("U_of_XTZ_weights.h5")
    K_of_U.NN.save_weights("K_of_U_weights.h5")
    print("Saved PDE-Only model weights.")

##########################
# Block 11: Param Space Residual
##########################
nZ = 10
tmp_zvals = tf.linspace(tf.cast(0.005, tf.float32), tf.cast(0.02, tf.float32), nZ)
z_vals = tf.cast(tmp_zvals, tf.float32)
residual_map = []
for zv in z_vals:
    z_i = tf.reshape(zv, [1,1])
    _, pdeRes, _ = PDE_Residual(z_i)
    mseRes = tf.reduce_mean(tf.square(pdeRes))
    residual_map.append(mseRes.numpy())

plt.figure()
plt.plot(z_vals.numpy(), residual_map, marker='o')
plt.xlabel('nu')
plt.ylabel('PDE residual MSE')
plt.title("Block 11: PDE Residual vs. param nu")
save_or_show(plt.gcf(), "artifacts/burgers_pinnpddlvm_residual.png", show=SHOW)

##########################
# Block 12: 2D Heatmap PDE solution
##########################
z_fixed = tf.constant([[0.01]], dtype=tf.float32)
mean_all, logvar_all = U_of_XTZ.Map([
    xm, tm, tf.tile(z_fixed, [xm.shape[0],1])
])
us_mean_2d = mean_all.numpy().reshape(nt, nx)
us_std_2d  = np.exp(0.5*logvar_all.numpy().reshape(nt,nx))

plt.figure(figsize=(6,4))
plt.contourf(x_vals.numpy(), t_vals.numpy(), us_mean_2d, 100, cmap='jet')
plt.colorbar()
plt.title("Block 12: PDE solution mean over (x,t)")
plt.xlabel('x')
plt.ylabel('t')
save_or_show(plt.gcf(), "artifacts/burgers_pinnpddlvm_mean.png", show=SHOW)

##########################
# Block 13: Data-Driven Extension
##########################
x_sensors = np.linspace(-1, 1, 10, dtype=np.float32)
t_sensors = np.linspace(0, 1, 5, dtype=np.float32)
Xs, Ts = np.meshgrid(x_sensors, t_sensors)
Xs_flat = Xs.flatten()
Ts_flat = Ts.flatten()
Nsensors = Xs_flat.shape[0]

z_data = 0.012
z_data_tf = tf.constant([[z_data]], dtype=tf.float32)
Xs_tf = tf.reshape(tf.constant(Xs_flat, dtype=tf.float32), [-1,1])
Ts_tf = tf.reshape(tf.constant(Ts_flat, dtype=tf.float32), [-1,1])

mean_truth, _ = U_of_XTZ.Map([
    Xs_tf, Ts_tf, tf.tile(z_data_tf, [Nsensors,1])
])
u_truth = mean_truth.numpy().flatten()

noise_std = 0.01
y_data = (u_truth + np.random.normal(0, noise_std, size=(Nsensors,))).astype(np.float32)

##########################
# Block 14: Observation Model
##########################
dimObs = Nsensors
dimSol = nx*nt

Enc_of_y = NeuralN('Enc_of_y', np.array([dimObs, 100, 100, dimSol]))

@tf.function
def pickSensors(u_tensor):
    """
    Returns the first Nsensors entries from the PDE solution vector
    to simulate sensor picks.

    Parameters
    ----------
    u_tensor : tf.Tensor
        Shape (batch, Nx*nt). Flattened PDE solutions.

    Returns
    -------
    tf.Tensor
        Shape (batch, Nsensors). Subset used as sensor data.
    """
    sensor_0 = u_tensor[:, :Nsensors]
    return sensor_0

noise_logvar = tf.constant(2.0*np.log(noise_std), dtype=tf.float32)
@tf.function
def logp_y_given_u(y_data_tf, u_tensor):
    """
    Observation likelihood p(y|u). Takes sensor picks from u_tensor
    and compares with y_data_tf under a diagonal Gaussian with variance noise_std^2.

    Parameters
    ----------
    y_data_tf : tf.Tensor
        Shape (Nsensors,). The observed data.
    u_tensor : tf.Tensor
        Shape (batch, Nx*nt). Flattened PDE solutions.

    Returns
    -------
    tf.Tensor
        Shape (batch,). Log-likelihood of y_data under the predicted sensor picks.
    """
    sensors_pred = pickSensors(u_tensor)
    logp = log_normal_pdf(
        sample=y_data_tf[None, :],
        mean=sensors_pred,
        logvar=noise_logvar*tf.ones_like(sensors_pred),
        raxis=1
    )
    return logp

##########################
# Block 15: Data-Driven ELBO
##########################
@tf.function
def Data_ELBO(y_data_tf):
    """
    Constructs data-driven ELBO ignoring PDE residual:
    1) Sample z from prior.
    2) Use encoder net to map y_data -> u_samp.
    3) Evaluate inverse net -> z distribution, prior on z.
    4) Evaluate p(y|u_samp).
    5) Combine => data-only objective.

    Parameters
    ----------
    y_data_tf : tf.Tensor
        Shape (Nsensors,). Observed sensor data.

    Returns
    -------
    tf.Tensor
        Scalar float for the data-based ELBO.
    """
    z_i = tf.reshape(LatentPriorZ.sample(), [1, dimZ])
    mean_enc, logvar_enc = Enc_of_y.Map(y_data_tf[None, :])
    u_samp = reparameterize(mean_enc, logvar_enc)

    mean_beta, logvar_beta = K_of_U.Map(u_samp)
    logP_beta_z_u = log_normal_pdf(z_i, mean_beta, logvar_beta, raxis=1)

    logP_data = logp_y_given_u(y_data_tf, u_samp)
    log_q_enc = log_normal_pdf(u_samp, mean_enc, logvar_enc, raxis=1)

    elbo_data = tf.reduce_mean(logP_data + logP_beta_z_u - log_q_enc)
    return elbo_data

##########################
# Block 16: Data-Driven Training
##########################
train_vars_data = Enc_of_y.trainable_variables + K_of_U.trainable_variables
optimizer_data = tf.keras.optimizers.Adam(1e-3)

y_data_tf = tf.constant(y_data, dtype=tf.float32)

@tf.function
def train_step_data():
    """
    Single training step for data-based approach:
    Minimizes negative Data_ELBO.

    Returns
    -------
    tf.Tensor
        Scalar float for the current data-based loss.
    """
    with tf.GradientTape() as tape:
        loss = -Data_ELBO(y_data_tf)
    grads = tape.gradient(loss, train_vars_data)
    optimizer_data.apply_gradients(zip(grads, train_vars_data))
    return loss

num_iterations_data = 1000
data_elbo_list = []
for i in range(num_iterations_data):
    loss_val = train_step_data()
    if i % 100 == 0:
        curr_elbo = -loss_val.numpy()
        data_elbo_list.append(curr_elbo)
        print(f"[Data-Driven] Iter={i}, ELBO={curr_elbo:.4f}")

plt.figure()
plt.plot(np.arange(len(data_elbo_list))*100, data_elbo_list, marker='o')
plt.xlabel("Iteration")
plt.ylabel("Data-Driven ELBO")
plt.title("Block 16: Data-Driven approach (observation to param) Training")
save_or_show(plt.gcf(), "artifacts/burgers_pinnpddlvm_train2.png", show=SHOW)

##########################
# Block 17: Param Sweep
##########################
nSamples = 10
zPreds = []
for _ in range(nSamples):
    mean_enc, logvar_enc = Enc_of_y.Map(y_data_tf[None,:])
    u_samp = reparameterize(mean_enc, logvar_enc)
    mean_b, logvar_b = K_of_U.Map(u_samp)
    zPred = mean_b.numpy().flatten()
    zPreds.append(zPred[0])

zPred_mean = np.mean(zPreds)
zPred_std  = np.std(zPreds)
print(f"True z_data={z_data}, Pred z={zPred_mean:.4f}±(2*{zPred_std:.4f})")

##########################
# Block 18: Compare PDE solution at predicted z
##########################
z_pred_tf = tf.constant([[zPred_mean]], dtype=tf.float32)

ti = 0.5
t_slice_ = tf.fill([nx,1], tf.cast(ti, tf.float32))
xm_slice_ = tf.reshape(x_vals, [-1,1])

m_pred, _ = U_of_XTZ.Map([
    xm_slice_,
    t_slice_,
    tf.tile(z_pred_tf, [nx,1])
])

z_data_tf2 = tf.constant([[z_data]], dtype=tf.float32)
m_true, _  = U_of_XTZ.Map([
    xm_slice_,
    t_slice_,
    tf.tile(z_data_tf2, [nx,1])
])

plt.figure()
plt.plot(x_vals.numpy(), m_true.numpy().flatten(), 'k--', label='Solution at z_data')
plt.plot(x_vals.numpy(), m_pred.numpy().flatten(), 'r-', label='Solution at predicted z')
plt.xlabel("x")
plt.ylabel("u(x,0.5)")
plt.legend()
plt.title("Block 18: PDE solutions at predicted param vs. true param")
save_or_show(plt.gcf(), "artifacts/burgers_pinnpddlvm_compare.png", show=SHOW)