"""Variational Emulator (legacy-compatible facade over the probabilistic layer).

The public API is unchanged from the original release (notebooks in
``reference/`` set ``model.Latent*Prior`` attributes, call
``initPriorNetworks`` / ``initPostNetworks`` / ``setModePrior`` /
``compute_nelbo_prior`` etc.), but the internals now delegate to
``pddlvm.probabilistic``:

- log-densities and reparameterization: ``distributions``
- physics residual likelihood: ``elbo.physics_gaussian_log_likelihood``
- per-component ELBO records: ``ElboTerms`` (use ``elbo_terms_prior`` /
  ``elbo_terms_post``; the legacy ``compute_nelbo_*`` methods return only
  the scalar NELBO)

Robustness fixes vs the original:
- explicit errors when priors / data are not configured (instead of
  failing deep inside a sample call),
- forcing-prior sampling via ``tfp.distributions.Normal`` instead of a
  ``tf.random.normal`` call with mean/stddev arguments,
- ``compute_nelbo_post`` requires rank-2 ``dataY`` (the legacy default
  ``tffloat([0.])`` placeholder is rejected with a clear message).
"""

import tensorflow_probability as tfp

from .utilities import *
from .NeuralNet import NeuralN
from .probabilistic.distributions import (
    diag_gaussian_log_prob,
    reparameterize as _reparameterize,
)
from .probabilistic.elbo import (
    combine_terms,
    physics_gaussian_log_likelihood,
)

_0 = tf.constant(0., dtype=floatf)
tfd = tfp.distributions

_PRIOR_DIST_ATTRS = ('LatentKappaPrior', 'LatentForcingPrior',
                     'LatentBlPrior', 'LatentBrPrior')


class VE():
    """Variational Emulator."""

    def __init__(self, pdeState, solver):
        self.pdeState = pdeState
        self.solver = solver
        self.dimU = self.pdeState.dimU
        self.dimK = self.pdeState.dimK
        self.dimF = self.pdeState.dimF
        self.dimBl = 1
        self.dimBr = 1
        self.epsilon_r = tf.Variable(tf.constant(1e-2, dtype=floatf),
                                     trainable=False)
        self.epsilon_y = tf.Variable(tf.constant(1e-1, dtype=floatf),
                                     trainable=False)

        self.LatentKappaPrior = None
        self.LatentForcingPrior = None
        self.LatentBlPrior = None
        self.LatentBrPrior = None

        self.dimY = self.dimU
        self.dataY = tffloat([0.])
        self.dataForcingPrior = None
        self.dataKappaPrior = None
        self.dataBlPrior = None
        self.dataBrPrior = None

        self.logProbPhysResidual = self.physicsPenalty_Cheby

        self.log2pi = tf.math.log(2. * tf.constant(np.pi, dtype=floatf))
        self.n_sgd = 1
        self.miniBatch = 1

    # ------------------------------------------------------------------
    # mode selection (legacy API)
    # ------------------------------------------------------------------

    def setModePrior(self):
        self.mode = 'prior'
        self.setPriorOnlyTrainable()

    def setModePost(self):
        self.mode = 'post'
        self.setPostOnlyTrainable()

    def setModeCouple(self):
        self.mode = 'post'
        self.setPriorPostTrainable()

    def setPriorOnlyTrainable(self):
        self.trainableVariables = [self.priorModels['trainable_variables']]

    def setPostOnlyTrainable(self):
        self.trainableVariables = [self.postModels['trainable_variables']]

    def setPriorPostTrainable(self):
        self.trainableVariables = [self.priorModels['trainable_variables'],
                                   self.postModels['trainable_variables']]

    # ------------------------------------------------------------------
    # network construction (legacy API)
    # ------------------------------------------------------------------

    def initPriorNetworks(self, U_KFBlBr=None, Z_UF=None, init_seed=None):

        if U_KFBlBr is None:
            dimKFBlBr = self.dimK + self.dimF + self.dimBl + self.dimBr
            self.U_KFBlBr = NeuralN('U_of_KFBlBr',
                                    np.array([dimKFBlBr,
                                              20, 20, 20, self.dimU]),
                                    init_seed=init_seed)
        else:
            self.U_KFBlBr = U_KFBlBr

        if Z_UF is None:
            k = self.dimK
            bl = k + self.dimBl
            br = bl + self.dimBr
            self.Z_UF = NeuralN('Z_of_UF',
                                np.array([self.dimU + self.dimF,
                                          20, 20, 20,
                                          self.dimK + self.dimBl
                                          + self.dimBr]),
                                multOutputLayout={'K': tf.range(0, k),
                                                  'Bl': tf.range(k, bl),
                                                  'Br': tf.range(bl, br)},
                                init_seed=init_seed)
        else:
            self.Z_UF = Z_UF

        self.U_KFBlBr.compile()
        self.Z_UF.compile()

        self.varListPriorNets = [self.U_KFBlBr.trainable_variables,
                                 self.Z_UF.trainable_variables]

        self.priorModels = {self.U_KFBlBr.NN_name: self.U_KFBlBr,
                            self.Z_UF.NN_name: self.Z_UF,
                            'trainable_variables': self.varListPriorNets}

    def initPostNetworks(self, Y_U=None, KBlBr_Y=None, init_seed=None):

        if Y_U is None:
            self.Y_U = NeuralN('Y_of_U',
                               np.array([self.dimU, 101, 101, self.dimY]),
                               transf=tfb.JointMap([tfb.Identity()]),
                               init_seed=init_seed)
        else:
            self.Y_U = Y_U

        if KBlBr_Y is None:
            k = self.dimK
            bl = k + self.dimBl
            br = bl + self.dimBr

            self.KBlBr_Y = NeuralN('KBlBr_of_Y',
                                   np.array([self.dimY,
                                             50, 50, 50, 50, 50, 50, 50, 50,
                                             self.dimK + self.dimBl
                                             + self.dimBr]),
                                   transf=tfb.JointMap([tfb.Identity()]),
                                   multOutputLayout={
                                       'K': tf.range(0, k),
                                       'Bl': tf.range(k, bl),
                                       'Br': tf.range(bl, br)},
                                   init_seed=init_seed)
        else:
            self.KBlBr_Y = KBlBr_Y

        self.Y_U.compile()
        self.KBlBr_Y.compile()
        self.varListPostNets = [
            self.Y_U.trainable_variables,
            self.KBlBr_Y.trainable_variables
        ]

        self.postModels = {
            self.Y_U.NN_name: self.Y_U,
            self.KBlBr_Y.NN_name: self.KBlBr_Y,
            'trainable_variables': self.varListPostNets}

    # ------------------------------------------------------------------
    # distribution helpers (legacy API)
    # ------------------------------------------------------------------

    @staticmethod
    def reparameterize(mean, logvar):
        return _reparameterize(mean, logvar)

    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
        return diag_gaussian_log_prob(sample, mean, logvar, axis=raxis)

    def physicsPenalty(self, u, epsilon):
        A, f = self.solver.assemble_tf(tf.reshape(u, [-1]))
        return physics_gaussian_log_likelihood(
            A, f, u[0, :], epsilon)

    def physicsPenalty_Cheby(self, uc, epsilon):
        u = self.pdeState.u_funXFE(tf.reshape(uc, [-1, 1]))
        A, f = self.solver.assemble_tf(u)
        return physics_gaussian_log_likelihood(
            A, f, tf.reshape(u, [-1]), epsilon)

    # ------------------------------------------------------------------
    # configuration helpers (new, non-breaking)
    # ------------------------------------------------------------------

    def setPriors(self, kappa=None, forcing=None, bl=None, br=None):
        """Assign the latent prior distributions (prior mode)."""
        if kappa is not None:
            self.LatentKappaPrior = kappa
        if forcing is not None:
            self.LatentForcingPrior = forcing
        if bl is not None:
            self.LatentBlPrior = bl
        if br is not None:
            self.LatentBrPrior = br

    def _require_prior_dists(self):
        for attr in _PRIOR_DIST_ATTRS:
            if getattr(self, attr) is None:
                raise ValueError(
                    f"VE.{attr} is None: assign the latent prior "
                    f"distributions (or call VE.setPriors) before running "
                    f"the prior-mode ELBO")

    def _require_data(self):
        if self.dataY.shape.rank != 2:
            raise ValueError(
                "VE.dataY must be a rank-2 tensor [N, dimU]; the legacy "
                "placeholder tffloat([0.]) is not usable. Assign real "
                "observations before running the posterior-mode ELBO.")
        for attr in ('dataForcingPrior', 'dataKappaPrior',
                     'dataBlPrior', 'dataBrPrior'):
            if getattr(self, attr) is None:
                raise ValueError(
                    f"VE.{attr} is None: assign the observation priors "
                    f"before running the posterior-mode ELBO")

    # ------------------------------------------------------------------
    # ELBO objectives
    # ------------------------------------------------------------------

    def elbo_terms_prior(self, iteration=None):
        """Prior-mode ELBO with per-component accounting -> ElboTerms."""
        self._require_prior_dists()
        log_lik, log_pri, log_q = [], [], []

        for _ in range(self.n_sgd):
            f_i = self.LatentForcingPrior.sample(sample_shape=(1))
            k_i = self.LatentKappaPrior.sample(sample_shape=(1))
            bdLeft_i = self.LatentBlPrior.sample(sample_shape=(1))
            bdRight_i = self.LatentBrPrior.sample(sample_shape=(1))

            # ensure rank-2 [1, dim] batches (Map requires rank-2 inputs)
            f_i = tf.reshape(f_i, [1, -1])
            k_i = tf.reshape(k_i, [1, -1])
            bdLeft_i = tf.reshape(bdLeft_i, [1, -1])
            bdRight_i = tf.reshape(bdRight_i, [1, -1])

            self.pdeState.forcing = tf.reshape(f_i, [-1])
            self.pdeState.kappa = tf.reshape(k_i, [-1])
            self.pdeState.bdLeft = tf.reshape(bdLeft_i, [])
            self.pdeState.bdRight = tf.reshape(bdRight_i, [])

            mu_alpha, logvar_alpha = self.U_KFBlBr.Map(
                [k_i, f_i, bdLeft_i, bdRight_i])

            us = self.reparameterize(mu_alpha, logvar_alpha)

            sampleZ = tf.concat([k_i, bdLeft_i, bdRight_i], 1)
            mean_beta_Z, logvar_beta_Z = self.Z_UF.Map(
                [us, f_i])
            logP_beta_Z_uf = self.log_normal_pdf(
                sampleZ, mean_beta_Z, logvar_beta_Z)

            logP_alpha_u_Z = self.log_normal_pdf(us, mu_alpha, logvar_alpha)

            logPr_Z = self.logProbPhysResidual(us, self.epsilon_r)

            log_lik.append(logPr_Z)
            log_pri.append(logP_beta_Z_uf)
            log_q.append(logP_alpha_u_Z)

        return combine_terms(tf.stack(log_lik), tf.stack(log_pri),
                             tf.stack(log_q), reduce=tf.reduce_mean)

    def compute_nelbo_prior(self, iteration=None):
        """Legacy scalar interface: negative prior-mode ELBO."""
        return self.elbo_terms_prior(iteration).nelbo

    def elbo_terms_post(self, iteration):
        """Posterior-mode ELBO with per-component accounting -> ElboTerms."""
        self._require_data()
        i = iteration % self.dataY.shape[0]
        y_i = self.dataY[i, :][None, :]

        log_lik, log_pri, log_q = [], [], []
        for _ in range(self.n_sgd):
            mean_phi, logvar_phi = self.KBlBr_Y.Map([y_i])
            Zs = self.reparameterize(mean_phi, logvar_phi)
            logq_phi_Z_y = self.log_normal_pdf(Zs, mean_phi, logvar_phi)

            layout = self.KBlBr_Y.multOutputLayout
            k_i = tf.gather(Zs, layout['K'], axis=1)
            bdLeft_i = tf.gather(Zs, layout['Bl'], axis=1)
            bdRight_i = tf.gather(Zs, layout['Br'], axis=1)

            # forcing prior: per-row Gaussian (mean/logvar at row i)
            f_mu = self.dataForcingPrior[0][i, :]
            f_logvar = self.dataForcingPrior[1][i, :]
            f_i = tfd.Normal(
                loc=f_mu[None, :],
                scale=tf.exp(f_logvar[None, :] * 0.5)).sample()

            mean_alpha, logvar_alpha = self.U_KFBlBr.Map(
                [k_i, f_i, bdLeft_i, bdRight_i])
            us = self.reparameterize(mean_alpha, logvar_alpha)
            logqU_alpha_ZF = self.log_normal_pdf(us, mean_alpha, logvar_alpha)

            self.pdeState.forcing = tf.reshape(f_i, [-1])
            self.pdeState.kappa = tf.reshape(k_i, [-1])
            self.pdeState.bdLeft = tf.squeeze(bdLeft_i)
            self.pdeState.bdRight = tf.squeeze(bdRight_i)

            uFE = tf.reshape(
                self.pdeState.u_funXFE(tf.reshape(us, [-1, 1])), [1, -1])
            logPr_y_u = self.log_normal_pdf(
                y_i, uFE, 2 * tf.math.log(self.epsilon_y))

            mean_theta, logvar_theta = self.Y_U.Map([us])

            logP_theta_y_u = self.log_normal_pdf(
                y_i, mean_theta, logvar_theta)

            logPKappa = self.log_normal_pdf(
                k_i, self.dataKappaPrior[0][i, :],
                self.dataKappaPrior[1][i, :])
            logPBl = self.log_normal_pdf(
                bdLeft_i, self.dataBlPrior[0][i, :],
                self.dataBlPrior[1][i, :])
            logPBr = self.log_normal_pdf(
                bdRight_i, self.dataBrPrior[0][i, :],
                self.dataBrPrior[1][i, :])
            logPZ = tf.reduce_sum(logPKappa + logPBl + logPBr)

            log_lik.append(logPr_y_u)
            log_pri.append(logP_theta_y_u + logPZ)
            log_q.append(logq_phi_Z_y + logqU_alpha_ZF)

        return combine_terms(tf.stack(log_lik), tf.stack(log_pri),
                             tf.stack(log_q), reduce=tf.reduce_mean)

    def compute_nelbo_post(self, iteration):
        """Legacy scalar interface: negative posterior-mode ELBO."""
        return self.elbo_terms_post(iteration).nelbo


if __name__ == '__main__':

    from poisson1DPDEState import PDE_State
    from poisson1DSolver import Solver

    pdeState = PDE_State(dimK=1, dimU=3, dimF=1)

    omegaLength = 2.0
    numFElems = 20
    xFE = tf.constant(np.linspace(0., omegaLength, numFElems + 1)) - 1.
    solver = Solver(pdeState, xFE, isNonLinear=False)

    model = VE(pdeState, solver)
    model.initPriorNetworks()
    model.initPostNetworks()
    model.setModePrior()
