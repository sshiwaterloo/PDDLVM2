"""Training loop for the variational objectives.

Design contract
---------------
``Trainer`` is objective-agnostic: the caller supplies a ``forward(batch)``
callable that returns an ``ElboTerms`` record, plus the list of trainable
variables.  The trainer owns the mechanics only:

- ``tf.GradientTape`` + ``optimizer.apply_gradients`` on the NELBO,
- per-component metric recording (log_likelihood / log_prior / log_q /
  elbo / nelbo are logged separately),
- an optional ``tf.data`` pipeline (batch / shuffle / prefetch),
- optional learning-rate schedule,
- optional early stopping on a monitored validation metric,
- checkpoint save/restore through ``pddlvm.probabilistic.checkpointing``.
"""

import logging

import tensorflow as tf

from pddlvm.probabilistic.elbo import ElboTerms

_logger = logging.getLogger(__name__)


def build_dataset(data, batch_size, shuffle=True, seed=None,
                  prefetch=tf.data.AUTOTUNE, repeat=None):
    """tf.data pipeline over a batch-major tensor ``data`` of shape [N, ...]."""
    ds = tf.data.Dataset.from_tensor_slices(data)
    if shuffle:
        ds = ds.shuffle(buffer_size=max(1, min(int(data.shape[0]), 1024)),
                        seed=seed)
    ds = ds.batch(batch_size)
    if repeat is not None:
        ds = ds.repeat(repeat)
    ds = ds.prefetch(prefetch)
    return ds


class Trainer:
    """Minimal, explicit training loop with metrics and checkpoints.

    Parameters
    ----------
    forward : callable(batch) -> ElboTerms
        Builds the objective for one batch (no gradient logic inside).
    variables : list of tf.Variable
        Weights optimized by the NELBO.
    optimizer : tf.keras.optimizers.Optimizer
    lr_schedule : callable(step:int) -> float, optional
        When given, the optimizer learning rate is updated before each step.
    checkpointer : Checkpointer, optional
        When given, the trainer saves after every ``checkpoint_every`` steps
        and restores from ``resume_prefix`` before training.
    jit : bool
        Wrap the training step in tf.function.  Only safe when ``forward``
        is side-effect free (pure tensor ops).  The legacy PDDLVM prior-mode
        forward assigns sampled latents onto ``pdeState`` attributes, which
        only executes at trace time under tf.function, so those forwards
        must run eagerly (default).
    """

    def __init__(self, forward, variables, optimizer,
                 lr_schedule=None, checkpointer=None,
                 checkpoint_every=100, jit=False):
        self.forward = forward
        self.variables = list(variables)
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.checkpointer = checkpointer
        self.checkpoint_every = checkpoint_every
        self.jit = jit
        self._step_fn = tf.function(self._train_step) if jit \
            else self._train_step
        self.history = {k: [] for k in
                        ("step", "log_likelihood", "log_prior", "log_q",
                         "elbo", "nelbo", "lr")}

    def _apply_lr(self, step):
        if self.lr_schedule is not None:
            lr = float(self.lr_schedule(step))
            self.optimizer.learning_rate.assign(lr)
            return lr
        return float(self.optimizer.learning_rate)

    def _train_step(self, batch):
        with tf.GradientTape() as tape:
            terms = self.forward(batch)
            loss = terms.nelbo
        grads = tape.gradient(loss, self.variables)
        self.optimizer.apply_gradients(zip(grads, self.variables))
        # return plain tensors: tf.function cannot return ElboTerms
        # (a dataclass), so the jit path wraps this method
        return (terms.log_likelihood, terms.log_prior, terms.log_q,
                terms.elbo, terms.nelbo)

    def train_step(self, batch):
        """One optimization step; records per-component metrics."""
        self._apply_lr(int(self.optimizer.iterations))
        vals = self._step_fn(batch)
        terms = ElboTerms(log_likelihood=vals[0], log_prior=vals[1],
                          log_q=vals[2], elbo=vals[3], nelbo=vals[4])
        step = int(self.optimizer.iterations)
        self.history["step"].append(step)
        for name in ("log_likelihood", "log_prior", "log_q", "elbo", "nelbo"):
            self.history[name].append(float(getattr(terms, name)))
        self.history["lr"].append(float(self.optimizer.learning_rate))
        return terms

    def evaluate(self, batch):
        """Forward pass without gradient updates (e.g. validation)."""
        return self.forward(batch)

    def fit(self, dataset, steps, val_batch=None, log_every=10,
            early_stop=None, patience=20):
        """Run ``steps`` optimization steps over ``dataset``.

        Parameters
        ----------
        dataset : tf.data.Dataset
        steps : int
        val_batch : optional batch on which to monitor ``early_stop``
        early_stop : {"nelbo", "elbo"} or None
            When given, training stops once the monitored validation metric
            has not improved for ``patience`` evaluations (checked every
            ``log_every`` steps).
        patience : int
            Number of tolerated non-improving validations.

        Returns
        -------
        self.history : dict of lists

        Notes
        -----
        ``dataset`` must yield at least ``steps`` batches; apply
        ``.repeat()`` (or ``repeat=None`` is NOT enough) when the number of
        steps exceeds one pass over the data.
        """
        self.history["val"] = []
        best_val = None
        stalled = 0
        for batch in dataset.take(steps):
            self.train_step(batch)
            step = self.history["step"][-1]

            if self.checkpointer is not None and step % self.checkpoint_every == 0:
                prefix = self.checkpointer.save()
                _logger.debug("saved checkpoint %s", prefix)

            if step % log_every == 0 and val_batch is not None:
                terms = self.evaluate(val_batch)
                val = float(getattr(terms, early_stop or "nelbo"))
                self.history["val"].append(val)
                improved = best_val is None or val < best_val
                if improved:
                    best_val = val
                    stalled = 0
                else:
                    stalled += 1
                if early_stop is not None and stalled >= patience:
                    _logger.info("early stop at step %d (val=%g)", step, val)
                    break
        return self.history

    def restore(self, prefix):
        """Restore weights/optimizer from a checkpoint prefix."""
        if self.checkpointer is None:
            raise ValueError("Trainer has no checkpointer")
        return self.checkpointer.ckpt.restore(prefix)
