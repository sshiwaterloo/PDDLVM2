"""Checkpointing helpers built on tf.train.Checkpoint.

A thin, tested convention so every training run can be resumed seamlessly:
one ``Checkpointer`` tracks everything that changes during training
(network weights, optimizer state, step counter) under a single object and
wraps ``tf.train.CheckpointManager`` for save/restore bookkeeping.
"""

import tensorflow as tf


class Checkpointer:
    """tf.train.Checkpoint + CheckpointManager with a minimal API."""

    def __init__(self, checkpoint_dir, max_to_keep=3, **trackables):
        self.ckpt = tf.train.Checkpoint(**trackables)
        self.manager = tf.train.CheckpointManager(
            self.ckpt, directory=checkpoint_dir, max_to_keep=max_to_keep)

    def save(self):
        """Write a checkpoint; returns the saved prefix."""
        return self.manager.save()

    def restore_latest(self):
        """Restore the latest checkpoint; returns the restore status."""
        latest = self.manager.latest_checkpoint
        if latest is None:
            raise FileNotFoundError(
                f"no checkpoint found in {self.manager.directory}")
        return self.ckpt.restore(latest)

    @property
    def latest_prefix(self):
        return self.manager.latest_checkpoint
