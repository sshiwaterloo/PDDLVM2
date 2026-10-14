"""Guards against import-time side effects.

Importing any module of the package must not:
- pull in matplotlib (plotting helpers are opt-in via pddlvm.plotting),
- mutate matplotlib rcParams,
- write anything to stdout.

Network construction must be silent unless ``verbose=True``, and the legacy
print-warning in the nonlinear solver must be a proper Python warning.
"""

import contextlib
import importlib
import io
import sys
import unittest
import warnings

import numpy as np
import tensorflow as tf

_PACKAGE_MODULES = (
    "pddlvm.utilities",
    "pddlvm.NeuralNet",
    "pddlvm.VEModel",
    "pddlvm.poisson1DSolver",
    "pddlvm.poisson1DPDEState",
)


class TestImportSideEffects(unittest.TestCase):
    """Import-time behaviour of the package modules."""

    def test_version_declared(self):
        import pddlvm
        self.assertRegex(pddlvm.__version__, r"^\d+\.\d+\.\d+")

    def test_a_imports_are_silent(self):
        """No module may print anything to stdout at import time."""
        for mod_name in _PACKAGE_MODULES:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                importlib.import_module(mod_name)
            self.assertEqual(buf.getvalue(), "", msg=mod_name)

    def test_a2_utilities_does_not_pull_matplotlib(self):
        # runs before any test imports matplotlib (alphabetical ordering), so
        # this observes the genuine first-import behaviour in-process
        before = set(sys.modules)
        importlib.import_module("pddlvm.utilities")
        pulled = sorted(
            m for m in sys.modules
            if m not in before and m.split(".")[0] == "matplotlib")
        self.assertEqual(pulled, [])

    def test_rcparams_not_mutated(self):
        import matplotlib
        from matplotlib import pyplot as plt
        snapshot = dict(plt.rcParams)
        importlib.import_module("pddlvm.utilities")
        self.assertEqual(dict(plt.rcParams), snapshot)
        # and the plotting module itself must be import-safe as well
        importlib.import_module("pddlvm.plotting")
        self.assertEqual(dict(plt.rcParams), snapshot)

    def test_plotting_style_is_explicit(self):
        from matplotlib import pyplot as plt
        from pddlvm.plotting import setup_paper_style
        # applying the style is an explicit, opt-in action
        setup_paper_style()
        self.assertTrue(plt.rcParams["text.usetex"])


class TestConstructionOutput(unittest.TestCase):
    """Constructing networks/models must not write to stdout."""

    def test_neuraln_silent_by_default(self):
        from pddlvm.NeuralNet import NeuralN
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            NeuralN("silent", np.array([2, 8, 2]))
        self.assertEqual(buf.getvalue(), "")

    def test_neuraln_verbose_prints_summary(self):
        from pddlvm.NeuralNet import NeuralN
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            NeuralN("loud", np.array([2, 8, 2]), verbose=True)
        out = buf.getvalue()
        self.assertIn("Layer", out)
        self.assertIn("loud", out)

    def test_solver_and_pde_state_silent(self):
        from pddlvm.poisson1DPDEState import PDE_State
        from pddlvm.poisson1DSolver import Solver
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            pde = PDE_State(xFE=tf.constant(np.linspace(-1., 1., 21),
                                            tf.float32),
                            dimK=1, dimU=3, dimF=1)
            Solver(pde, pde.xFE, isNonLinear=True)
        self.assertEqual(buf.getvalue(), "")


class TestWarnings(unittest.TestCase):
    """The legacy print-warning must be a proper Python warning."""

    def test_linear_solve_on_nonlinear_pde_warns(self):
        from pddlvm.poisson1DPDEState import PDE_State
        from pddlvm.poisson1DSolver import Solver
        pde = PDE_State(xFE=tf.constant(np.linspace(-1., 1., 21),
                                        tf.float32),
                        dimK=1, dimU=3, dimF=1)
        solver = Solver(pde, pde.xFE, isNonLinear=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            solver.solve_tf_linear()
        self.assertTrue(
            any(issubclass(w.category, UserWarning) for w in caught),
            "solve_tf_linear on a nonlinear PDE must emit a UserWarning")


if __name__ == "__main__":
    unittest.main()
