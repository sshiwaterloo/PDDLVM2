#!/usr/bin/env python
"""Line coverage for the pddlvm package (stdlib only, offline-friendly).

Runs the unittest suite under a line tracer and reports per-module line
coverage, so coverage can be measured without pytest-cov / coverage.py
(useful in the sandboxed dev environment; CI runs pytest-cov instead).

Usage (from the repo root):
    python tools/coverage.py [test_glob]

Line counting is crude by design: blank lines and comment-only lines are
excluded from the denominator; everything else counts.  Known limitation:
TensorFlow's data-pipeline background threads can clobber the process-wide
``sys.settrace`` hook mid-run, so modules exercised *after* such an abort
are undercounted (possibly to 0).  Treat these numbers as lower bounds;
the authoritative coverage figure comes from pytest-cov in CI.
"""

import os
import re
import sys
import trace
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "pddlvm")
sys.path.insert(0, ROOT)


def executable_lines(path):
    """Line numbers of non-blank, non-comment-only lines."""
    with open(path) as fh:
        lines = fh.readlines()
    out = []
    for i, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        out.append(i)
    return out


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else "*"
    # ignore stdlib/site-packages (tensorflow, numpy, ...) or the tracer
    # would slow the suite down by orders of magnitude
    tracer = trace.Trace(count=True, trace=False,
                         ignoredirs=[sys.prefix, sys.exec_prefix])
    suite = unittest.defaultTestLoader.discover(
        os.path.join(PKG, "tests"), pattern=f"test_{pattern}.py")
    runner = unittest.TextTestRunner(verbosity=0)
    result = tracer.runfunc(runner.run, suite)

    counts = tracer.results().counts
    modules = {}
    for dirpath, dirnames, filenames in os.walk(PKG):
        if os.path.basename(dirpath) == "tests":
            continue
        for fname in sorted(filenames):
            if not fname.endswith(".py"):
                continue
            path = os.path.join(dirpath, fname)
            rel = os.path.relpath(path, ROOT)
            exec_lines = executable_lines(path)
            if not exec_lines:
                continue
            hit = sum(1 for ln in exec_lines
                      if counts.get((path, ln), 0) > 0)
            modules[rel] = (hit, len(exec_lines))

    print("\n=== pddlvm line coverage (rough, stdlib tracer) ===")
    total_hit = total_all = 0
    for fname, (hit, total) in sorted(modules.items()):
        pct = 100.0 * hit / total
        total_hit += hit
        total_all += total
        print(f"  {fname:40s} {hit:4d}/{total:4d}  {pct:5.1f}%")
    print(f"  {'TOTAL':40s} {total_hit:4d}/{total_all:4d}  "
          f"{100.0 * total_hit / total_all:5.1f}%")
    print(f"\ntest run: {result.testsRun} tests, "
          f"{'OK' if result.wasSuccessful() else 'FAILED'}")


if __name__ == "__main__":
    main()
