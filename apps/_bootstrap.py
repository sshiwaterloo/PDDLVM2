"""Make the package importable when apps/*.py run as scripts.

``python apps/foo.py`` puts ``apps/`` (not the repo root) on sys.path, so
``import pddlvm`` fails without an installation.  Importing this module
first inserts the repo root.  (Prefer ``pip install -e .`` for anything
beyond quick demo runs.)
"""

import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
