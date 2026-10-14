"""Opt-in plotting style helpers.

Importing this module does not mutate any matplotlib state; call
``setup_paper_style()`` explicitly when the legacy paper figure style
(LaTeX serif fonts) is wanted.  Note that ``text.usetex=True`` requires a
working LaTeX installation.
"""

from matplotlib import pyplot as plt
from matplotlib.transforms import Bbox


def setup_paper_style():
    """Apply the legacy paper figure style (LaTeX Computer Modern, serif)."""
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"],
        'axes.labelsize': 20,
        'axes.titlesize': 20,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12
    })
    # latex font definition
    plt.rc('text', usetex=True)
    plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})


def full_extent(ax, pad=0.0):
    """Get the full extent of an axes, including axes labels, tick labels,
    and titles."""
    # For text objects, we need to draw the figure first, otherwise the extents
    # are undefined.
    ax.figure.canvas.draw()
    items = ax.get_xticklabels() + ax.get_yticklabels()
    items += [ax, ax.title]
    bbox = Bbox.union([item.get_window_extent() for item in items])

    return bbox.expanded(1.0 + pad, 1.0 + pad)


def save_or_show(fig=None, path="artifacts/figure.png", show=False):
    """Save a figure to ``path`` (creating parent dirs); only pop a window
    when ``show`` is explicitly requested.

    Training/evaluation pipelines must call this with ``show=False`` (the
    default) so they never block on an interactive backend.
    """
    import os

    if fig is None:
        fig = plt.gcf()
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def show_requested():
    """True when the caller passed ``--show`` (interactive mode)."""
    import sys

    return "--show" in sys.argv
