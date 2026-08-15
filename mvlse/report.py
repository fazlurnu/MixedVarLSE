"""Plots: the boundary per category, the convergence trace, the learned level correlations.

Needs matplotlib, which is an optional extra (``pip install mvlse[plots]``). Import this
module only when you want pictures — nothing in the estimation path depends on it.

The boundary panels assume ``q == 2`` continuous variables, because that is what can be
drawn as a contour. Everything else here works at any dimension.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .doe import grid_candidates
from .lse import LseResult
from .problems.catalog import Problem
from .space import DesignSpace

__all__ = [
    "animate_boundary",
    "boundary_panels",
    "convergence",
    "level_correlation_heatmap",
]

_INSIDE = "#4c72b0"
_ESTIMATE = "#c44e52"


def _require_2d(space: DesignSpace, what: str) -> None:
    if space.q != 2:
        raise ValueError(
            f"{what} draws a contour over two continuous variables, but this space has "
            f"q={space.q}. Slice the space, or use convergence() / "
            "level_correlation_heatmap(), which work at any dimension."
        )


def _panel_layout(space: DesignSpace) -> tuple[int, int]:
    """Rows and columns for one panel per category, most-significant level down the rows."""
    if space.r == 0:
        return 1, 1
    rows = space.n_levels[0]
    return rows, space.m // rows


def _category_titles(space: DesignSpace) -> list[str]:
    titles = []
    for category in space.categories:
        parts = [
            f"{name}={space.variable(name).levels[level]}"
            for name, level in zip(space.discrete_names, category, strict=True)
        ]
        titles.append(", ".join(parts))
    return titles


def _mesh(space: DesignSpace, n: int):
    bounds = space.unit_bounds()
    axis0 = np.linspace(bounds[0, 0], bounds[0, 1], n)
    axis1 = np.linspace(bounds[1, 0], bounds[1, 1], n)
    return np.meshgrid(axis0, axis1, indexing="ij")


def boundary_panels(
    result: LseResult,
    problem: Problem | None = None,
    *,
    n_per_axis: int = 41,
    path: str | Path | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
):
    """One panel per category: the estimated boundary, the truth if you have it, the samples.

    Args:
        result: a finished run.
        problem: supply it and the true boundary is drawn as a dashed line and the true set
            shaded — for a benchmark. Omit on a real problem, where you do not have it.
        n_per_axis: resolution of the contour lattice.
        path: save here instead of returning an unsaved figure.
        title: overrides the default.
        figsize: overrides the default, which scales with the panel count.

    Returns:
        The matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    space = result.space
    _require_2d(space, "boundary_panels()")

    grid = grid_candidates(space, n_per_axis)
    mesh0, mesh1 = _mesh(space, n_per_axis)
    per_panel = n_per_axis**2
    mu, _ = result.model.predict(grid)
    truth = problem.truth(grid) if problem is not None else None
    tau = result.config.threshold

    rows, cols = _panel_layout(space)
    figsize = figsize or (2.0 * cols + 1.5, 2.0 * rows + 1.2)
    fig, axes = plt.subplots(rows, cols, figsize=figsize, sharex=True, sharey=True, squeeze=False)
    titles = _category_titles(space)
    categories = space.category_of(result.w)

    for index in range(space.m):
        ax = axes[index // cols, index % cols]
        window = slice(index * per_panel, (index + 1) * per_panel)

        if truth is not None:
            true_block = truth[window].reshape(mesh0.shape)
            if true_block.min() <= tau <= true_block.max():
                ax.contourf(mesh0, mesh1, true_block, levels=[true_block.min(), tau],
                            colors=[_INSIDE], alpha=0.25)
                ax.contour(mesh0, mesh1, true_block, levels=[tau], colors="k",
                           linewidths=0.9, linestyles="--")

        estimate = mu[window].reshape(mesh0.shape)
        if estimate.min() <= tau <= estimate.max():
            ax.contour(mesh0, mesh1, estimate, levels=[tau], colors=[_ESTIMATE], linewidths=1.6)

        here = categories == index
        initial = here & (np.arange(len(result.w)) < result.n_initial)
        acquired = here & (np.arange(len(result.w)) >= result.n_initial)
        ax.plot(result.w[initial, 0], result.w[initial, 1], "ks", ms=3.5)
        ax.plot(result.w[acquired, 0], result.w[acquired, 1], "o", color=_ESTIMATE, ms=3.5)

        ax.set_title(titles[index], fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])

    for index in range(space.m, rows * cols):
        axes[index // cols, index % cols].axis("off")

    default = (
        f"{result.config.method} — {result.n_evaluations} evaluations; "
        f"red: estimated boundary" + (", dashed: truth" if truth is not None else "")
    )
    fig.suptitle(title or default, fontsize=10)
    fig.supxlabel(space.continuous_names[0], fontsize=9)
    fig.supylabel(space.continuous_names[1], fontsize=9)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=140, bbox_inches="tight")
    return fig


def convergence(
    result: LseResult,
    problem: Problem | None = None,
    *,
    path: str | Path | None = None,
    figsize: tuple[float, float] = (8.0, 3.2),
):
    """The run's own progress trace: max straddle and ambiguous fraction against evaluations.

    Both are computable **without the truth**, which is the point — they are what you have
    on a real problem. Pass ``problem`` on a benchmark and the true F1 is overlaid, so you
    can see whether the internal signal was telling you the truth.
    """
    import matplotlib.pyplot as plt

    evaluations = [r.n_evaluations for r in result.rounds]
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(evaluations, [r.ambiguous_fraction for r in result.rounds],
            "o-", ms=3, color=_INSIDE, label="ambiguous fraction")
    ax.set_xlabel("evaluations")
    ax.set_ylabel("fraction of monitor set still unclassified")
    ax.grid(alpha=0.25)

    twin = ax.twinx()
    twin.plot(evaluations, [r.max_straddle for r in result.rounds],
              "^-", ms=3, color="C2", alpha=0.8, label="max straddle")
    twin.axhline(0, color="C2", lw=0.8, ls=":")
    twin.set_ylabel("max straddle", color="C2")
    twin.grid(False)

    if problem is not None and result.snapshots:
        from .metrics import score_level_set

        truth = problem.truth(result.monitor)
        f1 = [
            score_level_set(mu, truth, result.config.threshold,
                            greater_is_inside=result.config.greater_is_inside).f1
            for mu, _ in result.snapshots
        ]
        ax.plot(evaluations[: len(f1)], f1, "s-", ms=3, color=_ESTIMATE, label="true F1")

    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = twin.get_legend_handles_labels()
    ax.legend(handles + handles2, labels + labels2, fontsize=8, loc="center right")
    ax.set_title(f"{result.config.method} — stopped by {result.stop_reason}", fontsize=9)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=140, bbox_inches="tight")
    return fig


def level_correlation_heatmap(
    result: LseResult,
    *,
    path: str | Path | None = None,
    figsize: tuple[float, float] | None = None,
):
    """What the fit decided each discrete variable's levels mean to each other.

    Worth looking at before you trust anything else. For an **ordinal** variable a good fit
    comes out monotone — neighbouring levels more correlated than distant ones — *without
    having been told there is an order*. For a **nominal** one, any pattern at all is
    admissible, including negative entries, and a negative entry is information no
    distance-based kernel could have produced.
    """
    import matplotlib.pyplot as plt

    matrices = result.level_correlations()
    if not matrices:
        raise ValueError(
            f"method {result.config.method!r} models no relationship between levels "
            "(category-wise Kriging fits every category in isolation), so there is "
            "nothing to plot."
        )

    figsize = figsize or (3.1 * len(matrices) + 0.8, 3.0)
    fig, axes = plt.subplots(1, len(matrices), figsize=figsize, squeeze=False)
    image = None
    for ax, (name, matrix) in zip(axes[0], matrices.items(), strict=True):
        image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r")
        labels = [str(level) for level in result.space.variable(name).levels]
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                ax.text(j, i, f"{matrix[i, j]:+.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(matrix[i, j]) > 0.6 else "k")
        ax.set_xticks(range(len(labels)), labels, fontsize=8, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)), labels, fontsize=8)
        ax.set_title(name, fontsize=9)
        ax.grid(False)
    fig.colorbar(image, ax=axes[0], fraction=0.035, pad=0.03, label="correlation")
    fig.suptitle(f"Learned level correlations — {result.config.method}", fontsize=10)
    if path is not None:
        fig.savefig(path, dpi=140, bbox_inches="tight")
    return fig


def animate_boundary(
    result: LseResult,
    problem: Problem | None = None,
    *,
    path: str | Path = "lse_evolution.gif",
    max_frames: int = 30,
    n_per_axis: int = 31,
    duration_ms: int = 450,
):
    """A GIF of the estimated boundary tightening, one frame per recorded round.

    Requires ``LseConfig(keep_snapshots=True)`` — the frames are drawn from the monitor-set
    snapshots taken during the run, so they are the model as it actually was, not a refit.
    Needs Pillow.
    """
    import matplotlib.pyplot as plt
    from PIL import Image

    space = result.space
    _require_2d(space, "animate_boundary()")
    if not result.snapshots:
        raise ValueError(
            "no snapshots recorded — rerun with LseConfig(keep_snapshots=True) "
            "(or run_lse(..., keep_snapshots=True))."
        )
    if space.q != 2 or result.monitor.size == 0:
        raise ValueError("animate_boundary() needs the default 2-D monitor lattice")

    side = round((len(result.monitor) / space.m) ** 0.5)
    per_panel = side * side
    mesh0, mesh1 = _mesh(space, side)
    truth = problem.truth(result.monitor) if problem is not None else None
    tau = result.config.threshold
    rows, cols = _panel_layout(space)
    titles = _category_titles(space)
    categories = space.category_of(result.w)

    indices = np.unique(
        np.round(np.linspace(0, len(result.snapshots) - 1, min(len(result.snapshots), max_frames)))
    ).astype(int)

    frames = []
    for frame_no in indices:
        mu, _ = result.snapshots[frame_no]
        n_shown = result.rounds[frame_no].n_evaluations
        fig, axes = plt.subplots(rows, cols, figsize=(1.7 * cols + 1.0, 1.7 * rows + 1.0),
                                 sharex=True, sharey=True, squeeze=False, dpi=80)
        for index in range(space.m):
            ax = axes[index // cols, index % cols]
            window = slice(index * per_panel, (index + 1) * per_panel)
            if truth is not None:
                block = truth[window].reshape(mesh0.shape)
                if block.min() <= tau <= block.max():
                    ax.contourf(mesh0, mesh1, block, levels=[block.min(), tau],
                                colors=[_INSIDE], alpha=0.25)
                    ax.contour(mesh0, mesh1, block, levels=[tau], colors="k",
                               linewidths=0.8, linestyles="--")
            estimate = mu[window].reshape(mesh0.shape)
            if estimate.min() <= tau <= estimate.max():
                ax.contour(mesh0, mesh1, estimate, levels=[tau], colors=[_ESTIMATE], linewidths=1.4)
            shown = (categories == index) & (np.arange(len(result.w)) < n_shown)
            ax.plot(result.w[shown, 0], result.w[shown, 1], "k.", ms=3)
            ax.set_title(titles[index], fontsize=6)
            ax.set_xticks([])
            ax.set_yticks([])
        for index in range(space.m, rows * cols):
            axes[index // cols, index % cols].axis("off")
        fig.suptitle(
            f"round {frame_no}   |   {n_shown} evaluations   |   "
            f"max straddle {result.rounds[frame_no].max_straddle:+.3g}",
            fontsize=9,
        )
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(
            Image.frombuffer("RGBA", fig.canvas.get_width_height(),
                             fig.canvas.buffer_rgba()).convert("RGB")
        )
        plt.close(fig)

    frames[0].save(
        path, save_all=True, append_images=frames[1:],
        duration=[duration_ms] * (len(frames) - 1) + [2500], loop=0,
    )
    return Path(path)
