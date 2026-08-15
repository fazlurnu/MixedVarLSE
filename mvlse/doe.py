"""Designs of experiments and candidate sets, in encoded coordinates.

Three jobs, kept apart because they answer different questions:

* :func:`stratified_doe` — the **initial** design. Latin hypercube in the continuous
  dimensions, repeated identically in every category, so no category starts blind. This is
  what the paper's comparisons use and what the active loop starts from.
* :func:`grid_candidates` — a regular lattice per category. The natural candidate set when
  ``q = 2``, and what the boundary plots in :mod:`mvlse.report` are drawn on.
* :func:`random_candidates` — a scrambled quasi-random cloud per category. What you want
  once ``q`` is large enough that a lattice would need ``n^q`` points per category.

All of these return the encoded matrix ``W`` of :meth:`DesignSpace.encode`, so they can be
handed straight to a kernel, and decoded back to physical named points when it is time to
call the blackbox.
"""
from __future__ import annotations

import numpy as np

from .space import DesignSpace

__all__ = [
    "GRID_POINT_LIMIT",
    "grid_candidates",
    "random_candidates",
    "stratified_doe",
]

GRID_POINT_LIMIT = 2_000_000
"""Refuse to build a lattice bigger than this — ``n^q * m`` grows fast enough to exhaust
memory by accident, and a silent multi-gigabyte allocation is not a helpful failure."""


def _scale_to_space(unit: np.ndarray, space: DesignSpace) -> np.ndarray:
    """Map ``[0, 1]^q`` samples onto the encoded continuous block."""
    bounds = space.unit_bounds()
    return bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * unit


def _with_categories(continuous: np.ndarray, space: DesignSpace) -> np.ndarray:
    """Repeat a continuous block once per category, tagging each copy with its levels."""
    if space.r == 0:
        return continuous
    blocks = []
    for category in space.categories:
        tags = np.tile(np.asarray(category, dtype=float), (len(continuous), 1))
        blocks.append(np.column_stack([continuous, tags]))
    return np.vstack(blocks)


def stratified_doe(
    space: DesignSpace, n_per_category: int = 2, seed: int | None = 11
) -> np.ndarray:
    """A Latin hypercube of ``n_per_category`` points, drawn afresh for every category.

    Each continuous axis is split into ``n_per_category`` equal bins and permuted
    independently, so every category gets full marginal coverage however small the budget.
    Total size is ``n_per_category * m``.
    """
    if n_per_category < 1:
        raise ValueError(f"n_per_category must be >= 1, got {n_per_category}")
    rng = np.random.default_rng(seed)
    n = n_per_category
    blocks = []
    for category in space.categories:
        unit = np.column_stack(
            [(rng.permutation(n) + rng.random(n)) / n for _ in range(space.q)]
        )
        row = _scale_to_space(unit, space)
        if space.r:
            row = np.column_stack([row, np.tile(np.asarray(category, dtype=float), (n, 1))])
        blocks.append(row)
    return np.vstack(blocks)


def grid_candidates(space: DesignSpace, n_per_axis: int = 15) -> np.ndarray:
    """A full-factorial lattice of ``n_per_axis`` points per continuous axis, per category.

    Size is ``n_per_axis**q * m``. Fine at ``q = 2``; use :func:`random_candidates` beyond.
    """
    if n_per_axis < 2:
        raise ValueError(f"n_per_axis must be >= 2, got {n_per_axis}")
    total = n_per_axis**space.q * space.m
    if total > GRID_POINT_LIMIT:
        raise ValueError(
            f"a {n_per_axis}-point lattice over q={space.q} continuous dimensions and "
            f"m={space.m} categories is {total:,} points, past the {GRID_POINT_LIMIT:,} "
            "limit. Use random_candidates() instead, or lower n_per_axis."
        )
    bounds = space.unit_bounds()
    axes = [np.linspace(lo, hi, n_per_axis) for lo, hi in bounds]
    mesh = np.meshgrid(*axes, indexing="ij")
    continuous = np.column_stack([g.ravel() for g in mesh])
    return _with_categories(continuous, space)


def random_candidates(
    space: DesignSpace, n_per_category: int = 256, seed: int | None = 0
) -> np.ndarray:
    """A scrambled Sobol' cloud of ``n_per_category`` points per category.

    Scales to any ``q``, unlike :func:`grid_candidates`. Falls back to uniform sampling if
    SciPy's QMC module is unavailable.
    """
    if n_per_category < 1:
        raise ValueError(f"n_per_category must be >= 1, got {n_per_category}")
    try:
        import warnings

        from scipy.stats import qmc

        with warnings.catch_warnings():
            # Sobol' warns unless n is a power of two. Balance matters for integration, not
            # for a candidate pool the acquisition is about to rank, so the advice does not
            # apply here and the warning would only train users to ignore warnings.
            warnings.filterwarnings("ignore", message=".*power of 2.*")
            unit = qmc.Sobol(d=space.q, scramble=True, seed=seed).random(n_per_category)
    except ImportError:  # pragma: no cover - scipy always ships qmc these days
        unit = np.random.default_rng(seed).random((n_per_category, space.q))
    return _with_categories(_scale_to_space(unit, space), space)
