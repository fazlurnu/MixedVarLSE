"""Comparing methods on a problem whose answer you already know.

This is the machinery behind ``docs/FINDINGS.md``. Two questions it answers, and they are
different questions:

:func:`compare_methods`
    Same fixed design of experiments, every method, scored against the truth. Isolates the
    **correlation function** — nothing else varies.

:func:`compare_active`
    Every method driving its own active loop under the same budget. Isolates what the
    method is worth **in use**, where a better surrogate also picks better points.

Both average over several DoE seeds, because one DoE is one sample and the spread between
seeds is often the same size as the gap between methods. A single-seed comparison of
surrogates is not evidence.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .doe import grid_candidates, random_candidates, stratified_doe
from .lse import LseConfig, run_lse
from .metrics import LevelSetScore, score_level_set
from .models import MODEL_NAMES, fit_surrogate
from .problems.catalog import Problem

__all__ = ["BenchmarkRow", "compare_active", "compare_methods", "test_grid"]

DEFAULT_SEEDS = (1, 2, 3)


@dataclass
class BenchmarkRow:
    """One method's scores on one problem, aggregated over seeds."""

    method: str
    problem: str
    n_train: int
    n_hyperparameters: int
    scores: list[LevelSetScore] = field(default_factory=list, repr=False)
    seconds: float = 0.0

    def _mean(self, field_name: str) -> float:
        return float(np.mean([getattr(s, field_name) for s in self.scores]))

    def _std(self, field_name: str) -> float:
        return float(np.std([getattr(s, field_name) for s in self.scores]))

    def as_dict(self) -> dict:
        return {
            "problem": self.problem,
            "method": self.method,
            "n_train": self.n_train,
            "n_hyper": self.n_hyperparameters,
            "f1": self._mean("f1"),
            "f1_std": self._std("f1"),
            "accuracy": self._mean("accuracy"),
            "rmse": self._mean("rmse"),
            "seconds": self.seconds,
        }


def test_grid(problem: Problem, n_per_axis: int = 21, n_per_category: int = 512) -> np.ndarray:
    """A dense encoded test set to score against — a lattice when ``q <= 2``, else a cloud."""
    if problem.space.q <= 2:
        return grid_candidates(problem.space, n_per_axis)
    return random_candidates(problem.space, n_per_category, seed=12345)


def _to_frame(rows: Sequence[BenchmarkRow]):
    import pandas as pd

    return pd.DataFrame([r.as_dict() for r in rows])


def compare_methods(
    problem: Problem,
    methods: Sequence[str] = MODEL_NAMES,
    *,
    n_per_category: int = 2,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    correlation: str = "squar_exp",
    grid: np.ndarray | None = None,
    n_starts: int = 4,
    verbose: bool = True,
):
    """Fit every method to the **same** stratified DoEs and score the level sets.

    Returns a pandas DataFrame, one row per method, with F1 averaged over ``seeds`` and its
    standard deviation alongside — read them together.
    """
    import time

    grid = test_grid(problem) if grid is None else grid
    truth = problem.truth(grid)
    rows: list[BenchmarkRow] = []

    for method in methods:
        started = time.perf_counter()
        row = BenchmarkRow(method, problem.name, 0, 0)
        for seed in seeds:
            w = stratified_doe(problem.space, n_per_category, seed=seed)
            y = problem(problem.space.decode(w))
            model = fit_surrogate(
                method, problem.space, w, y,
                correlation=correlation, rng=np.random.default_rng(seed), n_starts=n_starts,
            )
            mu, _ = model.predict(grid)
            row.scores.append(
                score_level_set(mu, truth, problem.threshold,
                                greater_is_inside=problem.greater_is_inside)
            )
            row.n_train = len(y)
            row.n_hyperparameters = getattr(model, "n_hyperparameters", 0)
        row.seconds = time.perf_counter() - started
        rows.append(row)
        if verbose:
            print(
                f"{method:14s} F1={row._mean('f1'):.3f} +- {row._std('f1'):.3f}  "
                f"acc={row._mean('accuracy'):.1%}  n_hyper={row.n_hyperparameters:<4d} "
                f"[{row.seconds:.1f}s]",
                flush=True,
            )
    return _to_frame(rows)


def compare_active(
    problem: Problem,
    methods: Sequence[str] = MODEL_NAMES,
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    config: LseConfig | None = None,
    grid: np.ndarray | None = None,
    verbose: bool = True,
    **overrides,
):
    """Run the full active loop once per method per seed, then score the final models.

    ``overrides`` are :class:`~mvlse.lse.LseConfig` fields — ``max_evaluations`` above all,
    since a comparison at unequal budget compares nothing.
    """
    import time

    grid = test_grid(problem) if grid is None else grid
    truth = problem.truth(grid)
    rows: list[BenchmarkRow] = []

    for method in methods:
        started = time.perf_counter()
        row = BenchmarkRow(method, problem.name, 0, 0)
        for seed in seeds:
            settings = dict(
                threshold=problem.threshold,
                greater_is_inside=problem.greater_is_inside,
                method=method,
                seed=seed,
                verbose=False,
            )
            settings.update(overrides)
            run_config = config if config is not None else LseConfig(**settings)
            result = run_lse(problem.space, problem, config=run_config)
            mu, _ = result.predict(grid)
            row.scores.append(
                score_level_set(mu, truth, problem.threshold,
                                greater_is_inside=problem.greater_is_inside)
            )
            row.n_train = result.n_evaluations
            row.n_hyperparameters = getattr(result.model, "n_hyperparameters", 0)
        row.seconds = time.perf_counter() - started
        rows.append(row)
        if verbose:
            print(
                f"{method:14s} F1={row._mean('f1'):.3f} +- {row._std('f1'):.3f}  "
                f"acc={row._mean('accuracy'):.1%}  n={row.n_train:<4d} [{row.seconds:.1f}s]",
                flush=True,
            )
    return _to_frame(rows)
