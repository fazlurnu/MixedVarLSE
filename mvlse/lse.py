"""The active level-set estimation loop — declare a space, hand over a blackbox, get a set.

    space  = DesignSpace({...})
    result = run_lse(space, my_blackbox, threshold=1e-3)

Each round: fit the surrogate to everything measured so far, score every candidate with the
straddle, spend a batch on the best of them, repeat. Stopping is deliberately
**ground-truth-free**, because on a real problem the truth is the thing you are trying to
find. Three rules, whichever fires first:

``confident``   ``max straddle < 0`` — no candidate's confidence interval still contains
                ``tau``, so the model believes every classification it is making.
``plateau``     the classification on the monitor set stopped changing — under
                ``flip_tolerance`` for ``flip_window`` consecutive rounds.
``budget``      ``max_evaluations`` spent. Not convergence; say so in the write-up.

The one that fires is recorded in :attr:`LseResult.stop_reason`, and you should read it —
``budget`` next to a still-positive ``max_straddle`` means the run was cut short, not that
it finished.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .acquisition import DEFAULT_KAPPA, select_batch, straddle
from .blackbox import Blackbox, evaluate
from .doe import grid_candidates, random_candidates, stratified_doe
from .models import MODEL_NAMES, Surrogate, fit_surrogate
from .space import DesignSpace, Point
from .store import EvaluationStore

__all__ = ["LseConfig", "LseResult", "Round", "run_lse"]

_LATTICE_MAX_Q = 2
"""Above this many continuous dimensions a lattice stops being affordable and the default
candidate set switches to a quasi-random cloud."""


@dataclass(frozen=True)
class LseConfig:
    """Everything a run needs beyond the space, the blackbox and the threshold.

    A run is fully described by its config plus its seed, which is what makes it
    reproducible from the record alone.
    """

    threshold: float
    method: str = "hs_dim"
    correlation: str = "squar_exp"
    greater_is_inside: bool = False

    # budget
    n_init_per_category: int = 2
    max_evaluations: int = 200
    batch_size: int = 1

    # acquisition
    kappa: float = DEFAULT_KAPPA
    min_batch_distance: float = 0.0

    # candidate and monitor sets (None -> chosen from q)
    n_candidates_per_axis: int = 15
    n_monitor_per_axis: int = 21
    n_candidates_per_category: int = 256
    n_monitor_per_category: int = 512

    # stopping
    flip_window: int = 10
    flip_tolerance: float = 0.005

    # fitting
    seed: int = 7
    n_starts: int = 6
    n_starts_warm: int = 1

    # bookkeeping
    keep_snapshots: bool = False
    verbose: bool = True

    def __post_init__(self) -> None:
        if self.method not in MODEL_NAMES:
            raise ValueError(f"unknown method {self.method!r}; expected one of {list(MODEL_NAMES)}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.max_evaluations < 2:
            raise ValueError(f"max_evaluations must be >= 2, got {self.max_evaluations}")


@dataclass
class Round:
    """One iteration's diagnostics."""

    index: int
    n_evaluations: int
    max_straddle: float
    ambiguous_fraction: float
    flip_fraction: float
    picked: list[int] = field(default_factory=list)
    seconds: float = 0.0

    def __str__(self) -> str:
        return (
            f"round {self.index:3d}  n={self.n_evaluations:4d}  "
            f"max straddle={self.max_straddle:+.4f}  "
            f"ambiguous={self.ambiguous_fraction:.1%}  flip={self.flip_fraction:.2%}"
        )


@dataclass
class LseResult:
    """Everything a finished run produced, and the model that produced it."""

    space: DesignSpace
    config: LseConfig
    model: Surrogate
    w: np.ndarray = field(repr=False)
    y: np.ndarray = field(repr=False)
    se: np.ndarray = field(repr=False)
    n_initial: int
    rounds: list[Round] = field(default_factory=list, repr=False)
    stop_reason: str = ""
    monitor: np.ndarray = field(default_factory=lambda: np.empty((0, 0)), repr=False)
    snapshots: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list, repr=False)
    seconds: float = 0.0

    # ------------------------------------------------------------------ access
    @property
    def n_evaluations(self) -> int:
        return len(self.y)

    def points(self) -> list[Point]:
        """Every evaluated point, in physical units with your own variable names."""
        return self.space.decode(self.w)

    def records(self) -> list[dict]:
        """One dict per evaluation: the physical inputs, the response, its standard error.

        The shape you would put in a table or hand to somebody who does not have this
        library installed.
        """
        return [
            {**dict(point), "y": float(y_i), "se": float(se_i), "initial": i < self.n_initial}
            for i, (point, y_i, se_i) in enumerate(zip(self.points(), self.y, self.se, strict=True))
        ]

    def to_dataframe(self):
        """:meth:`records` as a pandas DataFrame. pandas is imported lazily."""
        import pandas as pd

        return pd.DataFrame(self.records())

    def history(self):
        """The per-round diagnostics as a pandas DataFrame."""
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "round": r.index,
                    "n_evaluations": r.n_evaluations,
                    "max_straddle": r.max_straddle,
                    "ambiguous_fraction": r.ambiguous_fraction,
                    "flip_fraction": r.flip_fraction,
                    "seconds": r.seconds,
                }
                for r in self.rounds
            ]
        )

    # ------------------------------------------------------------- predictions
    def predict(self, points) -> tuple[np.ndarray, np.ndarray]:
        """Posterior mean and sd at physical named points (or an encoded matrix)."""
        w = points if isinstance(points, np.ndarray) else self.space.encode(list(points))
        return self.model.predict(w)

    def inside(self, points) -> np.ndarray:
        """Is each point in the estimated level set?"""
        mu, _ = self.predict(points)
        tau = self.config.threshold
        return mu >= tau if self.config.greater_is_inside else mu <= tau

    def uncertain(self, points) -> np.ndarray:
        """Points the model still cannot classify — the interval straddles ``tau``."""
        mu, sd = self.predict(points)
        return straddle(mu, sd, self.config.threshold, self.config.kappa) > 0

    def level_correlations(self) -> dict[str, np.ndarray]:
        """What the fit decided each discrete variable's levels mean to each other."""
        return self.model.level_correlations()

    def summary(self) -> str:
        last = self.rounds[-1] if self.rounds else None
        lines = [
            f"level set: {'>=' if self.config.greater_is_inside else '<='} "
            f"{self.config.threshold:g}   method: {self.config.method} "
            f"({self.config.correlation})",
            f"evaluations: {self.n_evaluations} "
            f"({self.n_initial} initial + {self.n_evaluations - self.n_initial} acquired)"
            f" in {len(self.rounds)} rounds, {self.seconds:.1f}s",
            f"stopped by: {self.stop_reason}",
        ]
        if last is not None:
            lines.append(
                f"final: max straddle={last.max_straddle:+.4f}  "
                f"ambiguous={last.ambiguous_fraction:.1%}"
            )
            if last.max_straddle > 0 and self.stop_reason.startswith("budget"):
                lines.append(
                    "  note: the run hit its budget while the straddle was still "
                    "positive — this is a stopped run, not a converged one."
                )
        return "\n".join(lines)


def _default_candidates(space: DesignSpace, config: LseConfig, seed_offset: int) -> np.ndarray:
    if space.q <= _LATTICE_MAX_Q:
        return grid_candidates(space, config.n_candidates_per_axis)
    return random_candidates(space, config.n_candidates_per_category, config.seed + seed_offset)


def _default_monitor(space: DesignSpace, config: LseConfig) -> np.ndarray:
    if space.q <= _LATTICE_MAX_Q:
        return grid_candidates(space, config.n_monitor_per_axis)
    return random_candidates(space, config.n_monitor_per_category, config.seed + 991)


def run_lse(
    space: DesignSpace,
    blackbox: Blackbox,
    threshold: float | None = None,
    *,
    config: LseConfig | None = None,
    candidates: np.ndarray | None = None,
    monitor: np.ndarray | None = None,
    initial_design: np.ndarray | None = None,
    store: str | Path | EvaluationStore | None = None,
    **overrides,
) -> LseResult:
    """Estimate the level set of ``blackbox`` over ``space``.

    Args:
        space: the design space, declared by name (see :class:`~mvlse.space.DesignSpace`).
        blackbox: your simulator — takes a batch of physical points, returns one output
            each. See :mod:`mvlse.blackbox`.
        threshold: the level ``tau``. Convenience for ``config.threshold``.
        config: the full configuration. Built from ``threshold`` and ``overrides`` if
            omitted.
        candidates: encoded points the loop may sample. Defaults to a lattice for
            ``q <= 2`` and a Sobol' cloud beyond.
        monitor: encoded points the stopping rule watches. Kept separate from
            ``candidates`` so stability is not measured only where sampling happened.
        initial_design: encoded starting points. Defaults to a stratified LHS with
            ``config.n_init_per_category`` per category.
        store: path to a JSONL evaluation cache, so a rerun costs only the new points.
        **overrides: any :class:`LseConfig` field, when you have not built one yourself.

    Returns:
        An :class:`LseResult` holding the data, the fitted model and the round history.
    """
    if config is None:
        if threshold is None:
            raise TypeError("pass either threshold=... or config=LseConfig(...)")
        config = LseConfig(threshold=threshold, **overrides)
    elif overrides:
        raise TypeError(f"got both config= and overrides {sorted(overrides)}")

    started = time.perf_counter()
    rng = np.random.default_rng(config.seed)
    if candidates is None:
        candidates = _default_candidates(space, config, 0)
    else:
        candidates = np.atleast_2d(candidates)
    monitor = _default_monitor(space, config) if monitor is None else np.atleast_2d(monitor)

    if isinstance(store, EvaluationStore):
        cache = store
    elif store is not None:
        cache = EvaluationStore(store, space)
    else:
        cache = None

    def measure(w_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if cache is not None:
            return cache.evaluate(w_batch, blackbox)
        return evaluate(blackbox, space.decode(w_batch))

    # ------------------------------------------------------------ initial design
    w = (
        stratified_doe(space, config.n_init_per_category, config.seed + 4)
        if initial_design is None
        else np.atleast_2d(np.asarray(initial_design, dtype=float))
    )
    if len(w) > config.max_evaluations:
        raise ValueError(
            f"the initial design alone is {len(w)} points, past max_evaluations="
            f"{config.max_evaluations}. Lower n_init_per_category or raise the budget."
        )
    if config.verbose:
        print(f"initial design: {len(w)} points over {space.m} categories", flush=True)
    y, se = measure(w)

    result = LseResult(
        space=space, config=config, model=None, w=w, y=y, se=se,  # type: ignore[arg-type]
        n_initial=len(w), monitor=monitor,
    )

    previous_class: np.ndarray | None = None
    warm: object | None = None
    spent: set[int] = set()

    # ------------------------------------------------------------------- rounds
    for index in range(config.max_evaluations):
        round_started = time.perf_counter()
        model = fit_surrogate(
            config.method, space, result.w, result.y, result.se,
            correlation=config.correlation, rng=rng,
            n_starts=config.n_starts if warm is None else config.n_starts_warm,
            warm_start=warm,
        )
        warm = getattr(model, "warm_start", None)
        result.model = model

        mu_m, sd_m = model.predict(monitor)
        classification = (
            mu_m >= config.threshold if config.greater_is_inside else mu_m <= config.threshold
        )
        flip = 1.0 if previous_class is None else float(np.mean(classification != previous_class))
        previous_class = classification
        ambiguous = float(np.mean(straddle(mu_m, sd_m, config.threshold, config.kappa) > 0))

        mu_c, sd_c = model.predict(candidates)
        scores = straddle(mu_c, sd_c, config.threshold, config.kappa)
        if spent:
            scores[list(spent)] = -np.inf

        record = Round(
            index=index,
            n_evaluations=len(result.y),
            max_straddle=float(scores.max()),
            ambiguous_fraction=ambiguous,
            flip_fraction=flip,
        )
        if config.keep_snapshots:
            result.snapshots.append((mu_m, sd_m))

        # -------------------------------------------------------------- stopping
        recent = [r.flip_fraction for r in result.rounds[-(config.flip_window - 1):]] + [flip]
        if record.max_straddle < 0:
            result.stop_reason = "confident (max straddle < 0)"
        elif (
            index >= config.flip_window
            and len(recent) >= config.flip_window
            and max(recent) <= config.flip_tolerance
        ):
            result.stop_reason = (
                f"plateau (classification moved <= {config.flip_tolerance:.1%} "
                f"for {config.flip_window} rounds)"
            )
        elif len(result.y) >= config.max_evaluations:
            result.stop_reason = f"budget ({config.max_evaluations} evaluations)"

        if result.stop_reason:
            record.seconds = time.perf_counter() - round_started
            result.rounds.append(record)
            if config.verbose:
                print(record, flush=True)
            break

        # ------------------------------------------------------------ acquisition
        k = min(config.batch_size, config.max_evaluations - len(result.y))
        chosen = select_batch(
            scores, candidates, space.q, k,
            exclude=spent, min_distance=config.min_batch_distance,
        )
        if not chosen:
            result.stop_reason = "candidate set exhausted"
            record.seconds = time.perf_counter() - round_started
            result.rounds.append(record)
            break

        spent.update(chosen)
        record.picked = chosen
        batch = candidates[chosen]
        y_new, se_new = measure(batch)
        result.w = np.vstack([result.w, batch])
        result.y = np.append(result.y, y_new)
        result.se = np.append(result.se, se_new)

        record.seconds = time.perf_counter() - round_started
        result.rounds.append(record)
        if config.verbose:
            print(record, flush=True)
    else:
        result.stop_reason = f"budget ({config.max_evaluations} evaluations)"

    result.seconds = time.perf_counter() - started
    if config.verbose:
        print(result.summary(), flush=True)
    return result
