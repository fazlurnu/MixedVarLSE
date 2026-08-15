"""The blackbox seam: you hand in points, you get back outputs.

A blackbox is any callable that takes a **batch of** :class:`~mvlse.space.Point` (physical
units, your own names) and returns one output per point::

    def my_sim(points):
        return [run(p["range_m"], p["p_rx"], p["material"]) for p in points]

It is called in batches so an expensive simulator can parallelise internally — the same
reason ``run_experiment`` in OpenCDaRR fans conditions out itself rather than being driven
one condition at a time. Wrap a one-point-at-a-time function with :func:`pointwise`.

**Deterministic or noisy, decided by what you return.** :func:`evaluate` accepts any of

===============================  ==================================================
``[1.2, 0.7, ...]``              deterministic; the GP gets a nugget only
``[(1.2, 0.1), (0.7, 0.3)]``     ``(value, standard error)`` — heteroscedastic
``[Observation(1.2, 0.1), ...]`` the same, named
``array (n, 2)``                 columns ``[y, se]``
``[{"y": 1.2, "se": 0.1}, ...]`` mappings, ``se`` optional
===============================  ==================================================

A stochastic estimator that reports a confidence interval belongs in the second family:
pass its standard error and :mod:`mvlse.gp` puts it on the covariance diagonal as a *fixed*
per-point noise variance, so the surrogate models the latent noise-free response — which is
what the straddle in :mod:`mvlse.acquisition` needs.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .space import Point

__all__ = ["Blackbox", "Observation", "evaluate", "parallel", "pointwise"]

_SE_KEYS = ("se", "std_error", "stderr", "sigma", "std")


@dataclass(frozen=True)
class Observation:
    """One evaluated point: a value and how well it is known."""

    y: float
    se: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.y):
            raise ValueError(f"blackbox returned a non-finite value: {self.y!r}")
        if self.se < 0 or not np.isfinite(self.se):
            raise ValueError(f"standard error must be finite and >= 0, got {self.se!r}")


class Blackbox(Protocol):
    """What the library asks of your simulator."""

    def __call__(self, points: Sequence[Point]) -> Any: ...


def pointwise(fn: Callable[[Point], Any]) -> Blackbox:
    """Lift a one-point function into a batch blackbox.

        blackbox = pointwise(lambda p: p["x1"] ** 2 + p["x2"])
    """

    def batched(points: Sequence[Point]) -> list[Any]:
        return [fn(p) for p in points]

    batched.__name__ = getattr(fn, "__name__", "blackbox")
    return batched


def parallel(fn: Callable[[Point], Any], n_jobs: int = -1) -> Blackbox:
    """Lift a one-point function into a batch blackbox that fans out over joblib.

    Requires ``joblib``; ``fn`` and everything it closes over must be picklable, which
    rules out a lambda. For a simulator that already parallelises internally, pass it
    directly as a batch blackbox instead — nesting two pools does not end well.
    """

    def batched(points: Sequence[Point]) -> list[Any]:
        from joblib import Parallel, delayed

        if len(points) == 0:
            return []
        workers = n_jobs if n_jobs != 0 else 1
        return list(Parallel(n_jobs=workers)(delayed(fn)(p) for p in points))

    batched.__name__ = getattr(fn, "__name__", "blackbox")
    return batched


def _split_one(item: Any) -> tuple[float, float]:
    """One blackbox return element -> ``(y, se)``."""
    if isinstance(item, Observation):
        return item.y, item.se
    if isinstance(item, Mapping):
        if "y" not in item:
            raise TypeError(f"blackbox mapping needs a 'y' key, got keys {list(item)!r}")
        se = next((float(item[k]) for k in _SE_KEYS if k in item), 0.0)
        return float(item["y"]), se
    if isinstance(item, (str, bytes)):
        raise TypeError(f"blackbox returned a string, expected a number: {item!r}")
    if isinstance(item, Sequence) or (isinstance(item, np.ndarray) and item.ndim == 1):
        seq = list(item)
        if len(seq) == 1:
            return float(seq[0]), 0.0
        if len(seq) == 2:
            return float(seq[0]), float(seq[1])
        raise TypeError(f"expected a scalar or a (value, se) pair, got {len(seq)} items")
    return float(item), 0.0


def evaluate(blackbox: Blackbox, points: Sequence[Point]) -> tuple[np.ndarray, np.ndarray]:
    """Call ``blackbox`` on ``points`` and normalise its return into ``(y, se)``.

    Both arrays have length ``len(points)``. ``se`` is all zeros for a deterministic
    blackbox, which is what :func:`mvlse.gp.fit` reads as "nugget only".
    """
    if len(points) == 0:
        return np.empty(0), np.empty(0)

    raw = blackbox(list(points))
    if raw is None:
        raise TypeError(
            f"blackbox {getattr(blackbox, '__name__', blackbox)!r} returned None — "
            "it must return one output per point"
        )

    arr = np.asarray(raw, dtype=object) if not isinstance(raw, np.ndarray) else raw
    # The clean fast paths: a plain (n,) vector of numbers, or an (n, 2) [y, se] block.
    if isinstance(raw, np.ndarray) and raw.dtype.kind in "fiu":
        if raw.ndim == 1:
            y, se = raw.astype(float), np.zeros(len(raw))
        elif raw.ndim == 2 and raw.shape[1] == 2:
            y, se = raw[:, 0].astype(float), raw[:, 1].astype(float)
        elif raw.ndim == 2 and raw.shape[1] == 1:
            y, se = raw[:, 0].astype(float), np.zeros(len(raw))
        else:
            raise TypeError(
                f"blackbox returned an array of shape {raw.shape}; expected (n,) or (n, 2)"
            )
    else:
        pairs = [_split_one(item) for item in list(raw)]
        y = np.array([p[0] for p in pairs], dtype=float)
        se = np.array([p[1] for p in pairs], dtype=float)
        del arr

    if len(y) != len(points):
        raise ValueError(
            f"blackbox returned {len(y)} outputs for {len(points)} points — "
            "it must return exactly one output per point, in order"
        )
    if not np.all(np.isfinite(y)):
        bad = int(np.argmax(~np.isfinite(y)))
        raise ValueError(f"blackbox returned a non-finite value at point {bad}: {y[bad]!r}")
    if np.any(se < 0) or not np.all(np.isfinite(se)):
        raise ValueError("blackbox returned a negative or non-finite standard error")
    return y, se
