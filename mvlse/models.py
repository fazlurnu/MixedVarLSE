"""Surrogates: the four mixed kernels, plus category-wise Kriging, behind one interface.

Everything downstream — the acquisition, the LSE loop, the benchmark — only ever asks a
surrogate for ``predict(w) -> (mean, sd)``. That is what lets category-wise Kriging sit
beside the kernels as a peer even though it is structurally a different animal: ``m``
independent continuous GPs with nothing shared, which is the paper's Section 3.2 and the
reference every mixed method is judged against.

    from mvlse.models import fit_surrogate

    model = fit_surrogate("hs_dim", space, w, y, se, rng=rng)
    mu, sd = model.predict(grid)
    model.level_correlations()      # what it decided your discrete variables mean
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .correlation import Correlation
from .gp import FittedGP, fit
from .kernels import KERNEL_NAMES, make_kernel
from .space import DesignSpace

__all__ = ["MODEL_NAMES", "CategoryWiseGP", "MixedGP", "Surrogate", "fit_surrogate"]

MODEL_NAMES = ("category_wise", *KERNEL_NAMES)
"""Every method from the paper this library implements, in Section order."""

_MIN_PER_FIT = 2
"""Below this a category cannot be fitted at all, and falls back to the pooled prior."""


@runtime_checkable
class Surrogate(Protocol):
    """What the rest of the library asks of a fitted model."""

    name: str

    def predict(self, w_new: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Posterior mean and standard deviation of the latent response."""
        ...


@dataclass(frozen=True)
class MixedGP:
    """One Kriging model over the whole space, using a mixed-variable kernel."""

    name: str
    gp: FittedGP = field(repr=False)

    def predict(self, w_new: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.gp.predict(w_new)

    @property
    def warm_start(self) -> np.ndarray:
        """Hyperparameters to seed the next fit with."""
        return self.gp.params

    @property
    def n_hyperparameters(self) -> int:
        return len(self.gp.params)

    def level_correlations(self) -> dict[str, np.ndarray]:
        """Per discrete variable, the learned ``b_k x b_k`` level-correlation matrix."""
        return self.gp.level_correlations()

    def category_correlations(self) -> np.ndarray:
        """The implied ``m x m`` correlation between categories."""
        return self.gp.category_correlations()


@dataclass(frozen=True)
class CategoryWiseGP:
    """Paper Section 3.2: one independent continuous GP per category.

    No information crosses a category boundary — which is the point. It is the industry
    default and the floor the mixed kernels have to beat, and it fails in the two ways the
    paper predicts: it needs a lot of data per category, and it cannot say anything at all
    about a category the DoE never visited (there it returns the pooled mean and spread,
    flagged by :attr:`unfitted`).
    """

    name: str
    space: DesignSpace = field(repr=False)
    models: dict[int, FittedGP] = field(repr=False)
    fallback_mean: float
    fallback_sd: float
    unfitted: tuple[int, ...]

    def predict(self, w_new: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        w_new = np.atleast_2d(np.asarray(w_new, dtype=float))
        mu = np.full(len(w_new), self.fallback_mean)
        sd = np.full(len(w_new), self.fallback_sd)
        categories = self.space.category_of(w_new)
        for cat, gp in self.models.items():
            mask = categories == cat
            if mask.any():
                mu[mask], sd[mask] = gp.predict(w_new[mask][:, : self.space.q])
        return mu, sd

    @property
    def warm_start(self) -> dict[int, np.ndarray]:
        return {cat: gp.params for cat, gp in self.models.items()}

    @property
    def n_hyperparameters(self) -> int:
        return sum(len(gp.params) for gp in self.models.values())

    def level_correlations(self) -> dict[str, np.ndarray]:
        """Empty: this method models no relationship between levels, by construction."""
        return {}

    def category_correlations(self) -> np.ndarray:
        """The identity — every category is modelled in isolation."""
        return np.eye(self.space.m)


def fit_surrogate(
    method: str,
    space: DesignSpace,
    w: np.ndarray,
    y: np.ndarray,
    se: np.ndarray | None = None,
    *,
    correlation: str | Correlation = "squar_exp",
    rng: np.random.Generator | None = None,
    n_starts: int = 6,
    warm_start: Any | None = None,
) -> Surrogate:
    """Fit ``method`` over ``space`` and return something that can ``predict``.

    Args:
        method: one of :data:`MODEL_NAMES`.
        space: the design space the encoded ``w`` belongs to.
        w: encoded inputs, ``(n, q + r)``.
        y: responses, ``(n,)``.
        se: per-point standard errors, or ``None`` for a deterministic blackbox.
        correlation: the continuous correlation function (paper Eq. 8) every method is
            built on — see :data:`mvlse.correlation.CORRELATIONS`.
        rng: source of the random restarts.
        n_starts: random restarts per fit.
        warm_start: the previous fit's ``warm_start``, if you are in an active loop.
    """
    if method not in MODEL_NAMES:
        raise ValueError(f"unknown method {method!r}; expected one of {list(MODEL_NAMES)}")
    w = np.atleast_2d(np.asarray(w, dtype=float))
    y = np.asarray(y, dtype=float).ravel()

    if method != "category_wise":
        kernel = make_kernel(method, space, correlation)
        gp = fit(kernel, w, y, se, rng=rng, n_starts=n_starts, warm_start=warm_start)
        return MixedGP(method, gp)

    return _fit_category_wise(space, w, y, se, correlation, rng, n_starts, warm_start)


def _fit_category_wise(
    space: DesignSpace,
    w: np.ndarray,
    y: np.ndarray,
    se: np.ndarray | None,
    correlation: str | Correlation,
    rng: np.random.Generator | None,
    n_starts: int,
    warm_start: Any | None,
) -> CategoryWiseGP:
    sub_space = space.continuous_only()
    # A kernel over a space with no discrete variables is just the continuous correlation;
    # "dc" is the cheapest builder that degenerates to exactly that.
    sub_kernel = make_kernel("dc", sub_space, correlation)
    categories = space.category_of(w)
    warm = warm_start if isinstance(warm_start, dict) else {}

    models: dict[int, FittedGP] = {}
    unfitted: list[int] = []
    for cat in range(space.m):
        mask = categories == cat
        if int(mask.sum()) < _MIN_PER_FIT:
            unfitted.append(cat)
            continue
        try:
            models[cat] = fit(
                sub_kernel,
                w[mask][:, : space.q],
                y[mask],
                None if se is None else np.asarray(se, dtype=float)[mask],
                rng=rng,
                n_starts=n_starts,
                warm_start=warm.get(cat),
            )
        except ValueError:  # degenerate cell (duplicate points, zero spread) — fall back
            unfitted.append(cat)

    return CategoryWiseGP(
        name="category_wise",
        space=space,
        models=models,
        fallback_mean=float(np.mean(y)),
        fallback_sd=float(np.std(y)) if len(y) > 1 else 1.0,
        unfitted=tuple(unfitted),
    )
