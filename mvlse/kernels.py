"""The mixed continuous/discrete correlation functions of GP2.pdf Sections 3.3-3.5.

Four kernels, four answers to one question: *a category has no distance, so what is the
correlation between two points in different categories?*

============  ========  ============================================  ==========================
name          paper     idea                                          hyperparameters
============  ========  ============================================  ==========================
``dc``        3.3 (15)  give each level a number, treat as continuous  ``c*(q + r)``
``gower``     3.4 (18)  0/1 mismatch score as the discrete distance    ``c*(q + r)``
``hs_full``   3.5 (19)  learn one ``m x m`` level-correlation matrix   ``c*q + m(m-1)/2``
``hs_dim``    3.5 (23)  learn one ``b_k x b_k`` matrix per variable    ``c*q + sum b_k(b_k-1)/2``
============  ========  ============================================  ==========================

``c`` is :attr:`~mvlse.correlation.Correlation.n_params_per_dim` — 1 with ``p`` fixed, 2
when it is fitted, which recovers the paper's ``2q`` counts in Table 3.

The fifth method, **category-wise** Kriging (Section 3.2), is not a kernel at all — it is
``m`` independent continuous GPs with nothing shared. It lives in :mod:`mvlse.models` as
:class:`~mvlse.models.CategoryWiseGP`, and it is the reference the others are judged
against.

**What separates them, before any data.** ``dc`` and ``gower`` both compute
``exp(-theta * d)``, which is strictly positive. Neither can say that two levels are
*opposed* — that where one rises, the other falls. The hypersphere kernels can, because
``T_jk`` is a cosine. On a problem with anti-correlated levels that gap is worth ~19 points
of F1 (see ``docs/FINDINGS.md``); a positive-only kernel's best move is to disown the
offending level by pushing its ``theta`` to the bound, which throws away that level's data.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .correlation import Correlation, get_correlation
from .hypersphere import ANGLE_BOUNDS, n_angles, pdude
from .space import DesignSpace

__all__ = ["KERNEL_NAMES", "MixedKernel", "make_kernel"]

KERNEL_NAMES = ("dc", "gower", "hs_full", "hs_dim")
"""Kernel-based methods. ``"category_wise"`` is a model, not a kernel — see
:func:`mvlse.models.make_model`."""

_CorrFn = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class MixedKernel:
    """A named mixed-variable correlation function bound to one design space.

    Calling it returns a **correlation** matrix with unit diagonal. The signal variance
    ``sigma^2`` is profiled out in :mod:`mvlse.gp`, exactly as in the paper's Eq. (10).
    """

    name: str
    space: DesignSpace
    correlation: Correlation
    bounds: tuple[tuple[float, float], ...]
    _corr: _CorrFn = field(repr=False)

    @property
    def n_params(self) -> int:
        return len(self.bounds)

    def __call__(self, params: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
        """``R*(w1_i, w2_j)`` — shape ``(len(w1), len(w2))``."""
        w1 = np.atleast_2d(np.asarray(w1, dtype=float))
        w2 = np.atleast_2d(np.asarray(w2, dtype=float))
        return self._corr(np.asarray(params, dtype=float), w1, w2)

    # ------------------------------------------------------- interpretability
    def category_correlations(self, params: np.ndarray) -> np.ndarray:
        """The ``m x m`` correlation between **categories** this kernel implies.

        Read off the kernel itself rather than re-derived per method: every kernel here
        factorises as ``R*(w_i, w_j) = R(x_i, x_j) * T_{c_i, c_j}``, so evaluating it at a
        single shared continuous location makes ``R(x, x) = 1`` and leaves exactly ``T``.
        """
        probe = self._category_probe()
        return self(params, probe, probe)

    def level_correlations(self, params: np.ndarray) -> dict[str, np.ndarray]:
        """Per discrete variable, the ``b_k x b_k`` correlation between its **levels**.

        This is the diagnostic worth looking at first: for an ordinal variable a
        well-fitted model should come out monotone (neighbouring levels more correlated
        than distant ones) *without having been told there is an order*.

        For ``hs_full`` the entries are a slice of the full ``m x m`` matrix, taken with
        every other discrete variable held at its level 0 — that kernel does not decompose
        per variable, which is the trade-off Eq. (23) exists to avoid.
        """
        out: dict[str, np.ndarray] = {}
        for k, name in enumerate(self.space.discrete_names):
            probe = self._level_probe(k)
            out[name] = self(params, probe, probe)
        return out

    def _category_probe(self) -> np.ndarray:
        """One encoded row per category, all sharing the same continuous coordinates."""
        cats = self.space.categories
        w = np.zeros((len(cats), self.space.q + self.space.r))
        w[:, : self.space.q] = 0.5 * self.space.unit_bounds().sum(axis=1)
        for i, cat in enumerate(cats):
            w[i, self.space.q :] = cat
        return w

    def _level_probe(self, k: int) -> np.ndarray:
        """One row per level of discrete variable ``k``; everything else held fixed."""
        b = self.space.n_levels[k]
        w = np.zeros((b, self.space.q + self.space.r))
        w[:, : self.space.q] = 0.5 * self.space.unit_bounds().sum(axis=1)
        w[:, self.space.q + k] = np.arange(b)
        return w


# --------------------------------------------------------------------------- builders
def make_kernel(
    name: str,
    space: DesignSpace,
    correlation: str | Correlation = "squar_exp",
) -> MixedKernel:
    """Build the named kernel for ``space``, over the given continuous correlation."""
    corr = get_correlation(correlation)
    builders = {
        "dc": _direct_conversion,
        "gower": _gower,
        "hs_full": _hypersphere_full,
        "hs_dim": _hypersphere_dimensionwise,
    }
    try:
        build = builders[name]
    except KeyError:
        raise ValueError(
            f"unknown kernel {name!r}; expected one of {list(KERNEL_NAMES)} "
            "(category-wise Kriging is a model, see mvlse.models.make_model)"
        ) from None
    return build(space, corr)


def _level_positions(space: DesignSpace) -> list[np.ndarray]:
    """Coded position in ``[0, 1]`` for every level of every discrete variable."""
    return [space.variable(n).positions() for n in space.discrete_names]


def _direct_conversion(space: DesignSpace, corr: Correlation) -> MixedKernel:
    """Eq. (15). Each discrete variable becomes one more continuous input.

    Exactly right for a variable that *is* a discretised continuous quantity, and a real
    bias otherwise: it forces level 0 and level 2 to be twice as far apart as 0 and 1.
    """
    q, r = space.q, space.r
    positions = _level_positions(space)

    def corr_fn(params: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
        d = np.empty((len(w1), len(w2), q + r))
        d[:, :, :q] = np.abs(w1[:, None, :q] - w2[None, :, :q])
        for k, pos in enumerate(positions):
            a = pos[w1[:, q + k].astype(int)]
            b = pos[w2[:, q + k].astype(int)]
            d[:, :, q + k] = np.abs(a[:, None] - b[None, :])
        return corr.from_distances(params, d)

    return MixedKernel("dc", space, corr, tuple(corr.bounds(q + r)), corr_fn)


def _gower(space: DesignSpace, corr: Correlation) -> MixedKernel:
    """Eqs. (16)-(18). Range-scaled distance on the continuous dimensions, a 0/1 mismatch
    score on the discrete ones, everything divided by the total dimension ``q + r``.

    Order-blind by construction: every mismatch costs the same, so it cannot say that two
    levels are *more* alike than another pair.
    """
    q, r = space.q, space.r
    total = float(q + r)
    spans = np.diff(space.unit_bounds(), axis=1).ravel()  # Delta x_k of Eq. (16)

    def corr_fn(params: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
        d = np.empty((len(w1), len(w2), q + r))
        d[:, :, :q] = np.abs(w1[:, None, :q] - w2[None, :, :q]) / spans / total
        for k in range(r):
            a = w1[:, q + k].astype(int)
            b = w2[:, q + k].astype(int)
            d[:, :, q + k] = (a[:, None] != b[None, :]) / total
        return corr.from_distances(params, d)

    return MixedKernel("gower", space, corr, tuple(corr.bounds(q + r)), corr_fn)


def _hypersphere_full(space: DesignSpace, corr: Correlation) -> MixedKernel:
    """Eqs. (19)-(22). One free correlation per **pair of categories**.

    The most expressive method here and the worst-scaling: ``m(m-1)/2`` angles, so 3
    variables of 3 levels each (``m = 27``) already costs 351. It also needs every category
    to appear in the training set, or the angles touching the missing one are unidentified.
    """
    q, m = space.q, space.m
    n_ang = n_angles(m)

    def corr_fn(params: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
        t = pdude(params[len(params) - n_ang :], m) if n_ang else np.ones((1, 1))
        c1, c2 = space.category_of(w1), space.category_of(w2)
        return corr(params[: corr.n_params(q)], w1[:, :q], w2[:, :q]) * t[c1[:, None], c2[None, :]]

    bounds = tuple(corr.bounds(q)) + (ANGLE_BOUNDS,) * n_ang
    return MixedKernel("hs_full", space, corr, bounds, corr_fn)


def _hypersphere_dimensionwise(space: DesignSpace, corr: Correlation) -> MixedKernel:
    """Eqs. (23)-(24). One small PDUDE matrix **per discrete variable**, multiplied together.

    The paper's own recommendation, and the best performer in every comparison in this
    repo. It scales with ``sum b_k(b_k-1)/2`` instead of ``m(m-1)/2`` — at ``r = 3, b = 3``
    that is 9 angles against 351 — and it does not need every category present in the DoE.
    The price is an assumption: the discrete variables do not interact.
    """
    q = space.q
    levels = space.n_levels
    n_cont = corr.n_params(q)
    offsets: list[tuple[int, int, int]] = []  # (start, n_angles, b) per discrete variable
    cursor = n_cont
    for b in levels:
        offsets.append((cursor, n_angles(b), b))
        cursor += n_angles(b)

    def corr_fn(params: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
        out = corr(params[:n_cont], w1[:, :q], w2[:, :q])
        for k, (start, n_ang, b) in enumerate(offsets):
            t = pdude(params[start : start + n_ang], b)
            a = w1[:, q + k].astype(int)
            c = w2[:, q + k].astype(int)
            out = out * t[a[:, None], c[None, :]]
        return out

    bounds = tuple(corr.bounds(q)) + (ANGLE_BOUNDS,) * sum(n_angles(b) for b in levels)
    return MixedKernel("hs_dim", space, corr, bounds, corr_fn)
