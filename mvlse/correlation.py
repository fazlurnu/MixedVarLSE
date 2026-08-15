"""Correlation functions over the **continuous** inputs — paper Eq. (8).

Every mixed-variable kernel in :mod:`mvlse.kernels` is built on one of these, so switching
from a squared exponential to a Matern changes all five methods at once. That is the point:
the paper's comparison is about how *discrete* variables enter, and holding the continuous
part fixed is what makes the comparison mean anything.

    p-exponential   R(x_i, x_j) = exp( -sum_k theta_k |x_k^i - x_k^j|^{p_k} )
    Matern          R(x_i, x_j) = prod_k  m_nu( theta_k |x_k^i - x_k^j| )

Both are **separable**: a product (or a sum inside one exponential) of one term per
dimension. That is what lets the coding kernels reuse them — direct conversion (Eq. 15) and
Gower (Eq. 18) are the *same* correlation function applied to a wider stack of
per-dimension distances, so :meth:`Correlation.from_distances` is the entry point they call
and :meth:`Correlation.__call__` is the convenience wrapper for plain continuous inputs.

Two notes on faithfulness to Eq. (8):

* the p-exponential is positive semi-definite only for ``0 < p <= 2``, which is why
  :data:`P_BOUNDS` caps ``p`` there. ``p = 2`` (the squared exponential) is the default and
  is what the paper uses for the whole of its Section 4;
* the paper writes the Matern with an explicit ``2 sqrt(nu)`` scale and a modified Bessel
  function. For ``nu = 3/2`` and ``5/2`` that collapses to the closed forms used below; the
  leading constant is a rescaling of ``theta`` and is absorbed into it.

Hyperparameters are carried as ``log10(theta)`` so the optimiser works on a scale where a
length scale of 0.03 and one of 300 are equally reachable. Bounds assume inputs on the unit
cube — with ``DesignSpace(normalize=False)`` you should widen them.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

__all__ = [
    "CORRELATIONS",
    "LOG_THETA_BOUNDS",
    "P_BOUNDS",
    "Correlation",
    "get_correlation",
]

LOG_THETA_BOUNDS = (-1.5, 2.5)
"""``log10(theta)`` bounds, calibrated for inputs on ``[0, 1]``."""

P_BOUNDS = (0.5, 2.0)
"""``p`` bounds for the p-exponential. Above 2 the function stops being PSD."""

_FromDistances = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Correlation:
    """A named continuous correlation function and its hyperparameter layout.

    ``n_params_per_dim`` is 1 for a fixed-``p`` function and 2 when ``p`` is fitted, which
    is where the paper's ``2q`` hyperparameter counts in Table 3 come from.
    """

    name: str
    n_params_per_dim: int
    _from_distances: _FromDistances
    _bounds_per_dim: tuple[tuple[float, float], ...]

    def n_params(self, d: int) -> int:
        """Number of hyperparameters when applied over ``d`` dimensions."""
        return self.n_params_per_dim * d

    def bounds(self, d: int) -> list[tuple[float, float]]:
        """Bounds in parameter-vector order: all ``log10(theta)`` first, then any second
        block (``p``, for the p-exponential)."""
        return [bound for bound in self._bounds_per_dim for _ in range(d)]

    def from_distances(self, params: np.ndarray, distances: np.ndarray) -> np.ndarray:
        """Correlation from a ``(n_a, n_b, d)`` stack of **non-negative per-dimension
        distances**.

        This is the general entry point. The coding kernels of Eqs. (15) and (18) build
        their own distance stack — a coded level difference, a 0/1 mismatch score — and
        call straight through to here.
        """
        distances = np.asarray(distances, dtype=float)
        return self._from_distances(np.asarray(params, dtype=float), distances)

    def __call__(self, params: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Correlation between continuous blocks ``a`` ``(n_a, d)`` and ``b`` ``(n_b, d)``."""
        a = np.atleast_2d(np.asarray(a, dtype=float))
        b = np.atleast_2d(np.asarray(b, dtype=float))
        return self.from_distances(params, np.abs(a[:, None, :] - b[None, :, :]))


def _fixed_p_exponential(p: float) -> _FromDistances:
    def call(params: np.ndarray, d: np.ndarray) -> np.ndarray:
        theta = 10.0 ** params
        return np.exp(-((d**p) * theta).sum(axis=2))

    return call


def _free_p_exponential(params: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Eq. (8) with ``p`` fitted per dimension."""
    n = d.shape[2]
    theta, p = 10.0 ** params[:n], params[n : 2 * n]
    return np.exp(-((d**p) * theta).sum(axis=2))


def _matern_3_2(h: np.ndarray) -> np.ndarray:
    return 1.0 + h


def _matern_5_2(h: np.ndarray) -> np.ndarray:
    return 1.0 + h + h**2 / 3.0


def _matern(nu: float) -> _FromDistances:
    """The half-integer Matern closed forms: ``polynomial(h) * exp(-h)``, per dimension."""
    polynomial = {1.5: _matern_3_2, 2.5: _matern_5_2}.get(nu)
    if polynomial is None:
        raise ValueError(f"only nu = 3/2 and 5/2 have closed forms here, got {nu}")
    root = np.sqrt(2.0 * nu)

    def call(params: np.ndarray, d: np.ndarray) -> np.ndarray:
        h = root * d * (10.0**params)
        return (polynomial(h) * np.exp(-h)).prod(axis=2)

    return call


CORRELATIONS: dict[str, Correlation] = {
    "squar_exp": Correlation("squar_exp", 1, _fixed_p_exponential(2.0), (LOG_THETA_BOUNDS,)),
    "abs_exp": Correlation("abs_exp", 1, _fixed_p_exponential(1.0), (LOG_THETA_BOUNDS,)),
    "pow_exp": Correlation("pow_exp", 2, _free_p_exponential, (LOG_THETA_BOUNDS, P_BOUNDS)),
    "matern32": Correlation("matern32", 1, _matern(1.5), (LOG_THETA_BOUNDS,)),
    "matern52": Correlation("matern52", 1, _matern(2.5), (LOG_THETA_BOUNDS,)),
}
"""The continuous correlation functions of Eq. (8).

=============  ==================================================================
``squar_exp``  p-exponential with ``p = 2``. The default, and the paper's choice.
``abs_exp``    p-exponential with ``p = 1`` (Ornstein-Uhlenbeck).
``pow_exp``    p-exponential with ``p`` fitted per dimension — ``2d`` parameters,
               the count the paper's Table 3 quotes.
``matern32``   Matern, ``nu = 3/2``.
``matern52``   Matern, ``nu = 5/2``.
=============  ==================================================================
"""


def get_correlation(name: str | Correlation) -> Correlation:
    """Look a correlation function up by name, or pass one through."""
    if isinstance(name, Correlation):
        return name
    try:
        return CORRELATIONS[name]
    except KeyError:
        raise ValueError(
            f"unknown correlation {name!r}; expected one of {sorted(CORRELATIONS)}"
        ) from None
