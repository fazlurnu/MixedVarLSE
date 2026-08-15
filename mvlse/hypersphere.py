"""The hypersphere decomposition — paper Eqs. (20)-(22).

The problem it solves: the correlation between two *levels* of a discrete variable is a
free parameter, but the matrix of all such correlations cannot be free — it has to come out
**PDUDE**, Positive Definite with Unit Diagonal Elements, or the Kriging system in
:mod:`mvlse.gp` has no Cholesky factor and Eqs. (10), (12) and (14) have nothing to invert.

Rather than fit ``T`` and check it, the decomposition *generates* only legal matrices::

    T = L L^T                                                          Eq. (20)

    l_{1,1} = 1
    l_{k,1} = cos(a_{k,1})                                             Eq. (21)
    l_{k,s} = sin(a_{k,1}) ... sin(a_{k,s-1}) cos(a_{k,s})
    l_{k,k} = sin(a_{k,1}) ... sin(a_{k,k-1})

Row ``k`` of ``L`` is a point on the unit sphere, written in spherical coordinates. So:

* ``T = L L^T`` is positive semi-definite for free — ``a' T a = ||L^T a||^2 >= 0``;
* every row has length 1, so ``T_kk = 1`` — the unit diagonal, also for free;
* ``T_jk = l_j . l_k = cos(angle between them)``, so it can be **negative**. This is the
  whole reason the hypersphere kernels can express two levels that move *against* each
  other, and the coding and Gower kernels cannot: those compute ``exp(-theta d) > 0``.

Keeping the angles strictly inside ``(0, pi)`` keeps ``l_kk > 0``, so ``L`` is non-singular
and ``T`` is positive *definite* rather than merely semi-definite — which PDUDE requires.
"""
from __future__ import annotations

import numpy as np

__all__ = ["ANGLE_BOUNDS", "cholesky_factor", "n_angles", "pdude"]

ANGLE_BOUNDS = (0.05, np.pi - 0.05)
"""Angle bounds. Held off ``0`` and ``pi`` so ``L`` stays comfortably non-singular."""


def n_angles(b: int) -> int:
    """Angles needed for a ``b x b`` PDUDE matrix: ``b(b-1)/2``."""
    if b < 1:
        raise ValueError(f"need at least one level, got {b}")
    return b * (b - 1) // 2


def cholesky_factor(angles: np.ndarray, b: int) -> np.ndarray:
    """The lower-triangular ``L`` of Eq. (21), whose rows are unit vectors."""
    angles = np.asarray(angles, dtype=float)
    if angles.size != n_angles(b):
        raise ValueError(f"a {b}x{b} matrix needs {n_angles(b)} angles, got {angles.size}")
    lower = np.zeros((b, b))
    lower[0, 0] = 1.0
    taken = 0
    for row in range(1, b):
        a = angles[taken : taken + row]
        taken += row
        # cos^2 + sin^2 = 1 collapses from the inside out, so the row has length 1.
        sin_prod = np.concatenate([[1.0], np.cumprod(np.sin(a))])
        for col in range(row):
            lower[row, col] = sin_prod[col] * np.cos(a[col])
        lower[row, row] = sin_prod[row]
    return lower


def pdude(angles: np.ndarray, b: int) -> np.ndarray:
    """The level-correlation matrix ``T`` of Eq. (20) — ``b x b``, PDUDE by construction.

    Entry ``T[i, j]`` is the learned correlation between levels ``i`` and ``j``. Reading
    these off a fitted model is how you find out what the surrogate decided your discrete
    variable *means* — see :meth:`mvlse.models.MixedGP.level_correlations`.
    """
    lower = cholesky_factor(angles, b)
    return lower @ lower.T
