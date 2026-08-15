"""The raw mathematics of the benchmark surfaces. No design spaces, no thresholds.

Two families, kept separate because they are cited for different things:

* **Pelamatti et al. (2020),** ``references/GP2.pdf`` — the Branin of Eqs. (28)-(29), its
  10-dimensional augmentation Eqs. (33)-(35), and the Goldstein polynomial of Eq. (36) with
  the discrete levels of Table 4. These are the paper's own mixed-variable benchmarks.
* **Moss, Kochenderfer, Gariel & Dubois,** ``references/bayes-safety-val.pdf`` Section IV.A
  — Booth, Squares and Himmelblau. There they are *failure regions* of a black-box system,
  written so that ``f <= 0`` is the failure set exactly as the paper defines it.

Everything here takes and returns plain arrays. Turning them into mixed-variable level-set
problems — which variables are ordinal, where the threshold sits — is :mod:`.catalog`.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "augmented_branin_h",
    "booth_h",
    "branin_h",
    "goldstein_h",
    "himmelblau_h",
    "squares_h",
    "wiggly_h",
]


def goldstein_h(x1, x2, x3, x4):
    """Paper Eq. (36). A smooth quartic polynomial on ``x1, x2 in [0, 100]``.

    In the paper's mixed-variable version (Eq. 37, Table 4) ``x3`` and ``x4`` are not
    continuous at all — they take three discrete values each, which is where the two
    ordinal variables come from.
    """
    x1, x2, x3, x4 = (np.asarray(v, dtype=float) for v in (x1, x2, x3, x4))
    return (
        53.3108
        + 0.184901 * x1
        - 5.02914e-6 * x1**3
        + 7.72522e-8 * x1**4
        - 0.0870775 * x2
        - 0.106959 * x3
        + 7.98772e-6 * x3**3
        + 0.00242482 * x4
        + 1.32851e-6 * x4**3
        - 0.00146393 * x1 * x2
        - 0.00301588 * x1 * x3
        - 0.00272291 * x1 * x4
        + 0.0017004 * x2 * x3
        + 0.0038428 * x2 * x4
        - 0.000198969 * x3 * x4
        + 1.86025e-5 * x1 * x2 * x3
        - 1.88719e-6 * x1 * x2 * x4
        + 2.50923e-5 * x1 * x3 * x4
        - 5.62199e-5 * x2 * x3 * x4
    )


def branin_h(x1, x2):
    """Paper Eq. (28): the Branin function rescaled onto ``[0, 1]^2``, standardised."""
    x1, x2 = np.asarray(x1, dtype=float), np.asarray(x2, dtype=float)
    t = 15 * x2 - 5.1 / (4 * np.pi**2) * (15 * x1 - 5) ** 2 + 5 / np.pi * (15 * x1 - 5) - 6
    raw = t**2 + 10 * (1 - 1 / (8 * np.pi)) * np.cos(15 * x1 - 5) + 10
    return (raw - 54.8104) / 51.9496


def augmented_branin_h(x):
    """Paper Eqs. (33)-(34): ``H(x_1..x_10) = sum_{i odd} h(x_i, x_{i+1})``.

    Args:
        x: ``(n, 10)`` array of continuous inputs on ``[0, 1]``.

    The paper uses this to ask how the methods behave when the *continuous* dimension grows
    while the discrete structure stays put — 10 continuous, 2 binary discrete, 4 categories.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    if x.shape[1] % 2:
        raise ValueError(f"augmented Branin needs an even number of columns, got {x.shape[1]}")
    return sum(branin_h(x[:, i], x[:, i + 1]) for i in range(0, x.shape[1], 2))


def wiggly_h(x1, x2):
    """A high-frequency trigonometric testbed — short correlation length, so a stationary
    kernel has to work for its fit."""
    x1, x2 = np.asarray(x1, dtype=float), np.asarray(x2, dtype=float)
    return np.sin(10 * x1) + np.cos(8 * x2) - np.cos(6 * x1 * x2)


def booth_h(x1, x2):
    """BSV 'Representative': Booth's function minus its failure threshold of 200, on
    ``[-10, 5]^2``. One smooth connected failure region; ``h <= 0`` is the set."""
    x1, x2 = np.asarray(x1, dtype=float), np.asarray(x2, dtype=float)
    return (x1 + 2 * x2 - 7) ** 2 + (2 * x1 + x2 - 5) ** 2 - 200.0


def squares_h(x1, x2):
    """BSV 'Squares': two disjoint axis-aligned squares on ``[0, 10]^2``, as a Chebyshev
    signed distance so that ``h <= 0`` is the set.

    The designed-to-hurt case. Kinked corners are exactly what a stationary smooth kernel
    cannot represent, and one of the two regions is small enough to be missed entirely.
    """
    x1, x2 = np.asarray(x1, dtype=float), np.asarray(x2, dtype=float)
    big = np.maximum(np.abs(x1 - 1.8), np.abs(x2 - 1.8)) - 1.0
    small = np.maximum(np.abs(x1 - 8.6), np.abs(x2 - 8.6)) - 0.6
    return np.minimum(big, small)


def himmelblau_h(x1, x2):
    """BSV 'Mixture': Himmelblau's function minus 15, on ``[-6, 6]^2``. Four disjoint
    smooth blobs with steep walls; ``h <= 0`` is the set."""
    x1, x2 = np.asarray(x1, dtype=float), np.asarray(x2, dtype=float)
    return (x1**2 + x2 - 11) ** 2 + (x1 + x2**2 - 7) ** 2 - 15.0
