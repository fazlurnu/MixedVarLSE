"""The benchmark catalogue: each surface as a ready-to-run mixed-variable LSE problem.

    from mvlse.problems import get_problem

    p = get_problem("himmelblau_flip")
    result = run_lse(p.space, p, threshold=p.threshold)

Every problem is declared the way a user would declare their own: **named variables in
physical units**, a blackbox that reads those names, and a threshold. Nothing here uses a
private path into the library.

Three groups:

**The paper's own** (``goldstein``, ``branin``, ``augmented_branin``) — Eqs. (28)-(29),
(33)-(37) and Table 4 of ``references/GP2.pdf``, unchanged.

**Ordinal scale/shift** (``branin_mix``, ``wiggly``, ``booth``, ``squares``,
``himmelblau``) — a base surface varied across ordinal levels by an affine map,
``f = a(z) h(x) + b(z)``. Booth, Squares and Himmelblau come from the Bayesian
safety-validation set; at ``z1 = z2 = 0`` with ``tau = 0`` each reproduces that paper's
failure region exactly, and the other categories are ordinal variations of it.

**Sign-flipped categorical** (``*_flip``) — a third, *nominal* variable ``C`` with
``S = (1.00, -0.75, 1.15)`` acting on the standardised base, which is the paper's own
Branin construction (Eq. 29) bolted onto every surface. It makes ``corr(A, C) = +1`` and
``corr(A, B) = corr(B, C) = -1``: A and C are the same surface, B is its mirror image.

That last group is the discriminating one. No positive correlation function can express a
negative entry, so ``dc`` and ``gower`` are structurally locked out of the truth while the
hypersphere kernels are not. ``goldstein_graded`` is the control: there ``C`` shifts ``x1``
*inside* the polynomial, giving graded positive correlations that every method can at least
reach the right sign of.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from ..doe import grid_candidates, random_candidates
from ..space import Continuous, DesignSpace, Nominal, Ordinal, Point
from .functions import (
    augmented_branin_h,
    booth_h,
    branin_h,
    goldstein_h,
    himmelblau_h,
    squares_h,
    wiggly_h,
)

__all__ = ["PROBLEMS", "Problem", "get_problem", "list_problems"]

_FLIP_SCALE = (1.00, -0.75, 1.15)
"""``S_C`` — the ``-0.75`` is the paper's own coefficient from Eq. (29)."""

_FLIP_SHIFT = (0.00, 0.10, -0.10)
_FLIP_LEVELS = ("A", "B", "C")
_GRADED_OFFSET = (0.00, 0.30, -0.05)

BlackboxFn = Callable[[Sequence[Point]], np.ndarray]


@dataclass(frozen=True)
class Problem:
    """A benchmark surface, its design space and its threshold.

    The instance *is* the blackbox: ``problem(points)`` returns the response, so it can be
    handed straight to :func:`mvlse.run_lse`.
    """

    name: str
    space: DesignSpace
    function: BlackboxFn
    threshold: float
    greater_is_inside: bool = False
    reference: str = ""
    description: str = ""

    def __call__(self, points: Sequence[Point]) -> np.ndarray:
        return np.asarray(self.function(points), dtype=float)

    def truth(self, w: np.ndarray) -> np.ndarray:
        """True response at **encoded** points — what :mod:`mvlse.metrics` scores against."""
        return self(self.space.decode(w))

    def inside(self, w: np.ndarray) -> np.ndarray:
        """The true level set at encoded points."""
        values = self.truth(w)
        return values >= self.threshold if self.greater_is_inside else values <= self.threshold


# --------------------------------------------------------------------------- helpers
def _col(points: Sequence[Point], name: str) -> np.ndarray:
    return np.array([float(p[name]) for p in points], dtype=float)

def _cat(points: Sequence[Point], name: str, levels: Sequence) -> np.ndarray:
    index = {level: i for i, level in enumerate(levels)}
    return np.array([index[p[name]] for p in points], dtype=int)


def _reference_grid(space: DesignSpace) -> np.ndarray:
    """A deterministic set of points to calibrate a quantile threshold on."""
    if space.q <= 2:
        return grid_candidates(space, 21)
    return random_candidates(space, 512, seed=0)


def _quantile_threshold(space: DesignSpace, fn: BlackboxFn, quantile: float) -> float:
    values = np.asarray(fn(space.decode(_reference_grid(space))), dtype=float)
    return float(np.quantile(values, quantile))


def _two_continuous(name1: str, lo1: float, hi1: float, name2: str, lo2: float, hi2: float):
    return {name1: Continuous(lo1, hi1), name2: Continuous(lo2, hi2)}


_ORDINAL_PAIR = {"z1": Ordinal([0, 1, 2]), "z2": Ordinal([0, 1, 2])}
"""Two abstract three-level ordinal variables — the notebooks' scale/shift structure."""


def _scale_shift(base: Callable[[Sequence[Point]], np.ndarray], a1, a2, b1, b2, scale=1.0):
    """``f = (1 + a1 z1 + a2 z2) h(x) + (b1 z1 + b2 z2) * scale``."""

    def fn(points: Sequence[Point]) -> np.ndarray:
        z1, z2 = _col(points, "z1"), _col(points, "z2")
        return (1.0 + a1 * z1 + a2 * z2) * base(points) + (b1 * z1 + b2 * z2) * scale

    return fn


# ------------------------------------------------------------------- the paper's own
def goldstein(levels: Sequence[float] = (20.0, 50.0, 80.0)) -> Problem:
    """Paper Eqs. (36)-(37), Table 4 — 2 continuous, 2 ordinal, 9 categories.

    ``x3`` and ``x4`` take three discrete values each; the default is the paper's own
    Table 4. The notebooks that preceded this repo used narrower ranges (``{40, 50, 60}``,
    and ``{45, 50, 55}`` once a third variable was added) to keep every category populated
    — pass ``levels`` to reproduce those.
    """
    space = DesignSpace(
        {
            **_two_continuous("x1", 0.0, 100.0, "x2", 0.0, 100.0),
            "x3": Ordinal(list(levels)),
            "x4": Ordinal(list(levels)),
        }
    )

    def fn(points: Sequence[Point]) -> np.ndarray:
        return goldstein_h(
            _col(points, "x1"), _col(points, "x2"), _col(points, "x3"), _col(points, "x4")
        )

    return Problem(
        "goldstein", space, fn, _quantile_threshold(space, fn, 0.55),
        reference="Pelamatti et al. (2020), Eqs. (36)-(37), Table 4",
        description="Smooth quartic polynomial; the ordinals enter inside the function, "
                    "so their level correlations are graded rather than +-1.",
    )


def branin() -> Problem:
    """Paper Eqs. (28)-(29) — 2 continuous, 2 binary discrete, 4 categories.

    The paper's own anti-correlated benchmark: the four categories are ``h``, ``0.4h+1.1``,
    ``-0.75h+5.2`` and ``-0.5h-2.1``, which it describes as showing "pair-wise
    anti-correlation and a considerable relative offset".
    """
    space = DesignSpace(
        {
            **_two_continuous("x1", 0.0, 1.0, "x2", 0.0, 1.0),
            "z1": Ordinal([0, 1]),
            "z2": Ordinal([0, 1]),
        }
    )
    scales = {(0, 0): (1.0, 0.0), (0, 1): (0.4, 1.1), (1, 0): (-0.75, 5.2), (1, 1): (-0.5, -2.1)}

    def fn(points: Sequence[Point]) -> np.ndarray:
        h = branin_h(_col(points, "x1"), _col(points, "x2"))
        z1, z2 = _col(points, "z1").astype(int), _col(points, "z2").astype(int)
        a = np.array([scales[(a_, b_)][0] for a_, b_ in zip(z1, z2, strict=True)])
        b = np.array([scales[(a_, b_)][1] for a_, b_ in zip(z1, z2, strict=True)])
        return a * h + b

    return Problem(
        "branin", space, fn, _quantile_threshold(space, fn, 0.45),
        reference="Pelamatti et al. (2020), Eqs. (28)-(29)",
        description="The paper's own anti-correlated benchmark. Categories 1 and 3 move "
                    "against each other, which no positive correlation function can express.",
    )


def augmented_branin() -> Problem:
    """Paper Eqs. (33)-(35) — 10 continuous, 2 binary discrete, 4 categories.

    The paper's test of what happens as the *continuous* dimension grows while the discrete
    structure stays fixed. A lattice candidate set is hopeless at ``q = 10``, so the loop
    falls back to a quasi-random cloud automatically.
    """
    space = DesignSpace(
        {
            **{f"x{i}": Continuous(0.0, 1.0) for i in range(1, 11)},
            "z1": Ordinal([0, 1]),
            "z2": Ordinal([0, 1]),
        }
    )
    scales = {(0, 0): (1.0, 0.0), (0, 1): (0.4, 1.1), (1, 0): (-0.75, 5.2), (1, 1): (-0.5, -2.1)}

    def fn(points: Sequence[Point]) -> np.ndarray:
        x = np.column_stack([_col(points, f"x{i}") for i in range(1, 11)])
        h = augmented_branin_h(x)
        z1, z2 = _col(points, "z1").astype(int), _col(points, "z2").astype(int)
        a = np.array([scales[(a_, b_)][0] for a_, b_ in zip(z1, z2, strict=True)])
        b = np.array([scales[(a_, b_)][1] for a_, b_ in zip(z1, z2, strict=True)])
        return a * h + b

    return Problem(
        "augmented_branin", space, fn, _quantile_threshold(space, fn, 0.45),
        reference="Pelamatti et al. (2020), Eqs. (33)-(35)",
        description="10 continuous dimensions with the same 4-category structure.",
    )


# ------------------------------------------------------------- ordinal scale/shift
def branin_mix() -> Problem:
    """Branin (Eq. 28) with two three-level ordinal variables applied as scale and shift."""
    space = DesignSpace({**_two_continuous("x1", 0.0, 1.0, "x2", 0.0, 1.0), **_ORDINAL_PAIR})
    base = lambda pts: branin_h(_col(pts, "x1"), _col(pts, "x2"))
    fn = _scale_shift(base, -0.12, 0.06, 0.25, -0.15)
    return Problem(
        "branin_mix", space, fn, _quantile_threshold(space, fn, 0.45),
        reference="Pelamatti et al. (2020), Eq. (28) + ordinal scale/shift",
        description="Smooth, one curved valley.",
    )


def wiggly() -> Problem:
    """A short-correlation-length trigonometric surface with ordinal scale and shift."""
    space = DesignSpace({**_two_continuous("x1", 0.0, 1.0, "x2", 0.0, 1.0), **_ORDINAL_PAIR})
    base = lambda pts: wiggly_h(_col(pts, "x1"), _col(pts, "x2"))
    fn = _scale_shift(base, -0.10, 0.05, 0.30, -0.20)
    return Problem(
        "wiggly", space, fn, _quantile_threshold(space, fn, 0.50),
        reference="this repo",
        description="High-frequency; short correlation length makes every point expensive.",
    )


def booth() -> Problem:
    """BSV 'Representative' — Booth's function, one smooth connected failure region."""
    space = DesignSpace({**_two_continuous("x1", -10.0, 5.0, "x2", -10.0, 5.0), **_ORDINAL_PAIR})
    base = lambda pts: booth_h(_col(pts, "x1"), _col(pts, "x2"))
    fn = _scale_shift(base, -0.15, 0.05, -0.20, 0.12, scale=200.0)
    return Problem(
        "booth", space, fn, 0.0,
        reference="Moss et al., Bayesian Safety Validation, Sec. IV.A",
        description="At z1=z2=0 with tau=0 this is exactly the BSV paper's failure region.",
    )


def squares() -> Problem:
    """BSV 'Squares' — two disjoint squares, kinked boundary. The designed-to-hurt case."""
    space = DesignSpace({**_two_continuous("x1", 0.0, 10.0, "x2", 0.0, 10.0), **_ORDINAL_PAIR})
    base = lambda pts: squares_h(_col(pts, "x1"), _col(pts, "x2"))
    fn = _scale_shift(base, -0.10, 0.05, -0.20, 0.10)
    return Problem(
        "squares", space, fn, 0.0,
        reference="Moss et al., Bayesian Safety Validation, Sec. IV.A",
        description="Axis-aligned kinks and a small island — a stationary kernel's worst case.",
    )


def himmelblau() -> Problem:
    """BSV 'Mixture' — four disjoint smooth blobs with steep walls."""
    space = DesignSpace({**_two_continuous("x1", -6.0, 6.0, "x2", -6.0, 6.0), **_ORDINAL_PAIR})
    base = lambda pts: himmelblau_h(_col(pts, "x1"), _col(pts, "x2"))
    fn = _scale_shift(base, -0.15, 0.05, -0.25, 0.12, scale=15.0)
    return Problem(
        "himmelblau", space, fn, 0.0,
        reference="Moss et al., Bayesian Safety Validation, Sec. IV.A",
        description="Disjoint blobs, steep walls. The surface where the choice of discrete "
                    "correlation function matters most.",
    )


# --------------------------------------------------------- sign-flipped categorical
def _add_flip(base_problem: Problem, name: str, quantile: float) -> Problem:
    """Bolt the paper's Eq. (29) sign flip onto a base as a third, *nominal* variable."""
    base_space = base_problem.space
    space = DesignSpace(
        {
            **{n: base_space.variable(n) for n in base_space.names},
            "C": Nominal(list(_FLIP_LEVELS)),
        }
    )
    # Standardise the base over its own grid so S and T mean the same thing on every surface.
    reference = base_space.decode(_reference_grid(base_space))
    values = np.asarray(base_problem.function(reference), dtype=float)
    mean, spread = float(values.mean()), float(values.std())
    scale, shift = np.asarray(_FLIP_SCALE), np.asarray(_FLIP_SHIFT)

    def fn(points: Sequence[Point]) -> np.ndarray:
        c = _cat(points, "C", _FLIP_LEVELS)
        standardised = (np.asarray(base_problem.function(points), dtype=float) - mean) / spread
        return scale[c] * standardised + shift[c]

    return Problem(
        name, space, fn, _quantile_threshold(space, fn, quantile),
        reference=f"{base_problem.reference}; sign flip after Pelamatti et al. Eq. (29)",
        description="Levels A and C are the same surface (corr +1); B is its mirror image "
                    "(corr -1). Only the hypersphere kernels can represent that.",
    )


def goldstein_graded() -> Problem:
    """The control for the ``*_flip`` family: ``C`` shifts ``x1`` *inside* the polynomial.

    Correlations come out positive but graded, so ``dc`` and ``gower`` can at least reach
    the right sign. Any remaining gap to the hypersphere kernels is therefore not "because
    of the sign" — which is what makes this the honest comparison to quote alongside.
    """
    base = goldstein(levels=(45.0, 50.0, 55.0))
    space = DesignSpace(
        {**{n: base.space.variable(n) for n in base.space.names}, "C": Nominal(list(_FLIP_LEVELS))}
    )
    offsets = np.asarray(_GRADED_OFFSET)

    def fn(points: Sequence[Point]) -> np.ndarray:
        c = _cat(points, "C", _FLIP_LEVELS)
        # The offset enters x1 (0..100), so it is scaled to that range.
        return goldstein_h(
            _col(points, "x1") + 100.0 * offsets[c],
            _col(points, "x2"),
            _col(points, "x3"),
            _col(points, "x4"),
        )

    return Problem(
        "goldstein_graded", space, fn, _quantile_threshold(space, fn, 0.55),
        reference="Pelamatti et al. (2020), Eq. (36) with a categorical x1 offset",
        description="Graded positive level correlations — the control for the flip family.",
    )


PROBLEMS: dict[str, Callable[[], Problem]] = {
    # the paper's own
    "goldstein": goldstein,
    "branin": branin,
    "augmented_branin": augmented_branin,
    # ordinal scale/shift
    "branin_mix": branin_mix,
    "wiggly": wiggly,
    "booth": booth,
    "squares": squares,
    "himmelblau": himmelblau,
    # sign-flipped categorical (3 discrete variables, 27 categories)
    "goldstein_flip": lambda: _add_flip(
        goldstein(levels=(45.0, 50.0, 55.0)), "goldstein_flip", 0.55),
    "branin_flip": lambda: _add_flip(branin_mix(), "branin_flip", 0.45),
    "wiggly_flip": lambda: _add_flip(wiggly(), "wiggly_flip", 0.50),
    "squares_flip": lambda: _add_flip(squares(), "squares_flip", 0.35),
    "himmelblau_flip": lambda: _add_flip(himmelblau(), "himmelblau_flip", 0.35),
    "goldstein_graded": goldstein_graded,
}
"""Every benchmark, by name. Values are builders — a :class:`Problem` is constructed on
demand because calibrating a quantile threshold means evaluating the surface."""


def list_problems() -> list[str]:
    """The names :func:`get_problem` accepts."""
    return list(PROBLEMS)


def get_problem(name: str) -> Problem:
    """Build the named benchmark."""
    try:
        return PROBLEMS[name]()
    except KeyError:
        raise ValueError(
            f"unknown problem {name!r}; expected one of {list_problems()}"
        ) from None
