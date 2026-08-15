"""The design space: what a user's variables are, and what the model gets to see.

This is the boundary of the library. On the user's side of it there are **named variables
in physical units** — ``range_m = 840.0``, ``material = "titanium"``. On the model's side
there is a plain float matrix ``W`` of shape ``(n, q + r)``: the ``q`` continuous columns
first, then the ``r`` discrete columns holding *level indices*.

    W = [ x_1 ... x_q | z_1 ... z_r ]        (paper Section 2: w = {x, z})

Encoding is the library's job, not yours. But it is **visible and optional**, because a
surrogate that silently rescales its inputs is a surrogate you cannot debug:

* :meth:`DesignSpace.encode` and :meth:`DesignSpace.decode` are public and round-trip.
* :meth:`DesignSpace.describe` prints exactly what each variable becomes.
* ``normalize=False`` hands the continuous columns through in physical units. The default
  is ``True`` because the kernel hyperparameter bounds in :mod:`mvlse.correlation` are
  calibrated for inputs on the unit cube; turn it off and you should widen them.

The paper's notation, kept throughout (GP2.pdf Section 2):

===========  ==========================================================
``q``        number of continuous variables
``r``        number of discrete variables
``b_k``      number of levels of discrete variable ``k``
``m``        number of categories, ``prod(b_k)``
===========  ==========================================================
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np

__all__ = [
    "Continuous",
    "DesignSpace",
    "Nominal",
    "Ordinal",
    "Point",
    "Variable",
]


class Point(Mapping[str, Any]):
    """One design point in **physical units**, keyed by the user's own variable names.

    Supports both ``p["range_m"]`` and ``p.range_m``. This is what a blackbox receives;
    it never sees a normalized number or a level index unless it asks the space for one.
    """

    __slots__ = ("_values",)

    def __init__(self, values: Mapping[str, Any]) -> None:
        self._values = dict(values)

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def __getattr__(self, key: str) -> Any:
        # Underscored names are never variable names, and refusing them here is what stops
        # `self._values` recursing forever while unpickling, before the slot is populated.
        # Point must stay picklable: mvlse.blackbox.parallel() sends them to worker processes.
        if key.startswith("_"):
            raise AttributeError(key)
        try:
            return self._values[key]
        except KeyError:
            raise AttributeError(key) from None

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:
        inner = ", ".join(f"{k}={v!r}" for k, v in self._values.items())
        return f"Point({inner})"


@dataclass(frozen=True)
class Variable:
    """Base class. Subclasses declare a variable's *nature*, which picks its encoding."""

    def __post_init__(self) -> None:  # pragma: no cover - overridden where needed
        pass


@dataclass(frozen=True)
class Continuous(Variable):
    """A real-valued input on ``[lower, upper]``."""

    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not self.upper > self.lower:
            raise ValueError(f"Continuous bounds must satisfy upper > lower, got {self}")

    @property
    def span(self) -> float:
        return float(self.upper - self.lower)

    def to_unit(self, value: np.ndarray) -> np.ndarray:
        return (np.asarray(value, dtype=float) - self.lower) / self.span

    def from_unit(self, unit: np.ndarray) -> np.ndarray:
        return self.lower + self.span * np.asarray(unit, dtype=float)


@dataclass(frozen=True)
class Discrete(Variable):
    """Base for the two discrete natures. ``levels`` are the values the user works in."""

    levels: tuple[Any, ...]

    def __post_init__(self) -> None:
        if len(self.levels) < 2:
            raise ValueError(f"a discrete variable needs >= 2 levels, got {self.levels!r}")
        if len(set(self.levels)) != len(self.levels):
            raise ValueError(f"discrete levels must be distinct, got {self.levels!r}")

    @property
    def n_levels(self) -> int:
        """``b_k`` in the paper."""
        return len(self.levels)

    def index_of(self, value: Any) -> int:
        return self.levels.index(value)


@dataclass(frozen=True)
class Ordinal(Discrete):
    """A discrete input whose levels have a **meaningful order** — 'small/medium/big',
    a mesh refinement, a number of engines.

    Order is information, and only some correlation functions can use it: direct
    conversion (:mod:`mvlse.kernels`, Eq. 15) builds it in by construction, Gower
    (Eq. 18) is blind to it, and the hypersphere kernels (Eqs. 19-24) can *learn* it.
    """

    def __init__(self, levels: Sequence[Any]) -> None:
        object.__setattr__(self, "levels", tuple(levels))
        self.__post_init__()

    def positions(self) -> np.ndarray:
        """Level positions in ``[0, 1]``, used by direct-conversion coding.

        Numeric levels keep their **spacing** (so ``[1, 10, 100]`` is not equally spaced);
        non-numeric levels fall back to their rank. Spacing only rescales into ``theta``,
        but the *ordering* it implies is a real modelling assumption — see the DC entry in
        ``docs/FORMULA_SHEET.md``.
        """
        try:
            raw = np.asarray(self.levels, dtype=float)
        except (TypeError, ValueError):
            raw = np.arange(self.n_levels, dtype=float)
        if raw.max() == raw.min():  # pragma: no cover - guarded by distinct levels
            return np.zeros(self.n_levels)
        return (raw - raw.min()) / (raw.max() - raw.min())


@dataclass(frozen=True)
class Nominal(Discrete):
    """A discrete input whose levels have **no order** — a material, an oxidant, a colour.

    Imposing an order on these is exactly the bias the paper warns about in Section 3.3.
    """

    def __init__(self, levels: Sequence[Any]) -> None:
        object.__setattr__(self, "levels", tuple(levels))
        self.__post_init__()

    def positions(self) -> np.ndarray:
        """Rank positions. Meaningless by construction — direct conversion uses them
        anyway, and that is precisely its documented failure mode."""
        return np.linspace(0.0, 1.0, self.n_levels)


class DesignSpace:
    """An ordered collection of named variables, and the encoding they imply.

    Declare it the way you would declare an experiment — by name::

        space = DesignSpace({
            "range_m":  Continuous(100.0, 1200.0),
            "p_rx":     Continuous(0.0, 1.0),
            "pos_ci95": Ordinal([3.0, 10.0, 30.0, 92.6]),
            "material": Nominal(["steel", "alu", "titanium"]),
        })

    Continuous variables are always presented to the model first, discrete ones after,
    regardless of the order you declare them in; :attr:`continuous_names` and
    :attr:`discrete_names` tell you the resulting column order.
    """

    def __init__(
        self,
        variables: Mapping[str, Variable] | Sequence[tuple[str, Variable]],
        *,
        normalize: bool = True,
    ) -> None:
        items = list(variables.items()) if isinstance(variables, Mapping) else list(variables)
        if not items:
            raise ValueError("a design space needs at least one variable")
        for name, var in items:
            if not isinstance(var, (Continuous, Discrete)):
                raise TypeError(f"{name!r}: expected Continuous/Ordinal/Nominal, got {var!r}")

        self.normalize = bool(normalize)
        self._continuous: dict[str, Continuous] = {
            n: v for n, v in items if isinstance(v, Continuous)
        }
        self._discrete: dict[str, Discrete] = {n: v for n, v in items if isinstance(v, Discrete)}
        if not self._continuous:
            raise ValueError(
                "a design space needs at least one Continuous variable — with none, every "
                "category is a single point and there is no level set to estimate"
            )

    # ------------------------------------------------------------------ shape
    @property
    def continuous_names(self) -> tuple[str, ...]:
        return tuple(self._continuous)

    @property
    def discrete_names(self) -> tuple[str, ...]:
        return tuple(self._discrete)

    @property
    def names(self) -> tuple[str, ...]:
        """Column order of the encoded matrix: continuous first, then discrete."""
        return self.continuous_names + self.discrete_names

    @property
    def q(self) -> int:
        """Number of continuous variables."""
        return len(self._continuous)

    @property
    def r(self) -> int:
        """Number of discrete variables."""
        return len(self._discrete)

    @property
    def n_levels(self) -> tuple[int, ...]:
        """``(b_1, ..., b_r)``."""
        return tuple(v.n_levels for v in self._discrete.values())

    @property
    def m(self) -> int:
        """Number of categories, ``m = prod(b_k)``. ``1`` when there are no discrete vars."""
        return int(np.prod(self.n_levels)) if self.r else 1

    @property
    def categories(self) -> list[tuple[int, ...]]:
        """Every combination of level indices, in odometer order."""
        return [tuple(c) for c in product(*(range(b) for b in self.n_levels))]

    def variable(self, name: str) -> Variable:
        if name in self._continuous:
            return self._continuous[name]
        return self._discrete[name]

    @property
    def continuous_bounds(self) -> np.ndarray:
        """``(q, 2)`` array of physical ``[lower, upper]`` per continuous variable."""
        return np.array([[v.lower, v.upper] for v in self._continuous.values()], dtype=float)

    # --------------------------------------------------------------- encoding
    def encode(self, points: Sequence[Mapping[str, Any]]) -> np.ndarray:
        """Physical named points -> the model matrix ``W``, shape ``(n, q + r)``.

        Continuous columns are mapped to ``[0, 1]`` when ``normalize`` is set, else passed
        through unchanged. Discrete columns become **level indices** stored as floats.
        """
        if len(points) == 0:
            return np.empty((0, self.q + self.r))
        rows = np.empty((len(points), self.q + self.r), dtype=float)
        for j, (name, var) in enumerate(self._continuous.items()):
            raw = np.array([float(p[name]) for p in points])
            rows[:, j] = var.to_unit(raw) if self.normalize else raw
        for j, (name, var) in enumerate(self._discrete.items(), start=self.q):
            rows[:, j] = [float(var.index_of(p[name])) for p in points]
        return rows

    def decode(self, w: np.ndarray) -> list[Point]:
        """The model matrix ``W`` -> physical named points. Inverse of :meth:`encode`."""
        w = np.atleast_2d(np.asarray(w, dtype=float))
        if w.shape[1] != self.q + self.r:
            raise ValueError(f"expected {self.q + self.r} columns, got {w.shape[1]}")
        out: list[Point] = []
        for row in w:
            values: dict[str, Any] = {}
            for j, (name, var) in enumerate(self._continuous.items()):
                values[name] = float(var.from_unit(row[j]) if self.normalize else row[j])
            for j, (name, var) in enumerate(self._discrete.items(), start=self.q):
                values[name] = var.levels[round(row[j])]
            out.append(Point(values))
        return out

    def category_of(self, w: np.ndarray) -> np.ndarray:
        """Flat category index ``0..m-1`` for each encoded row (odometer over ``z``)."""
        w = np.atleast_2d(np.asarray(w, dtype=float))
        if self.r == 0:
            return np.zeros(len(w), dtype=int)
        idx = np.zeros(len(w), dtype=int)
        for j, b in enumerate(self.n_levels):
            idx = idx * b + w[:, self.q + j].astype(int)
        return idx

    def continuous_only(self) -> DesignSpace:
        """The same space with the discrete variables dropped.

        Category-wise Kriging (paper Section 3.2) is ``m`` fits over *this* space — one per
        category, nothing shared — so it is the sub-space that method works in.
        """
        return DesignSpace(dict(self._continuous), normalize=self.normalize)

    def unit_bounds(self) -> np.ndarray:
        """``(q, 2)`` bounds of the *encoded* continuous block — what samplers work in."""
        if self.normalize:
            return np.tile(np.array([0.0, 1.0]), (self.q, 1))
        return self.continuous_bounds

    # ------------------------------------------------------------- reporting
    def describe(self) -> str:
        """A table of what the model actually sees. Print this before you trust a fit."""
        lines = [
            f"DesignSpace: q={self.q} continuous, r={self.r} discrete, m={self.m} categories",
            f"  normalize = {self.normalize}"
            + ("  (continuous columns mapped to [0, 1])" if self.normalize
               else "  (continuous columns passed through in physical units)"),
            "",
            f"  {'column':<6} {'name':<16} {'nature':<10} {'physical':<28} what the model sees",
            f"  {'-' * 6} {'-' * 16} {'-' * 10} {'-' * 28} {'-' * 28}",
        ]
        for j, (name, var) in enumerate(self._continuous.items()):
            phys = f"[{var.lower:g}, {var.upper:g}]"
            seen = "[0, 1]" if self.normalize else phys
            lines.append(f"  {j:<6} {name:<16} {'continuous':<10} {phys:<28} {seen}")
        for j, (name, var) in enumerate(self._discrete.items(), start=self.q):
            nature = "ordinal" if isinstance(var, Ordinal) else "nominal"
            phys = ", ".join(str(x) for x in var.levels)
            phys = phys if len(phys) <= 27 else phys[:24] + "..."
            lines.append(
                f"  {j:<6} {name:<16} {nature:<10} {phys:<28} level index 0..{var.n_levels - 1}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"DesignSpace(q={self.q}, r={self.r}, m={self.m}, "
            f"names={self.names!r}, normalize={self.normalize})"
        )
