"""A resumable cache of blackbox evaluations, one JSON object per line.

Point of it: if your blackbox takes minutes, a crashed or extended run must not pay for the
same point twice. Rerunning a study with a bigger budget then costs only the new points.

Keyed on the **encoded** coordinates rounded to a fixed number of decimals, because that is
the identity the loop actually works in — two candidates that encode to the same row are
the same evaluation. The physical values and the response are stored alongside, so the file
is readable on its own and survives being loaded by something that is not this library.

Nothing else in the library requires a store; pass ``None`` and every point is simply
evaluated as it comes.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .blackbox import Blackbox, evaluate
from .space import DesignSpace

__all__ = ["EvaluationStore"]

_KEY_DECIMALS = 9


class EvaluationStore:
    """An append-only JSONL cache of ``(encoded point) -> (y, se)``."""

    def __init__(self, path: str | Path, space: DesignSpace) -> None:
        self.path = Path(path)
        self.space = space
        self._entries: dict[tuple[float, ...], tuple[float, float]] = {}
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        with self.path.open() as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    self._entries[tuple(record["w"])] = (
                        float(record["y"]),
                        float(record.get("se", 0.0)),
                    )
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    raise ValueError(
                        f"{self.path}:{line_no} is not a valid evaluation record: {exc}"
                    ) from exc

    @staticmethod
    def _key(row: np.ndarray) -> tuple[float, ...]:
        return tuple(np.round(np.asarray(row, dtype=float), _KEY_DECIMALS))

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, row: np.ndarray) -> bool:
        return self._key(row) in self._entries

    def evaluate(self, w: np.ndarray, blackbox: Blackbox) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(y, se)`` for every row of ``w``, calling ``blackbox`` only for the
        rows this store has never seen."""
        w = np.atleast_2d(np.asarray(w, dtype=float))
        keys = [self._key(row) for row in w]
        missing = [i for i, key in enumerate(keys) if key not in self._entries]

        if missing:
            points = self.space.decode(w[missing])
            y_new, se_new = evaluate(blackbox, points)
            with self.path.open("a") as handle:
                for slot, y_i, se_i, point in zip(missing, y_new, se_new, points, strict=True):
                    self._entries[keys[slot]] = (float(y_i), float(se_i))
                    handle.write(
                        json.dumps(
                            {
                                "w": list(keys[slot]),
                                "y": float(y_i),
                                "se": float(se_i),
                                "point": {k: _plain(v) for k, v in point.items()},
                            }
                        )
                        + "\n"
                    )

        pairs = [self._entries[key] for key in keys]
        return (
            np.array([p[0] for p in pairs], dtype=float),
            np.array([p[1] for p in pairs], dtype=float),
        )


def _plain(value: object) -> object:
    """Make a level value JSON-serialisable without losing what it was."""
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
