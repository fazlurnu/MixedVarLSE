"""Scoring a level-set estimate against a known truth.

Only for benchmarks: on a real problem you do not have the truth, which is what the
straddle's own maximum is for (:mod:`mvlse.acquisition`).

With ``S = {w : f(w) <= tau}`` and ``S_hat = {w : mu(w) <= tau}`` over a test grid::

    TP = |S and S_hat|      FP = |S^c and S_hat|      FN = |S and S_hat^c|

    F1 = 2 TP / (2 TP + FP + FN)

**Report accuracy and F1 together.** True negatives never enter F1, so a model that
predicts "nothing is in the set" scores F1 = 0 however large the domain — which is the
honest answer. Accuracy alone would reward it for being right about the empty space, and
on a small level set that is most of the space.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

__all__ = ["LevelSetScore", "score_level_set"]


@dataclass(frozen=True)
class LevelSetScore:
    """How well a predicted mean recovers a known level set."""

    accuracy: float
    f1: float
    precision: float
    recall: float
    rmse: float
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int

    def as_dict(self) -> dict[str, float]:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"acc={self.accuracy:.1%}  F1={self.f1:.3f}  "
            f"prec={self.precision:.3f}  rec={self.recall:.3f}  RMSE={self.rmse:.4g}"
        )


def score_level_set(
    mu: np.ndarray, truth: np.ndarray, tau: float, *, greater_is_inside: bool = False
) -> LevelSetScore:
    """Compare a predicted mean against known function values at the same points.

    Args:
        mu: predicted mean, one per test point.
        truth: **true function values** at those points (not booleans).
        tau: the threshold.
        greater_is_inside: by default the level set is ``f <= tau``. Set this when your
            set of interest is ``f >= tau`` instead — for a failure probability you
            usually want the low side, for a stress margin the high side.

    Returns:
        A :class:`LevelSetScore`. ``rmse`` is over the values, everything else over the
        induced classification.
    """
    mu = np.asarray(mu, dtype=float).ravel()
    truth = np.asarray(truth, dtype=float).ravel()
    if mu.shape != truth.shape:
        raise ValueError(f"mu has shape {mu.shape} but truth has shape {truth.shape}")
    if mu.size == 0:
        raise ValueError("nothing to score: empty test set")

    inside_true = truth >= tau if greater_is_inside else truth <= tau
    inside_pred = mu >= tau if greater_is_inside else mu <= tau

    tp = int(np.sum(inside_pred & inside_true))
    fp = int(np.sum(inside_pred & ~inside_true))
    fn = int(np.sum(~inside_pred & inside_true))
    tn = int(np.sum(~inside_pred & ~inside_true))

    denominator = 2 * tp + fp + fn
    return LevelSetScore(
        accuracy=float(np.mean(inside_pred == inside_true)),
        f1=(2.0 * tp / denominator) if denominator else 0.0,
        precision=(tp / (tp + fp)) if (tp + fp) else 0.0,
        recall=(tp / (tp + fn)) if (tp + fn) else 0.0,
        rmse=float(np.sqrt(np.mean((mu - truth) ** 2))),
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
        true_negative=tn,
    )
