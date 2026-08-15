"""Where to measure next: the straddle, and how to pick a whole batch of them.

The level set is ``S = {w : f(w) <= tau}``, so the thing worth knowing is where the
boundary is — not where the function is smallest. The **straddle** (Bryan et al., 2005)
scores exactly that::

    a(w) = kappa * s(w) - |mu(w) - tau|

Large where the posterior is uncertain *and* sitting near the threshold. And it carries a
stopping rule that needs no ground truth, which is the reason to prefer it over plain
variance:

    a(w) > 0   <=>   mu - kappa*s  <  tau  <  mu + kappa*s
               <=>   "the confidence interval still straddles tau — I cannot tell
                      which side of the boundary this point is on"

so ``max_w a(w) < 0`` means every candidate is confidently classified. Confident is not the
same as correct, but it is a signal you can compute in the field, where the truth is exactly
what you do not have.

**Batching.** With an expensive blackbox you want several points per round, and the naive
top-``k`` of one acquisition field returns ``k`` neighbours of the same spot. :func:`select_batch`
does greedy local penalisation instead: take the argmax, blank out its neighbourhood *within
the same category*, repeat. Cheap, and enough to keep a round's points from piling up on one
patch of boundary.
"""
from __future__ import annotations

import numpy as np

__all__ = ["DEFAULT_KAPPA", "select_batch", "straddle"]

DEFAULT_KAPPA = 1.96
"""``z_{0.975}`` — so ``a > 0`` marks a two-sided 95% interval that contains ``tau``."""


def straddle(
    mu: np.ndarray, sd: np.ndarray, tau: float, kappa: float = DEFAULT_KAPPA
) -> np.ndarray:
    """``kappa * sd - |mu - tau|``, elementwise."""
    mu = np.asarray(mu, dtype=float)
    sd = np.asarray(sd, dtype=float)
    if mu.shape != sd.shape:
        raise ValueError(f"mu has shape {mu.shape} but sd has shape {sd.shape}")
    return kappa * sd - np.abs(mu - tau)


def select_batch(
    scores: np.ndarray,
    candidates: np.ndarray,
    n_continuous: int,
    k: int = 1,
    *,
    exclude: set[int] | None = None,
    min_distance: float = 0.0,
) -> list[int]:
    """Indices of up to ``k`` high-scoring, spread-out candidates.

    Args:
        scores: acquisition value per candidate.
        candidates: the encoded candidate matrix the scores belong to.
        n_continuous: ``space.q`` — where the continuous block ends and the level
            indices begin.
        k: how many points this round.
        exclude: candidate indices already spent; never returned again.
        min_distance: after a pick, candidates in the **same category** within this
            Euclidean distance (in encoded continuous coordinates) are blanked out.
            ``0`` disables the spreading and gives a plain top-``k``.

    Returns:
        The chosen indices, best first. Shorter than ``k`` if the penalisation ran the
        pool dry — which is itself informative: there was nothing else worth measuring.
    """
    scores = np.asarray(scores, dtype=float).copy()
    candidates = np.atleast_2d(np.asarray(candidates, dtype=float))
    if len(scores) != len(candidates):
        raise ValueError(f"{len(scores)} scores for {len(candidates)} candidates")
    if k < 1:
        return []
    if exclude:
        scores[list(exclude)] = -np.inf

    continuous = candidates[:, :n_continuous]
    levels = candidates[:, n_continuous:]
    picked: list[int] = []
    for _ in range(k):
        best = int(np.argmax(scores))
        if not np.isfinite(scores[best]):
            break
        picked.append(best)
        scores[best] = -np.inf
        if min_distance > 0:
            same_category = (
                np.all(levels == levels[best], axis=1)
                if levels.shape[1]
                else np.ones(len(scores), dtype=bool)
            )
            near = np.linalg.norm(continuous - continuous[best], axis=1) < min_distance
            scores[same_category & near] = -np.inf
    return picked
