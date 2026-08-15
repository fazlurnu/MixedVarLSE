"""Ordinary Kriging — paper Eqs. (9)-(14), with optional known per-point noise.

The model is the paper's Eq. (6), a constant trend plus a stationary stochastic term::

    Y(w) = beta + eps(w),      Cov(w_i, w_j) = sigma^2 R*(w_i, w_j)

``R*`` is whichever mixed kernel you picked (:mod:`mvlse.kernels`); everything in this
module is indifferent to which. Two of the three parameter groups are solved for in closed
form rather than searched (Eq. 10), leaving only the correlation hyperparameters to the
optimiser:

    beta     = (1' R^-1 y) / (1' R^-1 1)
    sigma^2  = (y - 1 beta)' R^-1 (y - 1 beta) / n

and the concentrated log-likelihood to minimise is

    L(theta) = 1/2 ( n log sigma^2 + log det R )
                     ^^^^^^^^^^^^   ^^^^^^^^^^^
                     fit the data   stay humble

Prediction is then Eqs. (12)-(14).

**Noisy blackboxes.** When the blackbox reports a standard error per point, those variances
go on the covariance diagonal as *fixed* known quantities::

    C = sigma^2 R* + diag(se_i^2)

``sigma^2`` no longer profiles out in closed form, so it joins the search as one extra
parameter. Predictions stay predictions of the **latent, noise-free** response, which is
what a level set is about and what the straddle in :mod:`mvlse.acquisition` needs.

**Standardisation.** ``y`` is centred and scaled internally, and ``se`` with it, so that the
``sigma^2`` bounds mean the same thing whether your blackbox returns probabilities or
stresses in megapascals. Everything this module returns is back in your original units.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

from .kernels import MixedKernel

__all__ = ["LOG_SIGMA2_BOUNDS", "NUGGET", "FittedGP", "fit"]

NUGGET = 1e-8
"""Added to the correlation diagonal before factorising. A numerical device, not a model
of noise — for that, have the blackbox report a standard error."""

LOG_SIGMA2_BOUNDS = (-4.0, 2.0)
"""``log10(sigma^2)`` bounds, on the *standardised* response, so ``0`` means "the signal
explains the whole spread of y"."""

_FAILED = 1e10


@dataclass(frozen=True)
class _Standardiser:
    """Centre and scale ``y``; carry ``se`` along with it."""

    mean: float
    scale: float

    @classmethod
    def of(cls, y: np.ndarray) -> _Standardiser:
        scale = float(np.std(y))
        return cls(float(np.mean(y)), scale if scale > 1e-12 else 1.0)

    def forward(self, y: np.ndarray) -> np.ndarray:
        return (y - self.mean) / self.scale

    def inverse(self, y: np.ndarray) -> np.ndarray:
        return self.mean + self.scale * y


def _factor(cov: np.ndarray):
    """Cholesky with the nugget already applied; ``None`` if the matrix is not PD."""
    cov = cov.copy()
    cov[np.diag_indices_from(cov)] += NUGGET
    try:
        return cho_factor(cov, lower=True)
    except (np.linalg.LinAlgError, ValueError):
        return None


def _profile_beta(chol, y: np.ndarray) -> float:
    """Eq. (10), left half: the generalised-least-squares constant trend."""
    ones = np.ones(len(y))
    return float((ones @ cho_solve(chol, y)) / (ones @ cho_solve(chol, ones)))


def _log_det(chol) -> float:
    """``log det`` from the factor — ``2 sum log L_ii``, which never overflows."""
    return 2.0 * float(np.log(np.diag(chol[0])).sum())


def _negative_log_likelihood(
    params: np.ndarray,
    kernel: MixedKernel,
    w: np.ndarray,
    y: np.ndarray,
    noise: np.ndarray | None,
) -> float:
    """Eq. (9), concentrated over ``beta`` (and over ``sigma^2`` when there is no noise)."""
    n = len(y)
    if noise is None:
        chol = _factor(kernel(params, w, w))
        if chol is None:
            return _FAILED
        beta = _profile_beta(chol, y)
        resid = y - beta
        sigma2 = max(float(resid @ cho_solve(chol, resid)) / n, 1e-12)
        return 0.5 * (n * np.log(sigma2) + _log_det(chol))

    sigma2 = 10.0 ** params[-1]
    chol = _factor(sigma2 * kernel(params[:-1], w, w) + np.diag(noise))
    if chol is None:
        return _FAILED
    beta = _profile_beta(chol, y)
    resid = y - beta
    return 0.5 * (float(resid @ cho_solve(chol, resid)) + _log_det(chol))


@dataclass(frozen=True)
class FittedGP:
    """A fitted model bound to its training data. Prediction is closed form."""

    kernel: MixedKernel
    params: np.ndarray
    w: np.ndarray = field(repr=False)
    y: np.ndarray = field(repr=False)
    noise: np.ndarray | None = field(repr=False)
    beta: float
    sigma2: float
    log_likelihood: float
    _scaler: _Standardiser = field(repr=False)
    _chol: tuple = field(repr=False)
    _alpha: np.ndarray = field(repr=False)

    @property
    def kernel_params(self) -> np.ndarray:
        """The correlation hyperparameters, without the trailing ``log10(sigma^2)`` that
        the noisy case appends. This is what :meth:`MixedKernel.level_correlations` wants."""
        return self.params[:-1] if self.noise is not None else self.params

    @property
    def n_train(self) -> int:
        return len(self.y)

    def predict(self, w_new: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Eqs. (12)-(14): posterior mean and standard deviation of the **latent**
        response at ``w_new``, in your original units."""
        w_new = np.atleast_2d(np.asarray(w_new, dtype=float))
        noisy = self.noise is not None
        # psi holds the cross-*covariance* when the diagonal carries noise (so it must
        # include sigma^2), and the cross-*correlation* when sigma^2 is factored out.
        psi = self.kernel(self.kernel_params, w_new, self.w)
        if noisy:
            psi = self.sigma2 * psi

        mu = self.beta + psi @ self._alpha
        explained = np.sum(psi.T * cho_solve(self._chol, psi.T), axis=0)
        var = self.sigma2 - explained if noisy else self.sigma2 * (1.0 - explained)

        return self._scaler.inverse(mu), self._scaler.scale * np.sqrt(np.clip(var, 0.0, None))

    def level_correlations(self) -> dict[str, np.ndarray]:
        """What the fit decided each discrete variable's levels mean to each other."""
        return self.kernel.level_correlations(self.kernel_params)

    def category_correlations(self) -> np.ndarray:
        """The implied ``m x m`` correlation between categories."""
        return self.kernel.category_correlations(self.kernel_params)


def fit(
    kernel: MixedKernel,
    w: np.ndarray,
    y: np.ndarray,
    se: np.ndarray | None = None,
    *,
    rng: np.random.Generator | None = None,
    n_starts: int = 6,
    warm_start: np.ndarray | None = None,
) -> FittedGP:
    """Maximum-likelihood fit by multi-start L-BFGS-B.

    Args:
        kernel: the mixed correlation function to fit.
        w: encoded training inputs, ``(n, q + r)`` — see :meth:`DesignSpace.encode`.
        y: training responses, ``(n,)``.
        se: per-point standard errors. ``None`` or all-zero means deterministic, and
            ``sigma^2`` is then profiled out instead of searched.
        rng: source of the random restarts; reproducible when you pass a seeded one.
        n_starts: random restarts, on top of the midpoint and any ``warm_start``.
        warm_start: a previous fit's ``params``. Worth passing inside an active loop —
            one extra local search from the last optimum is much cheaper than a fresh
            multi-start and is usually where the new optimum is.

    Raises:
        ValueError: if the shapes disagree or every restart failed to factorise.
    """
    w = np.atleast_2d(np.asarray(w, dtype=float))
    y = np.asarray(y, dtype=float).ravel()
    if len(w) != len(y):
        raise ValueError(f"{len(w)} inputs but {len(y)} responses")
    if len(y) < 2:
        raise ValueError(f"need at least 2 training points to fit, got {len(y)}")

    scaler = _Standardiser.of(y)
    y_scaled = scaler.forward(y)

    noise: np.ndarray | None = None
    if se is not None:
        se = np.asarray(se, dtype=float).ravel()
        if len(se) != len(y):
            raise ValueError(f"{len(se)} standard errors but {len(y)} responses")
        if np.any(se > 0):
            noise = (se / scaler.scale) ** 2

    bounds = list(kernel.bounds)
    if noise is not None:
        bounds.append(LOG_SIGMA2_BOUNDS)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    rng = rng if rng is not None else np.random.default_rng()
    starts = [lo + (hi - lo) * rng.random(len(bounds)) for _ in range(max(n_starts, 0))]
    starts.append(0.5 * (lo + hi))
    if warm_start is not None:
        warm = np.asarray(warm_start, dtype=float)
        if len(warm) == len(bounds):
            starts.append(np.clip(warm, lo + 1e-9, hi - 1e-9))

    best = None
    for x0 in starts:
        result = minimize(
            _negative_log_likelihood,
            x0,
            args=(kernel, w, y_scaled, noise),
            method="L-BFGS-B",
            bounds=bounds,
        )
        if np.isfinite(result.fun) and (best is None or result.fun < best.fun):
            best = result
    if best is None or best.fun >= _FAILED:
        raise ValueError(
            f"could not fit {kernel.name}: every restart produced a non-positive-definite "
            "correlation matrix. Duplicate training points are the usual cause."
        )

    params = best.x
    if noise is None:
        chol = _factor(kernel(params, w, w))
        beta = _profile_beta(chol, y_scaled)
        resid = y_scaled - beta
        sigma2 = max(float(resid @ cho_solve(chol, resid)) / len(y), 1e-12)
    else:
        sigma2 = 10.0 ** params[-1]
        chol = _factor(sigma2 * kernel(params[:-1], w, w) + np.diag(noise))
        beta = _profile_beta(chol, y_scaled)
        resid = y_scaled - beta

    return FittedGP(
        kernel=kernel,
        params=params,
        w=w.copy(),
        y=y.copy(),
        noise=noise,
        beta=beta,
        sigma2=sigma2,
        log_likelihood=-float(best.fun),
        _scaler=scaler,
        _chol=chol,
        _alpha=cho_solve(chol, resid),
    )
