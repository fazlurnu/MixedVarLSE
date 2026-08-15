"""mvlse — mixed-variable level-set estimation for black-box simulators.

You have a function you cannot differentiate, cannot see inside, and can only afford to
call a few dozen times. Some of its inputs are numbers and some are choices. You want to
know **where it crosses a threshold** — the failure boundary, the safe envelope, the
feasible region — not where it is smallest.

    from mvlse import Continuous, DesignSpace, Nominal, Ordinal, run_lse

    space = DesignSpace({
        "range_m":  Continuous(100.0, 1200.0),
        "p_rx":     Continuous(0.0, 1.0),
        "pos_ci95": Ordinal([3.0, 10.0, 30.0, 92.6]),
        "material": Nominal(["steel", "alu", "titanium"]),
    })

    def my_simulator(points):
        return [run(p["range_m"], p["p_rx"], p["pos_ci95"], p["material"]) for p in points]

    result = run_lse(space, my_simulator, threshold=1e-3, method="hs_dim")
    print(result.summary())
    result.to_dataframe()

The Gaussian-process machinery follows Pelamatti, Brevault, Balesdent, Talbi & Guerin
(2020), *Overview and Comparison of Gaussian Process-Based Surrogate Models for Mixed
Continuous and Discrete Variables* — ``references/GP2.pdf`` — and keeps its notation and
equation numbers, so the code and the paper can be read side by side. All five of its
methods are implemented and interchangeable:

===================  =========  =====================================================
``category_wise``    Sec. 3.2   one independent GP per category — the reference
``dc``               Eq. (15)   direct conversion: each level becomes a number
``gower``            Eq. (18)   Gower distance: a 0/1 mismatch score
``hs_full``          Eq. (19)   hypersphere: one free correlation per category pair
``hs_dim``           Eq. (23)   dimension-wise hypersphere — the default, and the best
===================  =========  =====================================================

``docs/FORMULA_SHEET.md`` puts every equation next to the line that implements it.
"""
from .acquisition import DEFAULT_KAPPA, select_batch, straddle
from .blackbox import Blackbox, Observation, evaluate, parallel, pointwise
from .correlation import CORRELATIONS, Correlation, get_correlation
from .doe import grid_candidates, random_candidates, stratified_doe
from .gp import FittedGP, fit
from .hypersphere import n_angles, pdude
from .kernels import KERNEL_NAMES, MixedKernel, make_kernel
from .lse import LseConfig, LseResult, Round, run_lse
from .metrics import LevelSetScore, score_level_set
from .models import MODEL_NAMES, CategoryWiseGP, MixedGP, Surrogate, fit_surrogate
from .space import Continuous, DesignSpace, Nominal, Ordinal, Point, Variable
from .store import EvaluationStore

__version__ = "0.1.0"

__all__ = [
    "CORRELATIONS",
    "DEFAULT_KAPPA",
    "KERNEL_NAMES",
    "MODEL_NAMES",
    "Blackbox",
    "CategoryWiseGP",
    "Continuous",
    "Correlation",
    "DesignSpace",
    "EvaluationStore",
    "FittedGP",
    "LevelSetScore",
    "LseConfig",
    "LseResult",
    "MixedGP",
    "MixedKernel",
    "Nominal",
    "Observation",
    "Ordinal",
    "Point",
    "Round",
    "Surrogate",
    "Variable",
    "evaluate",
    "fit",
    "fit_surrogate",
    "get_correlation",
    "grid_candidates",
    "make_kernel",
    "n_angles",
    "parallel",
    "pdude",
    "pointwise",
    "random_candidates",
    "run_lse",
    "score_level_set",
    "select_batch",
    "straddle",
    "stratified_doe",
]
