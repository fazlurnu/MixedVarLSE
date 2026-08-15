"""Benchmark problems: the paper's analytical test cases and the safety-validation toys.

    from mvlse.problems import get_problem, list_problems

    list_problems()
    p = get_problem("himmelblau_flip")

See :mod:`.functions` for the bare mathematics and :mod:`.catalog` for how each surface is
turned into a mixed-variable level-set problem.
"""
from .catalog import PROBLEMS, Problem, get_problem, list_problems
from .functions import (
    augmented_branin_h,
    booth_h,
    branin_h,
    goldstein_h,
    himmelblau_h,
    squares_h,
    wiggly_h,
)

__all__ = [
    "PROBLEMS",
    "Problem",
    "augmented_branin_h",
    "booth_h",
    "branin_h",
    "get_problem",
    "goldstein_h",
    "himmelblau_h",
    "list_problems",
    "squares_h",
    "wiggly_h",
]
