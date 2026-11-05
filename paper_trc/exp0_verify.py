"""Exp. 0 -- Stage 1 verification on surfaces whose level set is known (paper Sec. 5.4).

Same active loop, kernel and stopping as the CNS campaigns, so the verification
exercises the method being verified. Writes one CSV of F1/accuracy per problem.

    python exp0_verify.py
"""

import time

from mvlse.benchmark import compare_active
from mvlse.problems import get_problem

import common
import config

P = config.EXPERIMENTS["exp0_verify"]


def body(path):
    outcome = {}
    for name in P["problems"]:
        t0 = time.perf_counter()
        frame = compare_active(
            get_problem(name),
            methods=P["methods"],
            seeds=P["seeds"],
            max_evaluations=P["max_evaluations"],
            n_init_per_category=P["n_init_per_category"],
            flip_tolerance=config.LSE["flip_tolerance"],
        )
        frame.to_csv(path / f"{name}.csv", index=False)
        outcome[name] = {"csv": f"{name}.csv", "seconds": round(time.perf_counter() - t0, 1)}
    return outcome


if __name__ == "__main__":
    args = common.parser(__doc__.splitlines()[0]).parse_args()
    common.run_logged("exp0_verify", "active", P, args.allow_dirty, body)
