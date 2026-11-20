"""Exp. 1 with the relative heading on a log scale -- same space, budget rule and seed as exp1.

Does a log-scaled heading let the campaign find the shallow-end rise below ~10 deg that the
linear-heading campaign smoothed away?

    python exp1_log_heading.py     # the campaign; resumes from its store
"""

from mvlse import run_lse

import common
import config
import sim

P = config.EXPERIMENTS["exp1_log_heading"]


def campaign_body(path):
    result = run_lse(
        config.space_of(P["space"]),
        sim.blackbox(),
        threshold=config.TAU,
        max_evaluations=P["max_evaluations"],
        n_init_per_category=P["n_init_per_category"],
        store=path / "store.jsonl",
        **config.LSE,
    )
    print(result.summary())
    return common.lse_outcome(result)


if __name__ == "__main__":
    args = common.parser(__doc__.splitlines()[0]).parse_args()
    common.run_logged("exp1_log_heading", "campaign", P, args.allow_dirty, campaign_body)
