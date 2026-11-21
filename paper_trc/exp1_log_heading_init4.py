"""exp1_log_heading with 4 initial points per category (144) instead of 1 -- same space, budget (500) and seed.

Does a denser initial design steady the boundary at 30 m, where exp1_log_heading has few samples?

    python exp1_log_heading_init4.py     # the campaign; resumes from its store
"""

from mvlse import run_lse

import common
import config
import sim

P = config.EXPERIMENTS["exp1_log_heading_init4"]


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
    common.run_logged("exp1_log_heading_init4", "campaign", P, args.allow_dirty, campaign_body)
