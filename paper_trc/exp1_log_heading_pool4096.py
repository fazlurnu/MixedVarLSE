"""exp1_log_heading with a 4096-per-category candidate pool instead of 256 -- same space, kernel, budget (500) and seed.

The log heading resolves the shallow-end rise; the dense pool reaches the head-on corner, where the range limit is
set. The 36 initial points are those of exp1_log_heading (seed its store with them to skip re-simulating).

    python exp1_log_heading_pool4096.py     # the campaign; resumes from its store
"""

from mvlse import run_lse

import common
import config
import sim

P = config.EXPERIMENTS["exp1_log_heading_pool4096"]


def campaign_body(path):
    result = run_lse(
        config.space_of(P["space"]),
        sim.blackbox(),
        threshold=config.TAU,
        max_evaluations=P["max_evaluations"],
        n_init_per_category=P["n_init_per_category"],
        n_candidates_per_category=P["n_candidates_per_category"],
        store=path / "store.jsonl",
        **config.LSE,
    )
    print(result.summary())
    return common.lse_outcome(result)


if __name__ == "__main__":
    args = common.parser(__doc__.splitlines()[0]).parse_args()
    common.run_logged("exp1_log_heading_pool4096", "campaign", P, args.allow_dirty, campaign_body)
