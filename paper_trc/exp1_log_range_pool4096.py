"""exp1_log_heading_pool4096 with the surveillance range on a log scale and the Monte Carlo standard error as GP noise.

Same kernel, pool (4096 per category), budget (500) and seed. Log range spreads the short-range cliff in P(LoS)
over the axis; the standard error keeps zero-loss samples from pinning the surface (sanity checks 71, 72).

    python exp1_log_range_pool4096.py     # the campaign; resumes from its store
"""

from mvlse import run_lse

import common
import config
import sim

P = config.EXPERIMENTS["exp1_log_range_pool4096"]


def campaign_body(path):
    result = run_lse(
        config.space_of(P["space"]),
        sim.blackbox(with_se=P["with_se"]),
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
    common.run_logged("exp1_log_range_pool4096", "campaign", P, args.allow_dirty, campaign_body)
