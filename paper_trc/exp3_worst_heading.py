"""Exp. 3 -- the worst heading as an environment variable.

exp1 aimed at the requirement: estimate the level set of g(x) = max over dpsi of log10 P(LoS),
x = the CNS variables and the speed class. Each round picks the design with the largest straddle
of g and simulates it at the heading where it could be worst (highest mu + kappa*s), on a grid of
21 headings from 2 to 180 deg.

    python exp3_worst_heading.py       # the campaign; resumes from its store
"""

from mvlse import run_lse

import common
import config
import sim

P = config.EXPERIMENTS["exp3_worst_heading"]


def campaign_body(path):
    result = run_lse(
        config.space_of(P["space"]),
        sim.blackbox(),
        threshold=config.TAU,
        max_evaluations=P["max_evaluations"],
        n_init_per_category=P["n_init_per_category"],
        acquisition=P["acquisition"],
        environment=P["environment"],
        n_environment=P["n_environment"],
        store=path / "store.jsonl",
        **config.LSE,
    )
    print(result.summary())
    out = common.lse_outcome(result)
    out["environment_grid_deg"] = [float(v) for v in
                                   config.HEADING_RANGE[0] + (config.HEADING_RANGE[1] - config.HEADING_RANGE[0])
                                   * result.environment_grid]
    return out


if __name__ == "__main__":
    ap = common.parser(__doc__.splitlines()[0])
    args = ap.parse_args()
    common.run_logged("exp3_worst_heading", "campaign", P, args.allow_dirty, campaign_body)
