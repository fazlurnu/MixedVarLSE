"""Exp. 3 -- mechanism check: exp2 at head-on with the MVP aim point at 1.05 RPZ (Sec. 5.11).

Does a resolution margin restore a monotone position-accuracy ordering?

    python exp3_margin.py
"""

from mvlse import run_lse

import common
import config
import sim

P = config.EXPERIMENTS["exp3_margin"]


def body(path):
    evaluate = sim.blackbox(speed=P["speed"], dpsi=P["heading"],
                            resolution_margin=P["resolution_margin"])
    result = run_lse(
        config.space_of(P["space"]),
        evaluate,
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
    tag = f"dpsi{P['heading']:.0f}_margin{P['resolution_margin']:.2f}"
    common.run_logged("exp3_margin", tag, P, args.allow_dirty, body)
