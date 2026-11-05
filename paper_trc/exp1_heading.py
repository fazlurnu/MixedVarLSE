"""Exp. 1 -- heading scan over all six variables, all three speed classes (paper Sec. 5.7).

Where is the worst relative heading, and how do the two regimes move with speed?
This is what justifies the two headings exp2 conditions on.

    python exp1_heading.py --bracket   # corner check first: does the box contain tau?
    python exp1_heading.py             # the campaign; resumes from its store
"""

from mvlse import run_lse

import common
import config
import sim

P = config.EXPERIMENTS["exp1_heading"]


def bracket_body(path):
    benign, near_head_on = P["bracket_headings"]
    points = [{**cns, "speed": v, "dpsi": ang}
              for v in config.SPEED_CLASSES.values()
              for cns, ang in ((config.CNS_BEST, benign), (config.CNS_WORST, near_head_on))]
    rows = sim.bracket(sim.blackbox(), points)
    for r in rows:
        print(f"V={r['point']['speed'] / config.KT:2.0f} kt  dpsi={r['point']['dpsi']:5.0f}  "
              f"{common.show_p(r['y'])}  "
              f"{'acceptable' if r['acceptable'] else 'NOT acceptable'}", flush=True)
    return {"corners": rows}


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
    ap = common.parser(__doc__.splitlines()[0])
    ap.add_argument("--bracket", action="store_true", help="corner check only")
    args = ap.parse_args()
    if args.bracket:
        common.run_logged("exp1_heading", "bracket", P, args.allow_dirty, bracket_body)
    else:
        common.run_logged("exp1_heading", "campaign", P, args.allow_dirty, campaign_body)
