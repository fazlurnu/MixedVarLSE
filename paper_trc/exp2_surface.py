"""Exp. 2 -- the requirement surface: four CNS variables, 20 kt, at 2 and 180 deg (Sec. 5.8-5.10).

    python exp2_surface.py --dpsi 180 --bracket   # corner check at one heading
    python exp2_surface.py --dpsi 180             # one campaign; resumes from its store
    python exp2_surface.py --dpsi 2
"""

from mvlse import run_lse

import common
import config
import sim

P = config.EXPERIMENTS["exp2_surface"]


def make_bodies(dpsi: float):
    evaluate = sim.blackbox(speed=P["speed"], dpsi=dpsi)

    def bracket_body(path):
        rows = sim.bracket(evaluate, [config.CNS_BEST, config.CNS_WORST])
        for r in rows:
            print(f"dpsi={dpsi:5.0f}  {r['point']}  ->  {common.show_p(r['y'])}  "
                  f"{'acceptable' if r['acceptable'] else 'NOT acceptable'}", flush=True)
        return {"corners": rows}

    def campaign_body(path):
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

    return bracket_body, campaign_body


if __name__ == "__main__":
    ap = common.parser(__doc__.splitlines()[0])
    ap.add_argument("--dpsi", type=float, required=True, choices=P["headings"])
    ap.add_argument("--bracket", action="store_true", help="corner check only")
    args = ap.parse_args()

    bracket_body, campaign_body = make_bodies(args.dpsi)
    params = {**P, "dpsi": args.dpsi}
    tag = f"dpsi{args.dpsi:.0f}_" + ("bracket" if args.bracket else "campaign")
    common.run_logged("exp2_surface", tag, params, args.allow_dirty,
                      bracket_body if args.bracket else campaign_body)
