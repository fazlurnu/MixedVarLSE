"""Exp. 2 -- the requirement surface: four CNS variables, 2 deg at 20 kt and 180 deg at 60 kt (Sec. 5.8-5.10).

    python exp2_surface.py --dpsi 180 --bracket   # corner check at one heading
    python exp2_surface.py --dpsi 180             # one campaign; resumes from its store
    python exp2_surface.py --dpsi 2
    python exp2_surface.py --dpsi 2 --speed 40kt   # the extra speed classes at 2 deg
"""

from mvlse import run_lse

import common
import config
import sim

P = config.EXPERIMENTS["exp2_surface"]


def make_bodies(dpsi: float, speed: float):
    evaluate = sim.blackbox(speed=speed, dpsi=dpsi)

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
    ap.add_argument("--speed", choices=list(config.SPEED_CLASSES),
                    help="speed class other than the heading's own (config: speed_at); see extra_runs")
    ap.add_argument("--bracket", action="store_true", help="corner check only")
    args = ap.parse_args()

    speed = config.SPEED_CLASSES[args.speed] if args.speed else P["speed_at"][args.dpsi]
    own = speed == P["speed_at"][args.dpsi]
    if not own and (args.dpsi, args.speed) not in P["extra_runs"]:
        ap.error(f"{args.speed} at {args.dpsi:g} deg is not one of config's exp2 extra_runs")
    bracket_body, campaign_body = make_bodies(args.dpsi, speed)
    params = {**P, "dpsi": args.dpsi, "speed": speed}
    tag = (f"dpsi{args.dpsi:.0f}_" + ("" if own else f"{args.speed}_")
           + ("bracket" if args.bracket else "campaign"))
    common.run_logged("exp2_surface", tag, params, args.allow_dirty,
                      bracket_body if args.bracket else campaign_body)
