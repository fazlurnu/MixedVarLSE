"""Command-line entry point, for running the built-in benchmarks without writing code.

    python -m mvlse list
    python -m mvlse run himmelblau --method hs_dim --budget 150 --out runs/himmel
    python -m mvlse compare goldstein_flip --seeds 1 2 3
    python -m mvlse compare himmelblau_flip --active --budget 120

Your own problem does not go through here — for that, declare a
:class:`~mvlse.space.DesignSpace` and call :func:`~mvlse.lse.run_lse` directly. See
``examples/``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmark import compare_active, compare_methods
from .correlation import CORRELATIONS
from .lse import LseConfig, run_lse
from .metrics import score_level_set
from .models import MODEL_NAMES
from .problems import get_problem, list_problems


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("problem", choices=list_problems())
    parser.add_argument("--method", choices=list(MODEL_NAMES), default="hs_dim")
    parser.add_argument("--correlation", choices=sorted(CORRELATIONS), default="squar_exp")
    parser.add_argument("--seed", type=int, default=7)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mvlse", description="mixed-variable level-set estimation"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list the built-in benchmark problems")

    run = sub.add_parser("run", help="run the active loop on one benchmark")
    _add_common(run)
    run.add_argument("--budget", type=int, default=150, help="max evaluations")
    run.add_argument("--init", type=int, default=2, help="initial points per category")
    run.add_argument("--batch", type=int, default=1, help="points acquired per round")
    run.add_argument("--out", type=Path, default=None, help="directory for artefacts")
    run.add_argument("--plots", action="store_true", help="write boundary and convergence plots")

    compare = sub.add_parser("compare", help="compare every method on one benchmark")
    compare.add_argument("problem", choices=list_problems())
    compare.add_argument("--correlation", choices=sorted(CORRELATIONS), default="squar_exp")
    compare.add_argument("--methods", nargs="+", choices=list(MODEL_NAMES),
                         default=list(MODEL_NAMES))
    compare.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    compare.add_argument("--per-category", type=int, default=2, help="static DoE size per category")
    compare.add_argument("--active", action="store_true", help="compare under the active loop")
    compare.add_argument("--budget", type=int, default=120, help="max evaluations, --active only")
    compare.add_argument("--out", type=Path, default=None, help="write the table as CSV here")

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.command == "list":
        for name in list_problems():
            problem = get_problem(name)
            print(
                f"{name:20s} q={problem.space.q:<3d} r={problem.space.r} "
                f"m={problem.space.m:<4d} tau={problem.threshold:+.4g}   {problem.reference}"
            )
        return

    problem = get_problem(args.problem)

    if args.command == "run":
        config = LseConfig(
            threshold=problem.threshold,
            greater_is_inside=problem.greater_is_inside,
            method=args.method,
            correlation=args.correlation,
            n_init_per_category=args.init,
            max_evaluations=args.budget,
            batch_size=args.batch,
            seed=args.seed,
            keep_snapshots=args.plots,
        )
        result = run_lse(problem.space, problem, config=config)

        from .benchmark import test_grid

        grid = test_grid(problem)
        mu, _ = result.predict(grid)
        score = score_level_set(
            mu, problem.truth(grid), problem.threshold,
            greater_is_inside=problem.greater_is_inside,
        )
        print(f"scored against the truth: {score}")

        if args.out is not None:
            args.out.mkdir(parents=True, exist_ok=True)
            result.to_dataframe().to_csv(args.out / "evaluations.csv", index=False)
            result.history().to_csv(args.out / "history.csv", index=False)
            (args.out / "summary.json").write_text(
                json.dumps(
                    {
                        "problem": problem.name,
                        "method": args.method,
                        "correlation": args.correlation,
                        "seed": args.seed,
                        "threshold": problem.threshold,
                        "n_evaluations": result.n_evaluations,
                        "stop_reason": result.stop_reason,
                        "seconds": result.seconds,
                        **score.as_dict(),
                    },
                    indent=2,
                )
            )
            if args.plots:
                from . import report

                report.boundary_panels(result, problem, path=args.out / "boundaries.png")
                report.convergence(result, problem, path=args.out / "convergence.png")
                if args.method != "category_wise":
                    report.level_correlation_heatmap(result, path=args.out / "levels.png")
            print(f"wrote {args.out}")
        return

    # compare
    if args.active:
        table = compare_active(
            problem, args.methods, seeds=args.seeds,
            max_evaluations=args.budget, correlation=args.correlation,
        )
    else:
        table = compare_methods(
            problem, args.methods, seeds=args.seeds,
            n_per_category=args.per_category, correlation=args.correlation,
        )
    print()
    print(table.to_string(index=False))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out, index=False)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
