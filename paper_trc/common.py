"""Run bookkeeping: every result directory gets a run.json that says exactly what made it.

    results/<experiment>/<tag>/store.jsonl   the evaluations (mvlse EvaluationStore)
    results/<experiment>/<tag>/run.json      commits, config, timing, stop reason, outcome

A run refuses to start on uncommitted code, because a result whose code cannot be
recovered is a result that cannot be cited. ``--allow-dirty`` overrides it for a smoke
test; the manifest then records ``"dirty": true`` and status.py flags the run.
"""

import argparse
import dataclasses
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

import config

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
REPOS = {
    "mvlse": HERE.parent,
    "blueskycdarr": Path.home() / "Projects" / "BlueSkyCDaRR",
}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, check=True).stdout.strip()


def code_state() -> dict:
    """Commit and dirty flag of each repo the result depends on.

    Only tracked source files count as dirty. Untracked notebooks and images do not
    change what a run computes.
    """
    state = {}
    for name, repo in REPOS.items():
        dirty = _git(repo, "status", "--porcelain", "--untracked-files=no")
        state[name] = {"commit": _git(repo, "rev-parse", "HEAD"), "dirty": bool(dirty)}
    return state


def parser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--allow-dirty", action="store_true",
                    help="run on uncommitted code (smoke tests only; flagged in STATUS.md)")
    return ap


def check_clean(allow_dirty: bool) -> dict:
    state = code_state()
    dirty = [name for name, s in state.items() if s["dirty"]]
    if dirty and not allow_dirty:
        sys.exit(f"refusing to run: uncommitted changes in {', '.join(dirty)}. "
                 "Commit first, or pass --allow-dirty for a smoke test.")
    return state


def run_dir(experiment: str, tag: str) -> Path:
    path = RESULTS / experiment / tag
    path.mkdir(parents=True, exist_ok=True)
    return path


def _plain(value):
    """JSON-safe copy of config values (Continuous/Ordinal, numpy, tuples)."""
    if dataclasses.is_dataclass(value):
        return {"type": type(value).__name__, **_plain(dataclasses.asdict(value))}
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def start_manifest(experiment: str, tag: str, state: dict, params: dict, argv: list[str]) -> dict:
    return {
        "experiment": experiment,
        "tag": tag,
        "status": "running",
        "started": dt.datetime.now().isoformat(timespec="seconds"),
        "finished": None,
        "argv": argv,
        "code": state,
        "params": _plain(params),
        "shared": _plain({
            "FIXED": config.FIXED, "N_ENCOUNTERS": config.N_ENCOUNTERS,
            "TAU": config.TAU, "SEED": config.SEED, "LSE": config.LSE,
            "RESOLVER": config.RESOLVER, "RECOVERY": config.RECOVERY,
        }),
        "outcome": None,
    }


def write_manifest(path: Path, manifest: dict) -> None:
    (path / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")


def lse_outcome(result) -> dict:
    """What a finished mvlse run reports: counts, stop reason, final straddle, correlations."""
    last = result.rounds[-1] if result.rounds else None
    return {
        "n_evaluations": result.n_evaluations,
        "n_initial": result.n_initial,
        "stop_reason": result.stop_reason,
        "seconds": round(result.seconds, 1),
        "final_max_straddle": None if last is None else last.max_straddle,
        "final_ambiguous_fraction": None if last is None else last.ambiguous_fraction,
        "converged": bool(last is not None and last.max_straddle < 0),
        "rounds": [_plain(dataclasses.asdict(r)) for r in result.rounds],
        "level_correlations": _plain(result.level_correlations()),
    }


def run_logged(experiment: str, tag: str, params: dict, allow_dirty: bool, body):
    """Run ``body(path) -> outcome dict`` with a manifest written before and after.

    A crash leaves ``status: "running"`` with no ``finished`` time, which status.py
    reports as interrupted.
    """
    state = check_clean(allow_dirty)
    path = run_dir(experiment, tag)
    manifest = start_manifest(experiment, tag, state, params, sys.argv)
    write_manifest(path, manifest)
    outcome = body(path)
    manifest.update(status="done", outcome=_plain(outcome),
                    finished=dt.datetime.now().isoformat(timespec="seconds"))
    write_manifest(path, manifest)
    return outcome


def show_p(y: float, n: int = config.N_ENCOUNTERS) -> str:
    """log10 estimate -> probability with its 95% Monte Carlo interval, the binomial
    standard error recomputed from (y, n) as blueskycdarr's log10_p_los does. A zero-loss
    cell reads as the zero it is."""
    if y <= np.log10(0.5 / (n + 1.0)) + 1e-9:
        return f"P(LoS) = 0 observed (0 losses in {n}; < {3.0 / n:.1e} at 95%)"
    p = 10 ** y
    se = np.sqrt((1.0 - p) / (n * p)) / np.log(10.0)
    return (f"P(LoS) = {10 ** y:.2e}  "
            f"95% CI [{10 ** (y - 1.96 * se):.1e}, {10 ** (y + 1.96 * se):.1e}]")
