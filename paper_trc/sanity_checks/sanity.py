"""Shared helpers for the sanity-check notebooks.

Every notebook tests what the campaigns actually run. ``campaign_setup`` captures what
``sim.blackbox`` hands to the episodes for one point, so nothing here re-implements the
routing from ``config.py`` into blueskycdarr.
"""

import sys
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parent.parent
if str(PAPER) not in sys.path:
    sys.path.insert(0, str(PAPER))

import config  # noqa: E402
import sim  # noqa: E402
from blueskycdarr.detection import detect  # noqa: E402
from blueskycdarr.engine import PairwiseWorld  # noqa: E402
from blueskycdarr.state import counterpart  # noqa: E402

# Categorical slots 1-3 of the reference palette, one per speed class, in class order.
CLASS_COLORS = {"20kt": "#2a78d6", "40kt": "#eb6834", "60kt": "#1baf7a"}
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"


class _Captured(Exception):
    pass


def _capture(cfg, models, backend, root, n_jobs):
    raise _Captured(cfg, models, backend, n_jobs, root)


def campaign_setup(point: dict, n_encounters: int = config.N_ENCOUNTERS, **overrides):
    """``(Config, Models, MC, n_jobs, root)`` exactly as ``sim.blackbox`` would simulate ``point``.

    Captured at the hand-off to the episodes (``blueskycdarr.experiment._run_condition``),
    so the speed-class airframe, ``config.FIXED`` and the point's own values have all
    been routed by the campaign's own code path. ``root`` is the seed sequence every
    episode's streams hang off. Nothing is simulated.
    """
    with mock.patch("blueskycdarr.experiment._run_condition", _capture):
        try:
            sim.blackbox(n_encounters, **overrides)([point])
        except _Captured as captured:
            return captured.args
    raise RuntimeError("sim.blackbox never reached the episodes")


def point(speed_class: str, dpsi: float, cns: dict | None = None) -> dict:
    """A design point: one speed class, one relative heading, CNS_BEST unless given."""
    return {**(cns or config.CNS_BEST), "speed": config.SPEED_CLASSES[speed_class], "dpsi": dpsi}


def fly(cfg, models, *, seconds: float, pairs=(1, 1), script=None, stop_after_cpa=None):
    """Spawn the campaign encounter and integrate it with CDR off.

    ``script(t, world)`` may issue commands before each step. ``stop_after_cpa`` ends the
    flight that many seconds after every pair is past its closest approach (the
    episode's ``done_timeout`` rule, on truth). Returns a dict of per-step arrays: per
    aircraft ``lat``/``lon``/``trk``/``gs`` (ownship ``2k``, intruder ``2k + 1``), per pair
    ``dist``/``tcpa``/``dcpa``/``in_conflict`` from the ownship's side, on truth, in the
    package's own metric -- the one LoS is scored in.
    """
    scenario = replace(models.scenario, pairs=pairs)
    geometry = scenario.draw_geometry(np.random.default_rng(0))  # pinned slots draw nothing
    dt = cfg.simulation.dt
    rows = {k: [] for k in ("t", "lat", "lon", "trk", "gs", "dist", "tcpa", "dcpa", "in_conflict")}
    with PairwiseWorld(scenario, geometry, models.aircraft, cfg.conflict, cfg.simulation) as w:
        n = 2 * scenario.n_pairs
        perm = np.asarray(counterpart(np.arange(n)))
        past_since = None
        for step in range(int(round(seconds / dt)) + 1):
            t = step * dt
            truth = w.truth()
            c = detect(truth, truth.reindexed(perm), np.ones(n, bool),
                       cfg.conflict.rpz, cfg.conflict.t_lookahead)
            rows["t"].append(t)
            rows["lat"].append(truth.lat)
            rows["lon"].append(truth.lon)
            rows["trk"].append(truth.trk)
            rows["gs"].append(truth.gs)
            rows["dist"].append(w.pair_distances())
            rows["tcpa"].append(c.tcpa[0::2])
            rows["dcpa"].append(c.dcpa[0::2])
            rows["in_conflict"].append(c.in_conflict[0::2])
            if stop_after_cpa is not None:
                past = bool(np.all(c.tcpa < 0.0))
                past_since = (t if past_since is None else past_since) if past else None
                if past_since is not None and t - past_since >= stop_after_cpa:
                    break
            if script is not None:
                script(t, w)
            w.step()
    return {k: np.asarray(v) for k, v in rows.items()}


class Checks:
    """Collects PASS / FAIL / NOTE rows; ``table()`` is the notebook's verdict."""

    def __init__(self):
        self.rows = []

    def __call__(self, name: str, ok: bool, detail: str = "") -> bool:
        status = "PASS" if ok else "FAIL"
        self.rows.append({"check": name, "status": status, "detail": detail})
        print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))
        return ok

    def note(self, name: str, detail: str) -> None:
        self.rows.append({"check": name, "status": "NOTE", "detail": detail})
        print(f"[NOTE] {name} -- {detail}")

    def table(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def style(ax, *, xlabel: str, ylabel: str, title: str | None = None):
    """Recessive axes: light grid behind the data, ink-colored text, no top/right spines."""
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2)
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    if title:
        ax.set_title(title, color=INK, loc="left", fontsize=11)
