"""Chapter 8 campaign: the CNS requirement surface at P(LoS) = 1e-3, classic Monte Carlo.

The script form of ``ch08_cns_requirement.ipynb``, sized for a many-core machine:
1e5 encounters per design point (1000 episodes of 100 pairs, fanned out over every
core), so a point near the threshold expects ~100 losses of separation and carries a
standard error of ~0.04 decades.

    python ch08_campaign.py --probe        # 6-corner bracket check first (~minutes)
    python ch08_campaign.py                # the campaign; resumes from the store
    python ch08_campaign.py --budget 400   # raise the budget later: only new points run

The store (``ch08_p_los.jsonl``, written next to this script) is the campaign's whole
state — copy it back to a laptop and ``run_lse`` replays it there for free.
"""

import argparse
import time

import numpy as np

from blueskycdarr.blackbox import make_blackbox
from mvlse import Continuous, DesignSpace, Ordinal, run_lse

KT = 0.514444  # kt -> m/s
SPEEDS = [20 * KT, 40 * KT, 60 * KT]
N_ENCOUNTERS = 100_000  # default; --encounters overrides (1e4 is fine for the surrogate)
TAU = -3.0  # log10 of the P(LoS) = 1e-3 threshold

# log10 of the Jeffreys stand-in a zero-loss cell carries INSIDE the surrogate (the
# log scale has no zero); anything at or below it is a measured zero, reported as one.
ZERO_Y = np.log10(0.5 / (N_ENCOUNTERS + 1.0))


def show_p(y: float, se: float) -> str:
    """log10 estimate -> probabilities. A zero-loss cell reads as the zero it is."""
    if y <= ZERO_Y + 1e-9:
        return (f"P(LoS) = 0 observed (0 losses in {N_ENCOUNTERS}; "
                f"< {3.0 / N_ENCOUNTERS:.1e} at 95%)")
    return (f"P(LoS) = {10 ** y:.2e}  "
            f"95% CI [{10 ** (y - 1.96 * se):.1e}, {10 ** (y + 1.96 * se):.1e}]")


def airframe(speed: float) -> str:
    """The class airframe: 20 kt flies the M600, 40 and 60 kt the fixed-wing."""
    return "multirotor" if speed < 15.0 else "fixedwing"


def blackbox(points):
    return _mc([{**p, "kinematics": airframe(p["speed"])} for p in points])


space = DesignSpace({
    "p_reception": Continuous(0.20, 1.00),            # reception probability
    "max_range_m": Continuous(200.0, 3000.0),         # surveillance range
    "dpsi":        Continuous(0.0, 180.0),            # relative heading [deg]
    "speed":       Ordinal(SPEEDS),                   # 20 / 40 / 60 kt classes [m/s]
    "pos_ci95":    Ordinal([3.0, 10.0, 30.0, 92.6]),  # ADS-L accuracy classes [m]
    "vel_ci95":    Ordinal([1.0, 3.0, 10.0]),         # [m/s]
})


def probe() -> None:
    """The bracket check: best CNS stack at a benign crossing vs worst near head-on,
    per speed class. The box brackets the boundary iff each pair straddles TAU."""
    best = {"p_reception": 1.00, "max_range_m": 3000.0, "pos_ci95": 3.0, "vel_ci95": 1.0}
    worst = {"p_reception": 0.20, "max_range_m": 200.0, "pos_ci95": 92.6, "vel_ci95": 10.0}
    points = [{**cns, "speed": v, "dpsi": ang}
              for v in SPEEDS for cns, ang in ((best, 90.0), (worst, 170.0))]

    t0 = time.perf_counter()
    for pt, (y, se) in zip(points, blackbox(points)):
        side = "acceptable" if y <= TAU else "NOT acceptable"
        print(f"V={pt['speed'] / KT:2.0f}kt ({airframe(pt['speed']):10s}) "
              f"dpsi={pt['dpsi']:5.0f}  p_rx={pt['p_reception']:.2f}  "
              f"pos={pt['pos_ci95']:4.1f}  ->  {show_p(y, se)}   {side}", flush=True)
    print(f"[{(time.perf_counter() - t0) / 60:.1f} min]")


def campaign(budget: int) -> None:
    t0 = time.perf_counter()
    result = run_lse(
        space,
        blackbox,
        threshold=TAU,
        method="hs_dim",
        n_init_per_category=1,  # 36 initial points, one per discrete combination
        max_evaluations=budget,
        batch_size=1,
        store="ch08_p_los.jsonl",
        keep_snapshots=True,
        seed=7,
        verbose=True,
    )
    print(f"\n[{(time.perf_counter() - t0) / 3600:.1f} h]\n")
    print(result.summary())
    for name, matrix in result.level_correlations().items():
        print(f"\n{name}  level correlations:")
        print(np.round(matrix, 3))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--probe", action="store_true", help="run the 6-corner bracket check only")
    ap.add_argument("--budget", type=int, default=250, help="total evaluations (default 250)")
    ap.add_argument("--encounters", type=int, default=N_ENCOUNTERS,
                    help=f"MC encounters per design point (default {N_ENCOUNTERS})")
    args = ap.parse_args()

    N_ENCOUNTERS = args.encounters
    ZERO_Y = np.log10(0.5 / (N_ENCOUNTERS + 1.0))
    _mc = make_blackbox(
        n_encounters=N_ENCOUNTERS,
        seed=7,
        fixed={"rpz": 150.0, "dcpa": 0.0},  # dcpa 0: worst geometry every encounter
        n_jobs=-1,                          # episodes fan out over every core
    )
    probe() if args.probe else campaign(args.budget)
