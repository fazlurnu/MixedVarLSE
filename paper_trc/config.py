"""The ONLY place the TRC paper's experiment parameters live.

Every script reads from here, every run copies the values it used into its run.json,
and make_tables.py writes the paper's parameter tables from here. Change a value here,
commit, rerun. Never edit a script that already has results.

Units: SI throughout (m, m/s, s, deg). Knots appear only to name the speed class.
"""

import os
from dataclasses import replace

# One BLAS/OpenMP thread per process: the campaign's parallelism is its N_JOBS workers. Set
# before numpy loads; the workers inherit it. VECLIB is macOS, OPENBLAS/MKL are Linux numpy.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

from blueskycdarr.aircraft import FIXEDWING, MULTIROTOR, AircraftModel
from mvlse import Continuous, DesignSpace, Ordinal

KT = 0.514444  # kt -> m/s

# ----------------------------------------------------------------- the CDR system
# Fixed for every experiment. The requirement is conditional on this selection
# (paper Sections 1 and 2.1).
RESOLVER = "MVP"
RECOVERY = "ProbabilisticFTR"
RESOLUTION_MARGIN = 1.05   # MVP aims at margin x RPZ. At 1.00 the resolved CPA sits on the
                           # RPZ and dt / CDR cadence decide LoS (sanity_checks, 2026-11-04);
                           # 1.05 is CDaRR's asas_marh

# ------------------------------------------------------------ fixed parameters
FIXED = {
    "rpz": 150.0,           # protected-zone radius [m], near BUBBLES' 145 m lightest-class threshold
    "dcpa": 0.0,            # projected miss distance [m]: worst geometry every encounter
    "aim": "flown",         # dcpa holds for the FLOWN track; BlueSky's creconfs misses by up
                            # to ~30 m at 60 kt crossings (sanity_checks/01)
    "tlos": 400.0,          # spawn time to LoS [s]: head-on pairs start beyond the 15 km range
    "t_lookahead": 300.0,   # [s]
    "t_max": 900.0,         # episode cap [s]: room for the tlos spawn + the slow 2 deg endgame
    "resolution_margin": RESOLUTION_MARGIN,
}

# ---------------------------------------------------------------- Monte Carlo
N_ENCOUNTERS = 100_000     # per sample: ~100 LoS expected at tau, 95% CI [0.81, 1.22]e-3;
                           # tight enough that the GP takes each estimate as exact (Sec. 4.1)
TAU = -3.0                 # log10 of the P(LoS) = 1e-3 threshold
SEED = 7
# Worker processes per sample. Set per machine with TRC_N_JOBS (e.g. 100 on the server);
# the default 6 suits the 16 GB laptop (n_jobs=-1 ran it out of memory at tlos=400).
# Results do not depend on it: every episode's streams hang off its index alone.
N_JOBS = int(os.environ.get("TRC_N_JOBS", "6"))

# --------------------------------------------------------------- the estimator
LSE = {
    "method": "hs_dim",
    "correlation": "squar_exp",
    "batch_size": 1,
    "seed": SEED,
    # Fixed budget, no early stopping of any kind: every run uses all its samples.
    # A negative tolerance disables the plateau stop; stop_when_confident=False the
    # max-straddle < 0 stop (Sec. 4.3).
    "flip_tolerance": -1.0,
    "stop_when_confident": False,
    "keep_snapshots": False,
    "verbose": True,
}

# ------------------------------------------------------------- aircraft classes
SPEED_CLASSES = {          # [m/s]; 20 kt flies the M600 multirotor, 40/60 kt fixed-wing
    "20kt": 20 * KT,
    "40kt": 40 * KT,
    "60kt": 60 * KT,
}


# CDaRR's turn limits (fazlurnu/CDaRR, sim_configs/sim_config.json "bluesky" block), for
# every class. blueskycdarr's catalog sets them on the multirotor only; without them the
# fixed-wing turns at g tan(44 deg) / V from the first step, with no roll-in.
MAX_TURN_RATE = 15.0    # [deg/s]
MAX_TURN_ACCEL = 10.0   # [deg/s^2], rate of change of the turn rate


def airframe(speed: float) -> AircraftModel:
    """The class airframe, with CDaRR's turn limits."""
    base = MULTIROTOR if speed < 15.0 else FIXEDWING
    return replace(base, max_turn_rate=MAX_TURN_RATE, max_turn_accel=MAX_TURN_ACCEL)


# ----------------------------------------------------------------- CNS box
# One box for every experiment, so the surfaces are comparable.
CNS_VARIABLES = {
    "p_reception": Continuous(0.10, 1.00),             # reception probability [-]
    "max_range_m": Continuous(300.0, 15000.0),         # surveillance range [m]
    "pos_ci95":    Ordinal([3.0, 10.0, 30.0, 92.6]),   # ADS-L position accuracy, 95% [m]
    "vel_ci95":    Ordinal([1.0, 3.0, 10.0]),          # ADS-L velocity accuracy, 95% [m/s]
}
CNS_BEST = {"p_reception": 1.00, "max_range_m": 15000.0, "pos_ci95": 3.0, "vel_ci95": 1.0}
CNS_WORST = {"p_reception": 0.10, "max_range_m": 300.0, "pos_ci95": 92.6, "vel_ci95": 10.0}

HEADING_RANGE = (2.0, 180.0)   # relative heading [deg]; 0 deg is degenerate (no closure)

# -------------------------------------------------------------- experiments
EXPERIMENTS = {
    # Stage 1 verification: known-truth surfaces, the same active loop as the CNS runs.
    "exp0_verify": {
        # Same shape as exp1: 3 continuous, discrete levels (4, 3, 3) = 36 categories.
        # Goldstein: one smooth boundary per category. Himmelblau: disjoint islands, steep walls.
        "problems": ["goldstein36_graded", "goldstein36_flip", "himmelblau36_graded"],
        "methods": ["category_wise", "dc", "gower", "hs_dim"],
        "seeds": list(range(1, 11)),        # 10 repetitions, as Pelamatti et al. (2020)
        "max_evaluations": 250,     # = exp1
        "n_init_per_category": 1,   # = exp1: 36 initial samples
    },
    # Where is the worst heading, and how do the regimes move with speed? All three classes.
    "exp1_heading": {
        "space": {
            **CNS_VARIABLES,
            "dpsi": Continuous(*HEADING_RANGE),
            "speed": Ordinal(list(SPEED_CLASSES.values())),
        },
        "max_evaluations": 750,
        "n_init_per_category": 1,   # 36 categories -> 36 initial points
        "bracket_headings": (90.0, 170.0),   # best CNS at a benign crossing, worst near head-on
    },
    # The requirement surface: four CNS variables at the two regime extremes, each at the speed
    # class where exp1 found it hardest -- shallow (2 deg) at 20 kt, head-on (180 deg) at 60 kt.
    "exp2_surface": {
        "space": CNS_VARIABLES,
        "headings": (2.0, 180.0),
        "speed_at": {2.0: SPEED_CLASSES["20kt"], 180.0: SPEED_CLASSES["60kt"]},
        # Is 20 kt really the hardest shallow case? exp1 said so, but its shallow end is the part
        # it fitted worst (sanity_checks/16, 18): the other two classes at 2 deg as well.
        "extra_runs": [(2.0, "40kt"), (2.0, "60kt")],
        "max_evaluations": 250,
        "n_init_per_category": 2,   # 12 categories -> 24 initial points
    },
    # Mechanism check: exp2 at head-on with the MVP aim point raised to 1.05 RPZ.
    "exp3_margin": {
        "space": CNS_VARIABLES,
        "speed": SPEED_CLASSES["20kt"],
        "heading": 180.0,
        "resolution_margin": 1.05,
        "max_evaluations": 250,
        "n_init_per_category": 2,
    },
    # Stage 2 confirmation: direct Monte Carlo on the fitted exp2 boundary.
    "exp4_confirm": {
        "n_boundary_points": 20,           # TODO(author): pick
        "n_encounters": 1_000_000,         # 10x the campaign's N at each check point
        "second_engine": None,             # TODO(author): which simulator? (cdarr store?)
    },
}


def space_of(variables: dict) -> DesignSpace:
    return DesignSpace(dict(variables))
