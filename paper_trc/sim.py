"""The one place the simulator is built: the CDR system of config.py, plus per-run overrides."""

from blueskycdarr.blackbox import make_blackbox
from blueskycdarr.recovery import ProbabilisticFTR

import config

assert config.RECOVERY == "ProbabilisticFTR", "sim.py builds ProbabilisticFTR only"
assert config.RESOLVER == "MVP", "sim.py relies on make_blackbox's default resolver (MVP)"


def blackbox(n_encounters: int = config.N_ENCOUNTERS, **overrides):
    """``points -> [log10 P(LoS)]``. ``overrides`` replace or extend config.FIXED.

    The Monte Carlo standard error is dropped on purpose: at N = 1e5 the estimate at tau
    is good to about +-20%, so the GP takes each one as exact (paper Sec. 4.1).
    ``common.show_p`` recovers the interval from (y, N) when a result is reported.

    A point that carries ``speed``, or a run that pins it in ``overrides``, also gets its class
    airframe (``config.airframe``), so the 40 and 60 kt classes fly the fixed-wing, and every
    class CDaRR's turn limits.
    """
    mc = make_blackbox(
        n_encounters=n_encounters,
        seed=config.SEED,
        recovery=ProbabilisticFTR(),
        fixed={**config.FIXED, **overrides},
        n_jobs=config.N_JOBS,
    )

    def evaluate(points):
        # the speed class sets the airframe, whether the point varies it or the run pins it
        points = [{**p, "kinematics": config.airframe(p.get("speed", overrides.get("speed")))}
                  if "speed" in p or "speed" in overrides else p
                  for p in points]
        return [y for y, _se in mc(points)]

    return evaluate


def bracket(evaluate, points: list[dict]) -> list[dict]:
    """Evaluate corner points and say which side of tau each lands on."""
    rows = []
    for pt, y in zip(points, evaluate(points), strict=True):
        rows.append({"point": pt, "y": y, "acceptable": bool(y <= config.TAU)})
    return rows
