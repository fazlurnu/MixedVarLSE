"""Exp. 4 -- Stage 2 confirmation of the fitted exp2 boundary (paper Sec. 5.12). SKELETON.

(a) Direct Monte Carlo at points ON the predicted boundary, at 10x the surrogate's
    encounter budget: is the surrogate accurate where it decides the requirement?
(b) The same check on a second simulation engine: is the requirement a property of
    the CDR logic or of one implementation of it?

Neither check is verification; there is no truth here. Sec. 5.4 (exp0) is verification.

Not implemented until exp2 has finished and the open choices in config.py
(n_boundary_points, second_engine) are made.
"""

import config

P = config.EXPERIMENTS["exp4_confirm"]

if __name__ == "__main__":
    raise SystemExit(
        "exp4 is a skeleton. To do:\n"
        "  1. replay results/exp2_surface/dpsi*_campaign/store.jsonl with run_lse "
        "(same config, budget = stored count) to recover the model;\n"
        f"  2. pick {P['n_boundary_points']} candidates with |mu - tau| smallest, spread "
        "over the categories;\n"
        f"  3. evaluate them with sim.blackbox(n_encounters={P['n_encounters']}, ...);\n"
        "  4. report predicted vs measured log10 P(LoS) with the MC interval;\n"
        f"  5. (b) second engine: {P['second_engine']!r} -- decide which.")
