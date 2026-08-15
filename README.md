# mvlse — mixed-variable level-set estimation

You have a function you cannot differentiate, cannot see inside, and can only afford to
call a few dozen times. Some of its inputs are numbers and some are choices. You want to
know **where it crosses a threshold** — the failure boundary, the safe envelope, the
feasible region — not where it is smallest.

```python
from mvlse import Continuous, DesignSpace, Nominal, Ordinal, run_lse

space = DesignSpace({
    "range_m":  Continuous(100.0, 5000.0),
    "tx_power": Continuous(0.0, 20.0),
    "antenna":  Nominal(["patch", "dipole", "helix"]),
    "coding":   Ordinal(["none", "conv_1_2", "turbo"]),
})

def my_simulator(points):                       # a batch of points in, one output each out
    return [run(p["range_m"], p["tx_power"], p["antenna"], p["coding"]) for p in points]

result = run_lse(space, my_simulator, threshold=0.0)

print(result.summary())
result.to_dataframe()                           # every evaluation, in your names and units
result.inside([{"range_m": 4200.0, ...}])       # is this point in the set?
```

Your blackbox sees **your own variable names in physical units** — never a normalised
number or a level index. Encoding is the library's job, and it is visible
(`space.describe()`), reversible (`encode` / `decode` round-trip), and optional
(`normalize=False`).

---

## Install

```bash
pip install -e ".[plots,tables]"
```

Requires Python 3.10+. The core needs only **numpy** and **scipy**; matplotlib, pandas,
Pillow and joblib are optional extras.

---

## What it does

Active level-set estimation: fit a mixed-variable Gaussian process to everything measured
so far, score every candidate by how likely it is to sit on the boundary, spend the next
evaluation there, repeat.

The acquisition is the **straddle**, $a = \kappa s - \lvert \mu - \tau \rvert$, which is
large where the model is uncertain *and* near the threshold. Its sign carries a stopping
rule that needs no ground truth: $a \lt 0$ everywhere means every candidate is confidently
classified. That matters, because on a real problem the truth is exactly what you do not
have.

The Gaussian process follows Pelamatti, Brevault, Balesdent, Talbi & Guerin (2020),
*Overview and Comparison of Gaussian Process-Based Surrogate Models for Mixed Continuous
and Discrete Variables* (`references/GP2.pdf`), keeping its notation and equation numbers
so the code and the paper read side by side. All five of its methods are implemented and
interchangeable:

| `method=` | paper | how discrete variables enter | hyperparameters |
|---|---|---|---|
| `category_wise` | §3.2 | one independent GP per category — the reference | $2qm$ |
| `dc` | Eq. (15) | each level becomes a number, treated as continuous | $2(q+r)$ |
| `gower` | Eq. (18) | a 0/1 mismatch score as the discrete distance | $2(q+r)$ |
| `hs_full` | Eq. (19) | one free correlation per pair of categories | $2q + m(m-1)/2$ |
| **`hs_dim`** | Eq. (23) | one small matrix per discrete variable — **the default** | $2q + \sum_k b_k(b_k-1)/2$ |

The continuous correlation function (Eq. 8) is chosen separately and composes with all
five: `squar_exp` (default), `abs_exp`, `pow_exp`, `matern32`, `matern52`.

**Why the default is `hs_dim`.** `dc` and `gower` both compute $\exp(-\theta d)$, which is
strictly positive, so neither can represent two levels that move in *opposite* directions.
The hypersphere kernels can, because a correlation built from a cosine goes negative. On a
surface with anti-correlated levels that is worth up to **19 points of F1** at equal budget
— see [docs/FINDINGS.md](docs/FINDINGS.md).

---

## Noisy blackboxes

Return `(value, standard_error)` instead of a bare number and the standard errors go onto
the covariance diagonal as fixed known noise. Nothing else changes — it is detected from
what you return, and predictions are then of the **latent, noise-free** response, which is
what a level set is about.

```python
def noisy_simulator(points):
    return [(estimate(p), standard_error(p)) for p in points]
```

Deterministic, `(y, se)` pairs, `Observation` objects, an `(n, 2)` array and dicts with a
`"y"` key are all accepted.

---

## Repository layout

```
mvlse/
  space.py         DesignSpace, Continuous / Ordinal / Nominal, encode & decode
  blackbox.py      the blackbox seam; whatever you return becomes (y, se)
  correlation.py   continuous correlation functions — paper Eq. (8)
  hypersphere.py   the PDUDE construction — paper Eqs. (20)-(21)
  kernels.py       the four mixed kernels — paper Eqs. (15), (18), (19), (23)
  gp.py            ordinary Kriging — paper Eqs. (9)-(14), plus known per-point noise
  models.py        the four kernels + category-wise Kriging behind one interface
  acquisition.py   the straddle, and batch selection
  doe.py           initial designs and candidate sets
  lse.py           the active loop, stopping rules, LseResult
  metrics.py       accuracy / F1 / RMSE against a known truth (benchmarks only)
  store.py         a resumable JSONL evaluation cache
  benchmark.py     comparing methods on a problem whose answer you know
  report.py        boundary panels, convergence traces, level-correlation heatmaps
  cli.py           python -m mvlse
  problems/        the benchmark surfaces
docs/              FORMULA_SHEET.md, CHOOSING.md, FINDINGS.md
examples/          custom_blackbox.ipynb — how to run it on your own function
tests/             148 tests
references/        GP2.pdf, bayes-safety-val.pdf
```

---

## Benchmark problems

```python
from mvlse.problems import get_problem, list_problems
problem = get_problem("himmelblau_flip")
```

**From the paper** — `goldstein` (Eqs. 36–37, Table 4), `branin` (Eqs. 28–29),
`augmented_branin` (Eqs. 33–35, ten continuous dimensions).

**Ordinal scale/shift** — `branin_mix`, `wiggly`, and the three toy failure regions of
Moss, Kochenderfer, Gariel & Dubois, *Bayesian Safety Validation* (§IV.A): `booth`,
`squares`, `himmelblau`. At the base category with $\tau = 0$ each reproduces that paper's
region exactly.

**Sign-flipped categorical** — `goldstein_flip`, `branin_flip`, `wiggly_flip`,
`squares_flip`, `himmelblau_flip` add a third *nominal* variable using the paper's own
Eq. (29) construction, making two levels the same surface and the third its mirror image.
`goldstein_graded` is the control, with graded positive correlations instead.

Every problem is declared exactly the way you would declare your own — named variables,
physical units, a plain blackbox. `mvlse/problems/catalog.py` doubles as worked examples.

---

## Command line

```bash
python -m mvlse list
python -m mvlse run himmelblau --method hs_dim --budget 150 --out runs/himmel --plots
python -m mvlse compare goldstein_flip --seeds 1 2 3
python -m mvlse compare himmelblau_flip --active --budget 120
```

The CLI drives the built-in benchmarks. For your own problem, declare a `DesignSpace` and
call `run_lse` — see [examples/custom_blackbox.ipynb](examples/custom_blackbox.ipynb).

---

## Documentation

- **[docs/FORMULA_SHEET.md](docs/FORMULA_SHEET.md)** — every equation of `GP2.pdf` beside
  the line of code that implements it. Read this next to the paper.
- **[docs/CHOOSING.md](docs/CHOOSING.md)** — which `method` to use, and the checks worth
  running after any fit.
- **[docs/FINDINGS.md](docs/FINDINGS.md)** — what the five methods actually do, measured,
  with the caveats.
- **[examples/custom_blackbox.ipynb](examples/custom_blackbox.ipynb)** — the whole API on a
  user-defined function, including noisy blackboxes, caching and parallel evaluation.

---

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

---

## References

Pelamatti, J., Brevault, L., Balesdent, M., Talbi, E.-G. & Guerin, Y. (2020). *Overview and
Comparison of Gaussian Process-Based Surrogate Models for Mixed Continuous and Discrete
Variables: Application on Aerospace Design Problems.* In *High-Performance Simulation-Based
Optimization*, Studies in Computational Intelligence 833, 189–224. — `references/GP2.pdf`

Moss, R. J., Kochenderfer, M. J., Gariel, M. & Dubois, A. *Bayesian Safety Validation for
Failure Probability Estimation of Black-Box Systems.* — `references/bayes-safety-val.pdf`

Bryan, B., Schneider, J., Nichol, R., Miller, C., Genovese, C. & Wasserman, L. (2005).
*Active Learning For Identifying Function Threshold Boundaries.* NeurIPS. — the straddle.
