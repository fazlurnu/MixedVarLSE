# Findings

What the five methods actually do, measured on the benchmarks in `mvlse.problems`.

Reproduce any table here with:

```bash
python -m mvlse compare himmelblau_flip --seeds 1 2 3
```

---

## 1. The headline: the ranking is a property of the methods, the *gap* is not

Six surfaces, identical discrete structure, 54 training points (2 per category over 27
categories), F1 as the mean of 3 DoE seeds. Scored against a $21\times21$ per-category
truth grid.

| test case | `category_wise` | `dc` | `gower` | `hs_dim` | spread over the mixed kernels |
|---|---|---|---|---|---|
| goldstein_graded | 0.658 | 0.981 | 0.977 | **0.990** | 0.012 |
| goldstein_flip | 0.678 | 0.983 | 0.976 | **0.993** | 0.017 |
| branin_flip | 0.464 | 0.916 | 0.910 | **0.956** | 0.045 |
| squares_flip | 0.377 | 0.828 | 0.899 | **0.924** | 0.096 |
| wiggly_flip | 0.556 | 0.814 | 0.805 | **0.919** | 0.114 |
| **himmelblau_flip** | 0.400 | 0.695 | 0.693 | **0.882** | **0.189** |

`hs_dim` leads every row, so the ranking is about the kernels and not about any one
surface. But read the last column: **all five `*_flip` cases have exactly the same discrete
structure** — identical $S$, identical $T$, correlations exactly $\pm1$ — and the cost of
getting it wrong moves by a factor of **ten**.

On Himmelblau, `hs_dim` reaches 0.882 while `dc` and `gower` sit at 0.695 and 0.693: nearly
**nineteen points of F1** from the same 54 evaluations.

> **Never quote the ranking without the gap.** The ranking belongs to the methods; the size
> of the gap belongs to the test problem.

---

## 2. Why the gap moves: how much the fit has to borrow

Goldstein is a smooth low-order polynomial. A GP fits each category from a handful of
points, so nothing has to be borrowed across categories, and being *unable* to borrow costs
almost nothing — hence a spread of 0.017.

Himmelblau has steep walls and four disjoint blobs. Squares has kinked corners a stationary
kernel cannot represent. Wiggly has a short correlation length. On all three the continuous
points alone are not enough, the model **must** borrow across levels, and a kernel that
cannot borrow from an anti-correlated level is left with a third of its data.

**The rule:** the more a fit has to borrow across categories — because the surface is hard,
the budget is small, or the levels are anti-correlated — the more the discrete correlation
function decides the answer. *Benchmark on a surface where borrowing is actually necessary,
or you will measure nothing.*

This is also why a small gap on an easy surface is not evidence that the structure does not
matter. It usually means the experiment was near-saturated.

---

## 3. `dc` and `gower` cannot represent anti-correlation — and still score 0.98

Both compute $\exp(-\theta d)$, which is strictly positive. On `goldstein_flip` the truth is
$\operatorname{corr}(A,B) = -1$, $\operatorname{corr}(A,C) = +1$. That point is not merely
far from what they can reach — it is in a **half-plane neither can enter at any $\theta$**.
`tests/test_kernels.py` asserts exactly this over 500 random parameter draws.

And yet `dc` scores 0.983. How?

**It refuses to fit the structure rather than fitting it badly.** The likelihood pushes
$\theta_C$ to its upper bound, making the between-level correlation ~$10^{-35}$. That is
not an optimiser failure — it is the correct call under a wrong constraint. `dc` is allowed
to say "positively related, by this much" or "unrelated". It is *not* allowed to say
"opposed". Faced with a level that moves against the others, the best available answer is
**unrelated**, so it takes it, splitting the problem into independent sub-problems.

So the wrong assumption is **escapable, not fatal**. `dc` pays a fixed price — it throws
away the cross-level information — but it is never actively misled.

**Cannot represent $\ne$ performs badly.** That distinction is worth keeping: it is the
difference between a model that degrades and a model that breaks.

One caveat on the retreat: it is only partial. `dc` gives up the cross-level *values* but
keeps one shared set of hyperparameters, one $\beta$ and one $\sigma^2$, all estimated from
**all** the data. The anti-correlated block still teaches it how rough the surface is after
it has decided that block's values are useless. Pooled hyperparameters are a large part of
what a mixed kernel buys, and they survive an assumption that is completely wrong.

---

## 4. `hs_full` loses to its own smaller sibling

The full hypersphere (Eq. 19) is strictly more expressive than the dimension-wise variant
(Eq. 23) — it can model interactions between discrete variables, which `hs_dim` assumes
away. It still loses, badly, and the reason is arithmetic:

| | hyperparameters at $q=2$, $r=3$, $b=3$ | F1 on `goldstein_flip` | fit time |
|---|---|---|---|
| `hs_dim` | **11** | **0.991** | 5.4 s |
| `hs_full` | **353** | 0.828 | 76.7 s |

353 hyperparameters against 54 training points. The paper reports the same trend (Section
4.3: the standard hypersphere "becomes slightly less performant than its dimension-wise
variant" as the discrete dimension grows, "due to the increase in the number of
hyperparameters"), and it worsens with $m$: $m(m-1)/2$ against $\sum_k b_k(b_k-1)/2$.

`hs_full` also requires **every category to appear in the training set** — the angles
touching an absent category are unidentified. `hs_dim` does not.

Use `hs_full` only when $m$ is genuinely small (say $\le 6$) and you have reason to think
the discrete variables interact.

---

## 5. `category_wise` is the floor, and it is a long way down

0.38–0.68 across the board, against 0.88–0.99 for `hs_dim`. Two points per GP model
nothing, and on the hard surfaces it cannot find a kinked square at all.

This is *why* mixed kernels exist. It is also the paper's reference method precisely
because it is the industry default.

It fails in the two ways the paper predicts: it needs a lot of data per category, and it
can say nothing about a category the DoE never visited. `CategoryWiseGP.unfitted` reports
which ones it gave up on rather than quietly returning the pooled prior.

---

## 6. The hypersphere kernels discover order they were never told about

On `goldstein_flip`, `hs_dim` recovers a monotone $T$ for the two **ordinal** variables —
neighbouring levels more correlated than distant ones — and for the **nominal** variable it
recovers the sign pattern $(-,-,+)$, matching the built-in flip, from two points per
category.

```python
result.level_correlations()      # or report.level_correlation_heatmap(result)
```

Nothing told it $x_3$ was ordered. Look at this before you trust anything else a fit says.

The learned magnitudes fall short of $\pm1$ (typically $\approx\pm0.7$) because the
constant trend and the single global $\sigma^2$ absorb part of the offset and the scale
difference. A unit-diagonal $T$ with one global variance cannot represent per-level
*variance* at all — the paper's own Branin has scales $1, 0.4, -0.75, -0.5$, and no model
in this family captures those magnitudes exactly. **The sign is what carries the
information, and the sign is right.**

---

## 7. The internal stopping signal is honest

`max straddle < 0` means no candidate's confidence interval still contains $\tau$. It needs
no ground truth, which is the only reason it is usable on a real problem.

On `himmelblau` it fires at 116 evaluations, at which point the true F1 is 0.996 and
accuracy is 100%. Raising the budget to 200 changes nothing — the run had genuinely
converged, not merely run out.

Where the surface is hard the signal stays positive and honest rather than firing early.
Read `stop_reason` every time: `budget` alongside a still-positive `max_straddle` means the
run was **cut short**, not that it finished. `LseResult.summary()` says so explicitly.

---

## Caveats

- Three DoE seeds. The seed-to-seed standard deviation (in the `f1_std` column of
  `compare_methods`) is often the same size as the gap between `dc` and `gower`. Do not
  read a two-way split of 0.005 as a result.
- In `branin_flip`, `squares_flip`, `himmelblau_flip` and `wiggly_flip` the **ordinal**
  variables vary by an all-positive scale and shift, so their level-surfaces are perfectly
  correlated and $T_1, T_2$ have nothing to learn. Only the nominal variable carries
  discrete structure there. The two Goldstein cases are the ones where all three discrete
  variables do.
- `gower` beats `dc` on `squares_flip` (0.899 against 0.828) — the only case where it does,
  and by a wide margin. Both are excluded from the negative half-plane, so the difference
  must come from their *continuous and ordinal* treatment on a kinked boundary, not from
  the categorical. Worth a follow-up; do not read it as Gower handling the flip better.
- All of this is at 54 training points. The gaps shrink as the budget grows — that is the
  saturation effect of Section 2, not a change in the ranking.
