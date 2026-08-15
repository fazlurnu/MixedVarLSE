# Choosing a method

Short answer: **leave it on `hs_dim`**. It wins every comparison in
[FINDINGS.md](FINDINGS.md), it scales, and it does not need every category in your DoE.

The rest of this page is for when the short answer is not enough.

---

## The decision, in order

**1. Do your discrete variables have an order?**

If every one is genuinely ordinal *and* you believe the response varies smoothly along that
order — a mesh refinement, a discretised tolerance, a number of engines — then `dc` is
cheap, correct by construction, and has $q + r$ hyperparameters. It is a reasonable default
for that case alone.

If any variable is nominal — a material, a supplier, an architecture — `dc` imposes an
order that does not exist. Use `hs_dim`.

**2. Could two levels move in opposite directions?**

Ask it concretely: is there a level where the response *rises* over the region where the
others *fall*? If so, `dc` and `gower` cannot represent it at any hyperparameter value —
they compute $\exp(-\theta d)$, which is strictly positive. They will not fail loudly; they
will quietly set $\theta$ to its bound and treat that level as unrelated to the rest,
throwing away its data.

If you cannot rule it out, use `hs_dim`. You do not need to know in advance which levels
are opposed; that is what the fit is for.

**3. How many categories, and how many points?**

$m = \prod_k b_k$ grows fast. At $r = 3$ discrete variables of $3$ levels each, $m = 27$:

| method | hyperparameters at $q=2$, $m=27$ |
|---|---|
| `dc`, `gower` | 5 |
| `hs_dim` | 11 |
| `category_wise` | 54 |
| `hs_full` | **353** |

`hs_full` is only worth it when $m$ is genuinely small — say $m \le 6$ — **and** you have
reason to think the discrete variables interact. Otherwise its extra expressiveness is
swamped by the cost of estimating it: on `goldstein_flip` it scores 0.828 against
`hs_dim`'s 0.991, and takes fourteen times as long.

**4. Will every category appear in your design?**

`hs_full` needs every one — the angles touching an absent category are unidentified.
`category_wise` cannot say anything about a category it never saw (it returns the pooled
prior and lists the category in `unfitted`). `dc`, `gower` and `hs_dim` all extrapolate to
unseen categories.

---

## When to use `category_wise`

As a **baseline you expect to beat**, which is what the paper uses it for. It is also the
honest choice when you have a lot of data per category and genuine reason to believe the
categories share nothing — different physics, not different settings.

Run it alongside your real choice. If it wins, your categories really are unrelated and the
mixed kernel is spending hyperparameters on nothing.

---

## When to use `gower` over `dc`

When your discrete variables are nominal and you want the cheapest thing that does not
invent an order. `gower` treats every mismatch identically, which is wrong in a different
way from `dc` — it cannot say two levels are *more* alike than another pair — but it is at
least not asserting a false ordering.

Both are locked out of negative correlations, so neither is a substitute for `hs_dim` on a
problem where levels may oppose one another.

---

## Choosing the continuous correlation function

Independent of the above — `correlation=` composes with every `method=`.

| `correlation` | when |
|---|---|
| `squar_exp` | the default. Assumes a very smooth response. The paper's choice throughout. |
| `matern52` | slightly rougher; a good hedge when you do not know the smoothness |
| `matern32` | rougher still — kinks, discontinuous derivatives, short correlation length |
| `abs_exp` | very rough (Ornstein–Uhlenbeck). Rarely the right call for a design surface. |
| `pow_exp` | fits $p$ per dimension. Doubles the continuous hyperparameters; use when you have the budget to estimate them. |

If your boundary has corners (`squares` is the built-in example of this), `squar_exp` will
over-smooth them. Try `matern32`.

---

## Sanity checks after any fit

**Look at the level correlations first.**

```python
result.level_correlations()          # or report.level_correlation_heatmap(result)
```

- For an **ordinal** variable, a good fit usually comes out monotone — neighbours more
  correlated than distant levels — *without having been told there is an order*. If it does
  not, either the order is not real or you do not have enough data yet.
- Entries pinned near $0$ for a whole variable mean the model has decided that variable's
  levels are unrelated. With `dc` or `gower` that is the retreat described in
  [FINDINGS.md](FINDINGS.md) §3, and it is a signal to try `hs_dim`.
- A **negative** entry is information no distance-based kernel could have produced.

**Then read `stop_reason`.**

`budget` next to a still-positive `max_straddle` means the run was cut short, not that it
converged. `LseResult.summary()` says so explicitly. Do not report such a run as a
converged level set.

**Then check the ambiguous fraction.** `result.rounds[-1].ambiguous_fraction` is the share
of the monitor set whose confidence interval still contains the threshold. If it is not
small, the boundary is not pinned down wherever it is large.

---

## Budget

The initial design is `n_init_per_category * m` points, and it must fit inside
`max_evaluations`. With $m = 27$ and 3 points each you have spent 81 evaluations before the
loop acquires anything.

If that is most of your budget, lower `n_init_per_category` to 1 or 2 and let the straddle
spend the rest — it allocates far better than a uniform design does, because it can see
where the boundary is.

If your blackbox is slow, raise `batch_size` so each round evaluates several points at once,
and set `min_batch_distance` (try `0.15` on normalised inputs) so a batch does not pile up
on one patch of boundary.

Always pass `store=` for anything slow. A rerun then costs only the new points.
