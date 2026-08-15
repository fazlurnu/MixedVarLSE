# Formula sheet

Every equation of `references/GP2.pdf` beside the code that implements it.

Equation numbers **(1)–(37)** are the paper's — Pelamatti, Brevault, Balesdent, Talbi &
Guerin (2020), *Overview and Comparison of Gaussian Process-Based Surrogate Models for
Mixed Continuous and Discrete Variables*. Numbers **(A1)–(A6)** are results this repo
relies on that the paper states without proof or does not state at all.

> Math is written for GitHub-flavoured markdown (`$…$` inline, `$$…$$` display). It also
> renders in VS Code's preview and in JupyterLab.

---

## Notation (the paper's)

| written | meaning | in code |
|---|---|---|
| $\mathbf{x}^i$ | sample $i$ — the index is a **superscript** | a row of `w` |
| $x_k^i$ | dimension $k$ of sample $i$ — the dimension is a **subscript** | `w[i, k]` |
| $\mathbf{x} = (x_1 \ldots x_q)$ | the continuous inputs, $q$ of them | `w[:, :space.q]` |
| $\mathbf{z} = (z_1 \ldots z_r)$ | the discrete inputs; $z_k$ has $b_k$ levels | `w[:, space.q:]`, as level **indices** |
| $\mathbf{w} = \lbrace \mathbf{x}, \mathbf{z} \rbrace$ | one mixed input | one row of `DesignSpace.encode(...)` |
| $m = \prod_k b_k$ | number of **categories** | `space.m` |
| $\varepsilon(\mathbf{x},\omega)$ | the stochastic term — **not** $Z$, which is a discrete input | — |
| $\mathbf{R}$ | $n \times n$ **correlation** matrix, unit diagonal | `kernel(params, w, w)` |
| $\sigma^2 \mathbf{R}$ | the **covariance** — a scalar multiple, not a new object | `FittedGP.sigma2` |
| $\psi_{\mathcal{X}}$ | correlations between a new input and the training set | `psi` in `FittedGP.predict` |
| $\mathcal{T}_{c_i,c_j}$, $\mathbf{T}$ | correlation between two categories, and its matrix | `hypersphere.pdude` |
| **PDUDE** | Positive Definite with Unit Diagonal Elements | what `pdude` guarantees |
| $\boldsymbol{\theta}$, $\mathbf{p}$ | correlation hyperparameters; $p_k = 2$ unless you pick `pow_exp` | `params` |
| $\alpha_{k,s} \in (0,\pi)$ | a hypersphere angle | `hypersphere.ANGLE_BOUNDS` |
| $\tau$ | the threshold | `LseConfig.threshold` |

One letter, three roles: $R(\cdot,\cdot)$ is a function, $\mathbf{R}$ a matrix,
$\mathbf{R}_{i,j}$ an entry.

---

## 1. The problem — paper Section 2

$$
y = f(\mathbf{x}, \mathbf{z}),
\qquad
S = \lbrace  \mathbf{w} : f(\mathbf{w}) \le \tau  \rbrace
$$

The **boundary** $f = \tau$ is the answer, not the minimum. The paper calls discrete
variables *ordered* or *unordered*; this repo says `Ordinal` and `Nominal`.

→ `mvlse/space.py`, `mvlse/blackbox.py`

---

## 2. The Gaussian process — paper Section 3.1

**Practical form** (5), (6) — a constant trend is *ordinary Kriging*:

$$
Y(\mathbf{x},\omega) = \beta + \varepsilon(\mathbf{x},\omega)
$$

**Stationarity** (7):

$$
\operatorname{Cov}(\mathbf{x}^i,\mathbf{x}^j)
= \sigma^2  R\left( \lVert \mathbf{x}^i - \mathbf{x}^j \rVert \right)
$$

Here $\sigma^2$ is scale and $R$ is shape (equal to $1$ on the diagonal).

**Correlation function** (8), the $p$-exponential and the Matérn:

$$
R_{p\text{-}\exp} = \exp\left( -\sum_{k=1}^{q} \theta_k  \lvert x_k^i - x_k^j \rvert^{p_k} \right)
$$

$$
R_{\text{Matérn}} = \prod_{k=1}^{q} m_\nu\left( \theta_k  \lvert x_k^i - x_k^j \rvert \right)
$$

```python
# mvlse/correlation.py
def call(params, d):                       # d is (n_a, n_b, dims) of per-dim distances
    theta = 10.0 ** params
    return np.exp(-((d**p) * theta).sum(axis=2))
```

> **It needs a distance.** That is the line the discrete variables break.
> The $p$-exponential is PSD only for $0 \lt p \le 2$ — hence `P_BOUNDS = (0.5, 2.0)`.

**(A1) Separability is what makes the coding kernels free.** Both forms are a sum (or a
product) of one term per dimension, so a kernel that only changes *which distances go in*
needs no new correlation code. That is why `Correlation.from_distances` is the entry point,
and why Eqs. (15) and (18) are three lines each.

**Likelihood** (9), and the closed forms (10) that the paper states without proof:

$$
\beta = \frac{\mathbf{1}^\top \mathbf{R}^{-1} \mathbf{y}}{\mathbf{1}^\top \mathbf{R}^{-1} \mathbf{1}}
\qquad\qquad
\sigma^2 = \frac{(\mathbf{y} - \mathbf{1}\beta)^\top \mathbf{R}^{-1} (\mathbf{y} - \mathbf{1}\beta)}{n}
$$

```python
# mvlse/gp.py
beta   = (ones @ cho_solve(chol, y)) / (ones @ cho_solve(chol, ones))
sigma2 = max(float(resid @ cho_solve(chol, resid)) / n, 1e-12)
```

**(A2) Concentrated log-likelihood.** Substituting (10) back into (9) removes $\beta$ and
$\sigma^2$ from the search, leaving

$$
\mathcal{L}(\boldsymbol{\theta})
= \tfrac{1}{2} \left( \underbrace{n \log \sigma^2}_{\text{fit the data}}
+ \underbrace{\log \det \mathbf{R}}_{\text{stay humble}} \right)
$$

The second term is the density's normalising constant — Occam's razor as arithmetic.

**(A3) The determinant, safely.** From $\mathbf{R} = \mathbf{L}\mathbf{L}^\top$ we get
$\log \det \mathbf{R} = 2 \sum_i \log L_{ii}$, which never overflows.

```python
def _log_det(chol):
    return 2.0 * float(np.log(np.diag(chol[0])).sum())
```

**Prediction** (12), (13), (14), at a new input $\mathbf{x}^{\ast}$:

$$
\hat{Y}(\mathbf{x}^{\ast})
= \beta + \psi^\top \mathbf{R}^{-1} (\mathbf{y} - \mathbf{1}\beta)
\qquad\qquad
s^2(\mathbf{x}^{\ast})
= \sigma^2 \left( 1 - \psi^\top \mathbf{R}^{-1} \psi \right)
$$

```python
mu  = self.beta + psi @ self._alpha
var = self.sigma2 * (1.0 - explained)
```

> $s^2$ contains no $\mathbf{y}$: uncertainty depends on **where** you measured, not on what
> you found. That is an assumption, not a theorem — it follows from normality, which is
> chosen because normals are closed under conditioning (so (12) and (14) are algebra) and
> under marginalisation (so an $n \times n$ system suffices).

**(A4) Known per-point noise.** The paper's model is noise-free. For a stochastic blackbox
that reports a standard error, the variances go on the diagonal as *fixed* quantities:

$$
\mathbf{C} = \sigma^2 \mathbf{R} + \operatorname{diag}\left( se_i^2 \right)
$$

$\beta$ still profiles out by generalised least squares, but $\sigma^2$ no longer does — it
joins the search as one extra parameter. Predictions remain predictions of the **latent**
response, which is what the level set is about.

→ `mvlse/gp.py`

---

## 3. What makes a correlation function legal

**(A5)** For every finite set of inputs, $\mathbf{a}^\top \mathbf{R} \mathbf{a} \ge 0$,
because that quantity equals $\operatorname{Var}(\mathbf{a}^\top \mathbf{y}) / \sigma^2$.
Cholesky needs positive **definite**; (10), (12) and (14) all invert $\mathbf{R}$.

Two construction tools: $\mathbf{A} + \mathbf{B}$ is legal, and
$\mathbf{A} \circ \mathbf{B}$ (**element-wise**) is legal by the Schur product theorem. The
second is what licenses (19) and (23).

> `A * B` is a correlation matrix. `A @ B` is **not**.

---

## 4. Four ways to handle a discrete variable — paper Sections 3.2–3.5

Write $u$ for the correlation between levels $A$ and $B$, and $v$ for the correlation
between $A$ and $C$, at a shared continuous location. Each method reaches only a *subset* of
the $(u,v)$ square, and you can work out which subset with algebra, before seeing any data:

| method | § | `method=` | reachable set | hyperparameters |
|---|---|---|---|---|
| category-wise | 3.2 | `category_wise` | the single point $(0,0)$ | $2qm$ |
| direct conversion | 3.3 | `dc` | the curve $v = u^{2^p}$ | $2(q+r)$ |
| Gower | 3.4 | `gower` | the line $v = u$ | $2(q+r)$ |
| hypersphere | 3.5 | `hs_full` | the open square $(-1,1)^2$ | $2q + m(m-1)/2$ |
| dimension-wise HS | 3.5 | `hs_dim` | the open square, per variable | $2q + \sum_k b_k(b_k-1)/2$ |

Counts are the paper's, for a $p$-exponential with $p$ fitted. With `squar_exp` (the
default, $p$ fixed at $2$) halve the $q$ and $r$ terms — which is what
`MixedKernel.n_params` reports.

**Category-wise** (Section 3.2) — $m$ independent GPs, nothing shared. Not a kernel; it is
`mvlse/models.py::CategoryWiseGP`, and it is the reference the others are judged against.

**Direct conversion** (15):

$$
R^{\ast}_{p\text{-}\exp} = \exp\left(
-\sum_{k=1}^{q} \theta_k  \lvert x_k^i - x_k^j \rvert^{p_k}
-\sum_{k=1}^{r} \theta_{k+q}  \lvert \tilde{z}_k^i - \tilde{z}_k^j \rvert^{p_{k+q}}
\right)
$$

```python
# mvlse/kernels.py::_direct_conversion
d[:, :, :q] = np.abs(w1[:, None, :q] - w2[None, :, :q])
for k, pos in enumerate(positions):
    d[:, :, q + k] = np.abs(pos[a][:, None] - pos[b][None, :])
return corr.from_distances(params, d)
```

**(A6)** With three equally-spaced levels, $v = u^{2^p}$, so $v = u^4$ at $p = 2$. The
coding **spacing** cancels into $\theta_z$; the **order** does not. For $b$ levels,
$v = u^{(b-1)^p}$ — the assumption strengthens as levels are added.

**Gower** (16), (17), (18):

$$
d_{\text{gow}}
= \frac{\sum_k \lvert x_k^i - x_k^j \rvert / \Delta x_k}{q+r}
+ \frac{\sum_k s(z_k^i, z_k^j)}{q+r},
\qquad
s(z_k^i, z_k^j) = 0 \text{ if } z_k^i = z_k^j, \text{ else } 1
$$

```python
# mvlse/kernels.py::_gower
d[:, :, :q] = np.abs(w1[:, None, :q] - w2[None, :, :q]) / spans / total
d[:, :, q + k] = (a[:, None] != b[None, :]) / total
```

Every mismatch scores $1$, so $u = v$ exactly: Gower is **order-blind** by construction.

**Hypersphere** (19), and dimension-wise (23):

$$
R^{\ast}(\mathbf{w}^i,\mathbf{w}^j) = R(\mathbf{x}^i,\mathbf{x}^j)  \mathcal{T}_{c_i,c_j}
\qquad\qquad
R^{\ast}(\mathbf{w}^i,\mathbf{w}^j)
= R(\mathbf{x}^i,\mathbf{x}^j) \prod_{k=1}^{r} \mathcal{T}_{k, z_k^i z_k^j}
$$

```python
# mvlse/kernels.py::_hypersphere_dimensionwise
out = corr(params[:n_cont], w1[:, :q], w2[:, :q])
for k, (start, n_ang, b) in enumerate(offsets):
    t = pdude(params[start:start + n_ang], b)
    out = out * t[a[:, None], c[None, :]]
```

> **The structural fact.** $\exp(-\theta d) \gt 0$ **always**, so neither (15) nor (18) can
> express anti-correlation at any $\theta$. The paper says so twice — Section 3.4 notes that
> Gower handles "the simultaneous presence of correlation and anti-correlation trends
> described by the same discrete variable" poorly, and Section 4.4 explains a *good* Gower
> result by "the absence of anti-correlation trends". A locked-out method **retreats**:
> $\theta_z \to \infty$, so $u = v = 0$ and the levels become independent. It degrades; it
> does not break. See [FINDINGS.md](FINDINGS.md).

---

## 5. The hypersphere decomposition — paper Section 3.5

**(20)** $\mathbf{T} = \mathbf{L}\mathbf{L}^\top$, read as a **generator**. PSD is then
free, since

$$
\mathbf{a}^\top \mathbf{L} \mathbf{L}^\top \mathbf{a}
= \lVert \mathbf{L}^\top \mathbf{a} \rVert^2 \ge 0 .
$$

It becomes strict (full PD, as PDUDE requires) because $\alpha_{k,s} \in (0,\pi)$ keeps
$l_{k,k} = \prod_s \sin \alpha_{k,s} \gt 0$, so $\mathbf{L}$ is non-singular.

Unit diagonal holds if and only if every **row of $\mathbf{L}$ is a unit vector**, since
$\mathbf{T}_{kk} = \lVert \mathbf{l}_k \rVert^2$. And
$\mathcal{T}_{jk} = \mathbf{l}_j \cdot \mathbf{l}_k = \cos \gamma_{jk}$ — **this is where
negative correlation comes from.**

**(21)**, with $s$ running from $2$ to $k-1$:

$$
l_{1,1} = 1,
\qquad
l_{k,1} = \cos \alpha_{k,1},
$$

$$
l_{k,s} = \left( \prod_{t \lt s} \sin \alpha_{k,t} \right) \cos \alpha_{k,s},
\qquad
l_{k,k} = \prod_{t \lt k} \sin \alpha_{k,t}
$$

```python
# mvlse/hypersphere.py
sin_prod = np.concatenate([[1.0], np.cumprod(np.sin(a))])
for col in range(row):
    lower[row, col] = sin_prod[col] * np.cos(a[col])
lower[row, row] = sin_prod[row]
return lower @ lower.T
```

Length $1$ by collapsing $\cos^2 + \sin^2 = 1$ from the inside out.

**Counts.** $m(m-1)/2$ angles for (19); $\sum_k b_k(b_k-1)/2$ for (23). At $r = 3$, $b = 3$
that is **351 against 9**. In exchange, (23) assumes the discrete variables **do not
interact**, and — unlike (19) — it does **not** require every category to appear in the
training set.

---

## 6. Active learning: the straddle

$$
a(\mathbf{w}) = \kappa  s(\mathbf{w}) - \lvert \hat{Y}(\mathbf{w}) - \tau \rvert,
\qquad \kappa = z_{0.975} \approx 1.96
$$

```python
# mvlse/acquisition.py
return kappa * sd - np.abs(mu - tau)
```

The sign is the whole point:

$$
a(\mathbf{w}) \gt 0
\iff \hat{Y} - \kappa s  \lt  \tau  \lt  \hat{Y} + \kappa s
$$

which reads "I cannot tell which side of the boundary this point is on". So
$\max_{\mathbf{w}} a(\mathbf{w}) \lt 0$ means every candidate is confidently classified —
**a stopping rule that needs no ground truth**, which is the only kind you can use on a real
problem.

> Confident is not the same as correct. If accuracy plateaus while $\max a \gt 0$, believe
> $\max a$.

Bryan et al. (2005). Not from `GP2.pdf`.

---

## 7. Scoring — benchmarks only

Over a test grid, with $\hat{S} = \lbrace  \mathbf{w} : \hat{Y}(\mathbf{w}) \le \tau  \rbrace$:

$$
F_1 = \frac{2  TP}{2  TP + FP + FN}
$$

True negatives never appear, so "everything is outside" scores $F_1 = 0$ however large the
domain. **Report accuracy and $F_1$ together** — on a small level set, accuracy alone
rewards a model for being right about the empty space.

→ `mvlse/metrics.py`

---

## 8. The benchmark surfaces — paper Section 4

| problem | equations | shape |
|---|---|---|
| `branin` | (28)–(29) | 2 continuous, 2 binary discrete, $m = 4$ |
| `augmented_branin` | (33)–(35) | 10 continuous, 2 binary discrete, $m = 4$ |
| `goldstein` | (36)–(37), Table 4 | 2 continuous, 2 ternary ordinal, $m = 9$ |

Table 4 sets $x_3, x_4 \in \lbrace 20, 50, 80 \rbrace$ as a function of $z_1, z_2$. The Branin of
Eq. (29) is the paper's own anti-correlated case — the four categories are $h$, $0.4h + 1.1$,
$-0.75h + 5.2$ and $-0.5h - 2.1$, which it describes as showing "pair-wise anti-correlation
and a considerable relative offset".

Booth, Squares and Himmelblau are **not** from this paper — they are the toy failure regions
of Moss, Kochenderfer, Gariel & Dubois, *Bayesian Safety Validation*
(`references/bayes-safety-val.pdf`, Section IV.A), made mixed-variable here.

→ `mvlse/problems/`

---

## Not implemented

**Dummy coding** (Section 3.3, Table 2) — described by the paper but excluded from its own
results. `dc` covers the coding family.

**BTGP** (Section 3.6, Eqs. 25–27) — a tree partition with a local Kriging per leaf, fitted
by MCMC over partition trees. A different class of model from the other five, and the
paper's own results have it losing to category-wise Kriging on two of three analytical
cases.
