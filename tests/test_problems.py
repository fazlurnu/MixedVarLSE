"""The benchmark catalogue: every surface builds, and none of them is degenerate."""
import numpy as np
import pytest

from mvlse import grid_candidates, random_candidates, score_level_set
from mvlse.problems import get_problem, list_problems

FLIP_PROBLEMS = ["goldstein_flip", "branin_flip", "wiggly_flip", "squares_flip", "himmelblau_flip"]


def reference(problem, n=11):
    if problem.space.q <= 2:
        return grid_candidates(problem.space, n)
    return random_candidates(problem.space, 128, seed=0)


@pytest.mark.parametrize("name", list_problems())
def test_problem_builds_and_evaluates(name):
    problem = get_problem(name)
    assert problem.name == name
    assert problem.space.q >= 1
    assert problem.reference

    w = reference(problem)
    values = problem.truth(w)
    assert values.shape == (len(w),)
    assert np.all(np.isfinite(values))


@pytest.mark.parametrize("name", list_problems())
def test_no_category_is_empty_or_full(name):
    """A degenerate category measures nothing — every method scores the same on it."""
    problem = get_problem(name)
    w = reference(problem)
    inside = problem.inside(w)
    categories = problem.space.category_of(w)
    fractions = np.array([inside[categories == c].mean() for c in range(problem.space.m)])
    assert fractions.max() > 0.0, f"{name}: every category is entirely outside the set"
    assert fractions.min() < 1.0, f"{name}: every category is entirely inside the set"
    assert 0.02 < inside.mean() < 0.98, f"{name}: overall in-set fraction {inside.mean():.3f}"


@pytest.mark.parametrize("name", FLIP_PROBLEMS)
def test_flip_family_is_anti_correlated_as_designed(name):
    """corr(A, C) = +1 and corr(A, B) = corr(B, C) = -1 — the structure that separates
    the hypersphere kernels from the coding ones."""
    problem = get_problem(name)
    w = reference(problem)
    values = problem.truth(w)
    level = w[:, problem.space.q + problem.space.r - 1].astype(int)
    per_level = [values[level == k] for k in range(3)]

    assert np.corrcoef(per_level[0], per_level[2])[0, 1] == pytest.approx(1.0, abs=1e-6)
    assert np.corrcoef(per_level[0], per_level[1])[0, 1] == pytest.approx(-1.0, abs=1e-6)
    assert np.corrcoef(per_level[1], per_level[2])[0, 1] == pytest.approx(-1.0, abs=1e-6)


def test_graded_control_is_positively_correlated():
    """The control for the flip family: every method can at least reach the right sign."""
    problem = get_problem("goldstein_graded")
    w = reference(problem)
    values = problem.truth(w)
    level = w[:, problem.space.q + problem.space.r - 1].astype(int)
    per_level = [values[level == k] for k in range(3)]
    matrix = np.array([[np.corrcoef(a, b)[0, 1] for b in per_level] for a in per_level])
    assert matrix.min() > 0.0
    assert matrix[0, 1] < 0.9   # ... but graded, not all identical


def test_paper_test_cases_have_the_paper_shape():
    assert (get_problem("goldstein").space.q, get_problem("goldstein").space.m) == (2, 9)
    assert (get_problem("branin").space.q, get_problem("branin").space.m) == (2, 4)
    assert (get_problem("augmented_branin").space.q, get_problem("augmented_branin").space.m) == (10, 4)


def test_bsv_problems_reproduce_their_failure_region_at_the_base_category():
    """At z1 = z2 = 0 with tau = 0, the scale is 1 and the shift is 0, so the surface is
    the Bayesian-safety-validation paper's own function."""
    from mvlse.problems.functions import booth_h, himmelblau_h, squares_h

    for name, raw in [("booth", booth_h), ("squares", squares_h), ("himmelblau", himmelblau_h)]:
        problem = get_problem(name)
        assert problem.threshold == 0.0
        w = grid_candidates(problem.space, 7)
        base = w[(w[:, 2] == 0) & (w[:, 3] == 0)]
        points = problem.space.decode(base)
        expected = raw(np.array([p["x1"] for p in points]), np.array([p["x2"] for p in points]))
        assert np.allclose(problem.truth(base), expected)


def test_hs_dim_beats_the_coding_kernels_on_an_anti_correlated_surface():
    """The paper's central claim, as a regression test. Himmelblau-flip is where the gap
    is widest: a hard continuous surface forces the fit to borrow across levels, and only
    the hypersphere kernels can borrow from an anti-correlated one."""
    from mvlse import fit_surrogate, stratified_doe

    problem = get_problem("himmelblau_flip")
    grid = grid_candidates(problem.space, 11)
    truth = problem.truth(grid)

    scores = {}
    for method in ["dc", "gower", "hs_dim"]:
        f1 = []
        for seed in (1, 2, 3):
            w = stratified_doe(problem.space, 2, seed=seed)
            model = fit_surrogate(method, problem.space, w, problem.truth(w),
                                  rng=np.random.default_rng(seed), n_starts=4)
            mu, _ = model.predict(grid)
            f1.append(score_level_set(mu, truth, problem.threshold).f1)
        scores[method] = float(np.mean(f1))

    assert scores["hs_dim"] > scores["dc"] + 0.05
    assert scores["hs_dim"] > scores["gower"] + 0.05


def test_unknown_problem_is_rejected():
    with pytest.raises(ValueError, match="unknown problem"):
        get_problem("not_a_problem")
