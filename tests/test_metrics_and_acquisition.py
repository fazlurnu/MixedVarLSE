import numpy as np
import pytest

from mvlse import (
    CORRELATIONS,
    Continuous,
    DesignSpace,
    Ordinal,
    get_correlation,
    grid_candidates,
    random_candidates,
    score_level_set,
    select_batch,
    straddle,
    stratified_doe,
)


# ------------------------------------------------------------------------------ metrics
def test_perfect_and_inverted_predictions():
    truth = np.array([0.0, 1.0, 2.0, 3.0])
    assert score_level_set(truth, truth, 1.5).f1 == 1.0
    assert score_level_set(truth, truth, 1.5).accuracy == 1.0
    # exactly inverted: everything truly inside is predicted outside, and vice versa
    assert score_level_set(3.0 - truth, truth, 1.5).f1 == 0.0
    assert score_level_set(3.0 - truth, truth, 1.5).accuracy == 0.0


def test_predicting_nothing_inside_scores_zero_f1_not_high_accuracy():
    """True negatives never enter F1 — which is the point of reporting it."""
    truth = np.concatenate([np.zeros(2), np.full(98, 10.0)])   # a small level set
    prediction = np.full(100, 10.0)                            # "nothing is inside"
    score = score_level_set(prediction, truth, 1.0)
    assert score.f1 == 0.0
    assert score.accuracy == 0.98                              # ... and accuracy flatters it


def test_greater_is_inside_flips_the_set():
    truth = np.array([0.0, 1.0, 2.0, 3.0])
    below = score_level_set(truth, truth, 1.5)
    above = score_level_set(truth, truth, 1.5, greater_is_inside=True)
    assert below.true_positive == 2 and above.true_positive == 2
    assert below.true_negative == 2 and above.true_negative == 2


def test_metrics_reject_mismatched_or_empty_input():
    with pytest.raises(ValueError, match="shape"):
        score_level_set(np.zeros(3), np.zeros(4), 0.0)
    with pytest.raises(ValueError, match="empty"):
        score_level_set(np.zeros(0), np.zeros(0), 0.0)


# -------------------------------------------------------------------------- acquisition
def test_straddle_is_positive_exactly_when_the_interval_contains_tau():
    mu = np.array([0.0, 0.0, 5.0])
    sd = np.array([1.0, 0.01, 1.0])
    scores = straddle(mu, sd, tau=0.5, kappa=1.96)
    contains = (mu - 1.96 * sd < 0.5) & (mu + 1.96 * sd > 0.5)
    assert np.array_equal(scores > 0, contains)


def test_batch_is_ordered_and_respects_exclusions():
    candidates = np.column_stack([np.linspace(0, 1, 10), np.zeros(10), np.zeros(10)])
    scores = np.arange(10, dtype=float)
    assert select_batch(scores, candidates, 2, k=3) == [9, 8, 7]
    assert select_batch(scores, candidates, 2, k=2, exclude={9, 8}) == [7, 6]


def test_min_distance_spreads_a_batch_within_a_category():
    candidates = np.column_stack([np.linspace(0, 1, 21), np.zeros(21), np.zeros(21)])
    scores = np.zeros(21)
    scores[[10, 11, 20]] = [3.0, 2.9, 2.0]
    # without spreading it takes the two neighbours; with it, it skips to the far one
    assert select_batch(scores, candidates, 2, k=2) == [10, 11]
    assert select_batch(scores, candidates, 2, k=2, min_distance=0.2) == [10, 20]


def test_penalisation_is_confined_to_the_same_category():
    candidates = np.array([[0.5, 0.0, 0.0], [0.5, 0.0, 1.0]])   # same spot, different level
    scores = np.array([2.0, 1.0])
    assert select_batch(scores, candidates, 2, k=2, min_distance=0.9) == [0, 1]


def test_batch_returns_short_rather_than_repeating():
    candidates = np.column_stack([np.linspace(0, 1, 4), np.zeros(4), np.zeros(4)])
    picked = select_batch(np.arange(4.0), candidates, 2, k=10)
    assert len(picked) == 4 and len(set(picked)) == 4


# --------------------------------------------------------------------------------- doe
def test_stratified_doe_covers_every_category():
    space = DesignSpace({"x": Continuous(0.0, 1.0), "z": Ordinal([0, 1, 2])})
    w = stratified_doe(space, 4, seed=0)
    assert len(w) == 4 * space.m
    counts = np.bincount(space.category_of(w), minlength=space.m)
    assert np.all(counts == 4)


def test_candidate_sets_stay_inside_the_encoded_bounds():
    space = DesignSpace({"x1": Continuous(-5.0, 5.0), "x2": Continuous(0.0, 2.0), "z": Ordinal([0, 1])})
    for w in [grid_candidates(space, 5), random_candidates(space, 16, seed=0), stratified_doe(space, 3, seed=0)]:
        assert w[:, : space.q].min() >= 0.0 and w[:, : space.q].max() <= 1.0
        assert set(np.unique(w[:, space.q])) <= {0.0, 1.0}


def test_grid_refuses_to_allocate_absurdly():
    space = DesignSpace({f"x{i}": Continuous(0.0, 1.0) for i in range(8)})
    with pytest.raises(ValueError, match="limit"):
        grid_candidates(space, 20)


# ------------------------------------------------------------------------- correlation
@pytest.mark.parametrize("name", sorted(CORRELATIONS))
def test_correlation_is_one_on_the_diagonal_and_decays(name):
    corr = get_correlation(name)
    rng = np.random.default_rng(0)
    a = rng.random((6, 2))
    params = np.array([0.0] * corr.n_params(2)) if corr.n_params_per_dim == 1 else np.array([0.0, 0.0, 2.0, 2.0])

    r = corr(params, a, a)
    assert np.allclose(np.diag(r), 1.0)
    assert np.allclose(r, r.T)
    assert np.linalg.eigvalsh(r + 1e-10 * np.eye(6)).min() > 0

    near = corr(params, np.array([[0.0, 0.0]]), np.array([[0.05, 0.0]]))[0, 0]
    far = corr(params, np.array([[0.0, 0.0]]), np.array([[5.0, 0.0]]))[0, 0]
    assert near > far


def test_unknown_correlation_is_rejected():
    with pytest.raises(ValueError, match="unknown correlation"):
        get_correlation("nope")
