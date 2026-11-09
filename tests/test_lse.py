"""The active loop, its stopping rules, and the evaluation cache."""
import numpy as np
import pytest

from mvlse import (
    Continuous,
    DesignSpace,
    EvaluationStore,
    LseConfig,
    Nominal,
    Ordinal,
    grid_candidates,
    run_lse,
    score_level_set,
    straddle,
)


@pytest.fixture
def space():
    return DesignSpace(
        {"x1": Continuous(0.0, 1.0), "x2": Continuous(0.0, 1.0),
         "z": Ordinal([0, 1, 2]), "c": Nominal(["a", "b"])}
    )


BIAS = {"a": 0.0, "b": 0.35}


def blackbox(points):
    return [
        (p["x1"] - 0.5) ** 2 + (p["x2"] - 0.5) ** 2 - 0.08 * p["z"] + BIAS[p["c"]]
        for p in points
    ]


def test_loop_finds_the_level_set(space):
    result = run_lse(space, blackbox, threshold=0.1, n_init_per_category=3,
                     max_evaluations=110, seed=1, verbose=False)
    grid = grid_candidates(space, 15)
    truth = np.asarray(blackbox(space.decode(grid)))
    mu, _ = result.predict(grid)
    assert score_level_set(mu, truth, 0.1).f1 > 0.9


def test_records_are_in_physical_named_units(space):
    result = run_lse(space, blackbox, threshold=0.1, n_init_per_category=1,
                     max_evaluations=20, seed=1, verbose=False)
    records = result.records()
    assert len(records) == result.n_evaluations
    first = records[0]
    assert set(space.names) <= set(first)
    assert first["c"] in {"a", "b"}
    assert 0.0 <= first["x1"] <= 1.0
    assert isinstance(first["initial"], bool)


def test_budget_is_never_exceeded(space):
    for budget in (13, 26, 41):
        result = run_lse(space, blackbox, threshold=0.1, n_init_per_category=1,
                         max_evaluations=budget, seed=1, verbose=False)
        assert result.n_evaluations <= budget


def test_batching_acquires_several_points_per_round(space):
    result = run_lse(space, blackbox, threshold=0.1, n_init_per_category=1,
                     max_evaluations=40, batch_size=5, min_batch_distance=0.2,
                     seed=1, verbose=False)
    acquiring = [r for r in result.rounds if r.picked]
    assert acquiring and max(len(r.picked) for r in acquiring) > 1


def test_stop_reason_is_one_of_the_documented_three(space):
    result = run_lse(space, blackbox, threshold=0.1, n_init_per_category=2,
                     max_evaluations=200, seed=1, verbose=False)
    assert result.stop_reason.startswith(("confident", "plateau", "budget", "candidate set"))
    if result.stop_reason.startswith("confident"):
        assert result.rounds[-1].max_straddle < 0


def test_uncertain_agrees_with_the_straddle(space):
    result = run_lse(space, blackbox, threshold=0.1, n_init_per_category=2,
                     max_evaluations=40, seed=1, verbose=False)
    grid = grid_candidates(space, 9)
    mu, sd = result.predict(grid)
    assert np.array_equal(
        result.uncertain(grid), straddle(mu, sd, 0.1, result.config.kappa) > 0
    )


def test_greater_is_inside_flips_the_set(space):
    below = run_lse(space, blackbox, threshold=0.1, n_init_per_category=2,
                    max_evaluations=40, seed=1, verbose=False)
    above = run_lse(space, blackbox, threshold=0.1, n_init_per_category=2,
                    max_evaluations=40, seed=1, greater_is_inside=True, verbose=False)
    grid = grid_candidates(space, 9)
    assert not np.array_equal(below.inside(grid), above.inside(grid))


def test_store_makes_a_rerun_free(space, tmp_path):
    calls = []

    def counted(points):
        calls.append(len(points))
        return blackbox(points)

    path = tmp_path / "cache.jsonl"
    first = run_lse(space, counted, threshold=0.1, n_init_per_category=2,
                    max_evaluations=30, store=path, seed=1, verbose=False)
    spent_first = sum(calls)
    calls.clear()

    second = run_lse(space, counted, threshold=0.1, n_init_per_category=2,
                     max_evaluations=30, store=path, seed=1, verbose=False)
    assert second.n_evaluations == first.n_evaluations
    assert np.allclose(second.y, first.y)
    assert sum(calls) == 0                     # every point came from the cache
    assert spent_first > 0
    assert len(EvaluationStore(path, space)) == first.n_evaluations


def test_store_survives_reload(space, tmp_path):
    path = tmp_path / "cache.jsonl"
    store = EvaluationStore(path, space)
    w = grid_candidates(space, 3)[:5]
    y, se = store.evaluate(w, blackbox)
    reloaded = EvaluationStore(path, space)
    assert len(reloaded) == len(w)
    y2, se2 = reloaded.evaluate(w, lambda _: pytest.fail("should not be called"))
    assert np.allclose(y, y2) and np.allclose(se, se2)


def test_config_validation():
    with pytest.raises(ValueError, match="unknown method"):
        LseConfig(threshold=0.0, method="nope")
    with pytest.raises(ValueError, match="batch_size"):
        LseConfig(threshold=0.0, batch_size=0)


def test_initial_design_larger_than_budget_is_rejected(space):
    with pytest.raises(ValueError, match="initial design"):
        run_lse(space, blackbox, threshold=0.1, n_init_per_category=5,
                max_evaluations=10, seed=1, verbose=False)


def test_threshold_or_config_is_required(space):
    with pytest.raises(TypeError, match="threshold"):
        run_lse(space, blackbox)
    with pytest.raises(TypeError, match="both config"):
        run_lse(space, blackbox, config=LseConfig(threshold=0.1), seed=3)


def test_snapshots_are_recorded_on_request(space):
    result = run_lse(space, blackbox, threshold=0.1, n_init_per_category=1,
                     max_evaluations=25, keep_snapshots=True, seed=1, verbose=False)
    assert len(result.snapshots) == len(result.rounds)
    assert result.snapshots[0][0].shape == (len(result.monitor),)


def test_stop_when_confident_false_runs_to_the_budget() -> None:
    """The confident stop can be switched off, and the run then uses its whole budget."""
    from mvlse.problems import get_problem

    prob = get_problem("goldstein36_graded")
    kw = dict(threshold=prob.threshold, n_init_per_category=1, flip_tolerance=-1.0, verbose=False,
              method="dc", seed=1)
    stopped = run_lse(prob.space, prob, max_evaluations=250, **kw)
    assert stopped.stop_reason.startswith("confident") and stopped.n_evaluations < 250
    full = run_lse(prob.space, prob, max_evaluations=stopped.n_evaluations + 5,
                   stop_when_confident=False, **kw)
    assert full.n_evaluations == stopped.n_evaluations + 5
    assert full.stop_reason.startswith("budget")
    assert (full.w[: stopped.n_evaluations] == stopped.w).all()   # identical up to the stop


def test_gotovos_acquisition_classifies_monotonically() -> None:
    """Gotovos LSE: a classified candidate never reverts and is never sampled again."""
    from mvlse import Continuous, DesignSpace

    space = DesignSpace({"x": Continuous(0.0, 1.0)})
    fn = lambda pts: [float(np.sin(6 * p["x"])) for p in pts]
    res = run_lse(space, fn, threshold=0.2, acquisition="gotovos", max_evaluations=25,
                  n_init_per_category=4, stop_when_confident=False, flip_tolerance=-1.0, verbose=False,
                  seed=3, keep_snapshots=False)
    assert res.labels is not None and res.labels.shape == (len(res.candidates),)
    assert set(np.unique(res.labels)) <= {0, 1, 2}
    # A sample is only ever taken from the unclassified pool, and each candidate at most once.
    picked = [i for r in res.rounds for i in r.picked]
    assert len(picked) == len(set(picked))
    assert all(r.unclassified_fraction > 0 for r in res.rounds if r.picked)
    lo, hi = res.interval
    classified = res.labels > 0
    assert np.all(lo[classified] <= hi[classified])
    # inside the level set (sin <= 0.2) where classified inside, by the truth on the candidates
    x = res.candidates[:, 0]
    truth_inside = np.sin(6 * x) <= 0.2
    assert np.mean(truth_inside[res.labels == 1]) > 0.95
    assert np.mean(~truth_inside[res.labels == 2]) > 0.95


def test_acquisition_is_validated() -> None:
    from mvlse import LseConfig

    with pytest.raises(ValueError, match="acquisition"):
        LseConfig(threshold=0.0, acquisition="ucb")


def test_worst_case_acquisition_targets_the_worst_environment() -> None:
    """worst_case: samples sit at each design's worst environment value, and the worst-case
    level set max_e f(x, e) <= tau comes out right."""
    space = DesignSpace({"x": Continuous(0.0, 1.0), "e": Continuous(0.0, 1.0)})
    # worst over e is at e = 1 for every x; max_e f = 2x - 1 + 0.5, so g <= 0 for x <= 0.25
    fn = lambda pts: [2 * p["x"] - 1 + 0.5 * p["e"] ** 2 for p in pts]
    res = run_lse(space, fn, threshold=0.0, acquisition="worst_case", environment="e",
                  n_environment=11, max_evaluations=30, n_init_per_category=4,
                  stop_when_confident=False, flip_tolerance=-1.0, verbose=False, seed=3)
    # while a design is still undecided (max straddle > 0), it is sampled at its worst e = 1;
    # once everything is classified the run, without early stopping, spends the rest anywhere
    open_ = [r.max_straddle > 0 for r in res.rounds if r.picked]
    e_new = res.w[res.n_initial:, 1]
    assert any(open_) and np.all(e_new[np.array(open_)] == 1.0)
    assert res.rounds[-1].max_straddle < 0
    assert res.designs.shape[1] == 1 and len(res.environment_grid) == 11
    x = np.linspace(0, 1, 101)
    mu = np.max([res.model.predict(np.column_stack([x, np.full_like(x, e)]))[0]
                 for e in res.environment_grid], axis=0)
    assert np.mean((mu <= 0) == (x <= 0.25)) > 0.95


def test_worst_case_acquisition_validates() -> None:
    with pytest.raises(ValueError, match="environment"):
        LseConfig(threshold=0.0, acquisition="worst_case")
    with pytest.raises(ValueError, match="batch_size"):
        LseConfig(threshold=0.0, acquisition="worst_case", environment="e", batch_size=2)
