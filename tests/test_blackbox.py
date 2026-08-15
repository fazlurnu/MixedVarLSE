"""The blackbox seam: whatever shape you return, it becomes ``(y, se)``."""
import numpy as np
import pytest

from mvlse import Continuous, DesignSpace, Nominal, Observation, evaluate, pointwise
from mvlse.blackbox import parallel


@pytest.fixture
def points():
    space = DesignSpace({"x": Continuous(0.0, 1.0), "c": Nominal(["a", "b"])})
    return space.decode(np.array([[0.0, 0], [0.5, 1], [1.0, 0]]))


@pytest.mark.parametrize(
    "returns, expected_y, expected_se",
    [
        ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [0.0, 0.0, 0.0]),
        (np.array([1.0, 2.0, 3.0]), [1.0, 2.0, 3.0], [0.0, 0.0, 0.0]),
        ([(1.0, 0.1), (2.0, 0.2), (3.0, 0.3)], [1.0, 2.0, 3.0], [0.1, 0.2, 0.3]),
        ([Observation(1.0, 0.1), Observation(2.0, 0.2), Observation(3.0, 0.3)], [1.0, 2.0, 3.0], [0.1, 0.2, 0.3]),
        (np.array([[1.0, 0.1], [2.0, 0.2], [3.0, 0.3]]), [1.0, 2.0, 3.0], [0.1, 0.2, 0.3]),
        ([{"y": 1.0, "se": 0.1}, {"y": 2.0}, {"y": 3.0, "std_error": 0.3}], [1.0, 2.0, 3.0], [0.1, 0.0, 0.3]),
    ],
)
def test_every_documented_return_shape(points, returns, expected_y, expected_se):
    y, se = evaluate(lambda _: returns, points)
    assert np.allclose(y, expected_y)
    assert np.allclose(se, expected_se)


def test_pointwise_lifts_a_scalar_function(points):
    y, se = evaluate(pointwise(lambda p: p["x"] * 2), points)
    assert np.allclose(y, [0.0, 1.0, 2.0])
    assert np.allclose(se, 0.0)


def _double(point):  # module level so joblib can pickle it
    return point["x"] * 2


def test_parallel_lifts_a_scalar_function(points):
    pytest.importorskip("joblib")
    y, _ = evaluate(parallel(_double, n_jobs=2), points)
    assert np.allclose(y, [0.0, 1.0, 2.0])


def test_empty_batch_is_not_an_error():
    y, se = evaluate(lambda _: [], [])
    assert len(y) == len(se) == 0


@pytest.mark.parametrize(
    "returns, message",
    [
        (None, "returned None"),
        ([1.0, 2.0], "exactly one output per point"),
        ([1.0, 2.0, np.nan], "non-finite"),
        ([(1.0, -0.5), (2.0, 0.1), (3.0, 0.1)], "negative or non-finite"),
    ],
)
def test_bad_returns_fail_loudly(points, returns, message):
    with pytest.raises((TypeError, ValueError), match=message):
        evaluate(lambda _: returns, points)


def test_observation_validates_itself():
    with pytest.raises(ValueError, match="non-finite"):
        Observation(np.inf)
    with pytest.raises(ValueError, match=">= 0"):
        Observation(1.0, -0.1)
