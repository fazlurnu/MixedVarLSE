import numpy as np
import pytest

from mvlse import Continuous, DesignSpace, Nominal, Ordinal


def make_space(**kwargs):
    return DesignSpace(
        {
            "range_m": Continuous(100.0, 1200.0),
            "p_rx": Continuous(0.0, 1.0),
            "accuracy": Ordinal([3.0, 10.0, 30.0]),
            "material": Nominal(["steel", "alu"]),
        },
        **kwargs,
    )


def test_shape_matches_paper_notation():
    space = make_space()
    assert space.q == 2
    assert space.r == 2
    assert space.n_levels == (3, 2)
    assert space.m == 6  # m = prod(b_k)
    assert space.names == ("range_m", "p_rx", "accuracy", "material")
    assert len(space.categories) == 6


def test_encode_decode_round_trips_exactly():
    space = make_space()
    points = [
        {"range_m": 640.0, "p_rx": 0.25, "accuracy": 10.0, "material": "alu"},
        {"range_m": 100.0, "p_rx": 1.0, "accuracy": 3.0, "material": "steel"},
    ]
    w = space.encode(points)
    back = space.decode(w)
    assert np.allclose(space.encode(back), w)
    for original, recovered in zip(points, back, strict=True):
        for key, value in original.items():
            assert recovered[key] == value


def test_normalisation_is_optional_and_visible():
    normalised = make_space()
    raw = make_space(normalize=False)
    point = [{"range_m": 640.0, "p_rx": 0.25, "accuracy": 10.0, "material": "alu"}]

    assert normalised.encode(point)[0, 0] == pytest.approx((640.0 - 100.0) / 1100.0)
    assert raw.encode(point)[0, 0] == pytest.approx(640.0)
    # ... and either way the round trip is exact
    assert raw.decode(raw.encode(point))[0]["range_m"] == pytest.approx(640.0)

    assert "[0, 1]" in normalised.describe()
    assert "physical units" in raw.describe()


def test_point_supports_both_access_styles():
    space = make_space()
    point = space.decode(space.encode([{"range_m": 100.0, "p_rx": 0.0, "accuracy": 3.0, "material": "steel"}]))[0]
    assert point["material"] == point.material == "steel"
    assert set(point) == set(space.names)


def test_category_index_is_an_odometer():
    space = make_space()
    w = np.array([[0.0, 0.0, z1, z2] for z1 in range(3) for z2 in range(2)], dtype=float)
    assert list(space.category_of(w)) == list(range(6))


def test_continuous_only_drops_the_discrete_variables():
    sub = make_space().continuous_only()
    assert sub.q == 2
    assert sub.r == 0
    assert sub.m == 1


@pytest.mark.parametrize(
    "build, message",
    [
        # the variables validate themselves ...
        (lambda: Continuous(1.0, 0.0), "upper > lower"),
        (lambda: Ordinal([1]), ">= 2 levels"),
        (lambda: Nominal(["x", "x"]), "distinct"),
        # ... and the space validates the combination
        (lambda: DesignSpace({"a": Ordinal([1, 2])}), "at least one Continuous"),
        (lambda: DesignSpace({}), "at least one variable"),
        (lambda: DesignSpace({"a": Continuous(0.0, 1.0), "b": "not a variable"}), "expected Continuous"),
    ],
)
def test_rejects_ill_formed_declarations(build, message):
    with pytest.raises((ValueError, TypeError), match=message):
        build()


def test_ordinal_positions_keep_spacing_nominal_do_not():
    assert np.allclose(Ordinal([1.0, 10.0, 100.0]).positions(), [0.0, 9 / 99, 1.0])
    assert np.allclose(Nominal(["a", "b", "c"]).positions(), [0.0, 0.5, 1.0])
