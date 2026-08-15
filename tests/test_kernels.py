"""The kernels of GP2.pdf Sections 3.3-3.5, and the structural claim that separates them."""
import numpy as np
import pytest

from mvlse import (
    CORRELATIONS,
    KERNEL_NAMES,
    Continuous,
    DesignSpace,
    Nominal,
    Ordinal,
    make_kernel,
    n_angles,
    pdude,
    stratified_doe,
)
from mvlse.hypersphere import ANGLE_BOUNDS


@pytest.fixture
def space():
    return DesignSpace(
        {
            "x1": Continuous(0.0, 1.0),
            "x2": Continuous(0.0, 1.0),
            "z": Ordinal([0, 1, 2]),
            "c": Nominal(["A", "B", "C"]),
        }
    )


# --------------------------------------------------------------- hypersphere (Eqs. 20-21)
@pytest.mark.parametrize("b", [2, 3, 4, 6])
def test_pdude_is_positive_definite_with_unit_diagonal(b):
    rng = np.random.default_rng(b)
    for _ in range(20):
        angles = rng.uniform(*ANGLE_BOUNDS, size=n_angles(b))
        t = pdude(angles, b)
        assert np.allclose(np.diag(t), 1.0)          # unit diagonal
        assert np.allclose(t, t.T)                    # symmetric
        assert np.linalg.eigvalsh(t).min() > 0        # positive DEFINITE, not merely semi
        assert np.all(np.abs(t) <= 1.0 + 1e-9)


def test_pdude_can_reach_negative_correlations():
    """The whole reason the hypersphere kernels exist — see Section 3.5."""
    rng = np.random.default_rng(0)
    reached = min(
        pdude(rng.uniform(*ANGLE_BOUNDS, size=n_angles(3)), 3)[0, 1] for _ in range(400)
    )
    assert reached < -0.5


# --------------------------------------------------------------------------- every kernel
@pytest.mark.parametrize("name", KERNEL_NAMES)
@pytest.mark.parametrize("correlation", sorted(CORRELATIONS))
def test_kernel_produces_a_valid_correlation_matrix(space, name, correlation):
    kernel = make_kernel(name, space, correlation)
    rng = np.random.default_rng(1)
    params = np.array([lo + (hi - lo) * rng.random() for lo, hi in kernel.bounds])
    w = stratified_doe(space, 2, seed=3)

    r = kernel(params, w, w)
    assert r.shape == (len(w), len(w))
    assert np.allclose(np.diag(r), 1.0)
    assert np.allclose(r, r.T)
    assert np.linalg.eigvalsh(r + 1e-10 * np.eye(len(w))).min() > 0
    assert np.all(np.abs(r) <= 1.0 + 1e-9)


@pytest.mark.parametrize("name", KERNEL_NAMES)
def test_hyperparameter_counts_match_the_paper(space, name):
    kernel = make_kernel(name, space, "squar_exp")
    q, r, m = space.q, space.r, space.m
    expected = {
        "dc": q + r,                                            # Eq. (15)
        "gower": q + r,                                         # Eq. (18)
        "hs_full": q + m * (m - 1) // 2,                        # Eq. (19)
        "hs_dim": q + sum(b * (b - 1) // 2 for b in space.n_levels),  # Eq. (23)
    }[name]
    assert kernel.n_params == expected


@pytest.mark.parametrize("name", KERNEL_NAMES)
def test_cross_correlation_shape(space, name):
    kernel = make_kernel(name, space, "squar_exp")
    params = np.array([0.5 * (lo + hi) for lo, hi in kernel.bounds])
    a, b = stratified_doe(space, 1, seed=1), stratified_doe(space, 2, seed=2)
    assert kernel(params, a, b).shape == (len(a), len(b))


# ----------------------------------------------------- the claim that separates the families
@pytest.mark.parametrize("name", ["dc", "gower"])
def test_coding_kernels_cannot_express_anti_correlation(space, name):
    """``exp(-theta d) > 0`` always, so no parameter setting reaches a negative entry."""
    kernel = make_kernel(name, space, "squar_exp")
    rng = np.random.default_rng(0)
    lo = np.array([b[0] for b in kernel.bounds])
    hi = np.array([b[1] for b in kernel.bounds])
    worst = 1.0
    for _ in range(500):
        params = lo + (hi - lo) * rng.random(len(lo))
        worst = min(worst, min(m.min() for m in kernel.level_correlations(params).values()))
    assert worst > 0.0


@pytest.mark.parametrize("name", ["hs_full", "hs_dim"])
def test_hypersphere_kernels_can_express_anti_correlation(space, name):
    kernel = make_kernel(name, space, "squar_exp")
    rng = np.random.default_rng(0)
    lo = np.array([b[0] for b in kernel.bounds])
    hi = np.array([b[1] for b in kernel.bounds])
    worst = 1.0
    for _ in range(500):
        params = lo + (hi - lo) * rng.random(len(lo))
        worst = min(worst, min(m.min() for m in kernel.level_correlations(params).values()))
    assert worst < -0.3


def test_gower_is_order_blind_and_dc_is_not(space):
    """Gower charges every mismatch the same; direct conversion charges by distance."""
    params_gower = np.array([0.5 * (lo + hi) for lo, hi in make_kernel("gower", space, "squar_exp").bounds])
    gower = make_kernel("gower", space, "squar_exp").level_correlations(params_gower)["z"]
    assert gower[0, 1] == pytest.approx(gower[0, 2])

    params_dc = np.array([0.5 * (lo + hi) for lo, hi in make_kernel("dc", space, "squar_exp").bounds])
    dc = make_kernel("dc", space, "squar_exp").level_correlations(params_dc)["z"]
    assert dc[0, 1] > dc[0, 2]  # neighbours are closer than the far pair


def test_level_and_category_correlation_shapes(space):
    kernel = make_kernel("hs_dim", space, "squar_exp")
    params = np.array([0.5 * (lo + hi) for lo, hi in kernel.bounds])
    levels = kernel.level_correlations(params)
    assert set(levels) == {"z", "c"}
    assert levels["z"].shape == (3, 3)
    assert kernel.category_correlations(params).shape == (space.m, space.m)


def test_unknown_kernel_name_is_rejected(space):
    with pytest.raises(ValueError, match="unknown kernel"):
        make_kernel("nope", space)
    with pytest.raises(ValueError, match="category-wise"):
        make_kernel("category_wise", space)
