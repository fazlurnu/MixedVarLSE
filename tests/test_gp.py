"""The Kriging core of Eqs. (9)-(14), deterministic and with known per-point noise."""
import numpy as np
import pytest

from mvlse import (
    MODEL_NAMES,
    Continuous,
    DesignSpace,
    Ordinal,
    fit,
    fit_surrogate,
    grid_candidates,
    make_kernel,
    stratified_doe,
)


@pytest.fixture
def space():
    return DesignSpace({"x1": Continuous(0.0, 1.0), "x2": Continuous(0.0, 1.0), "z": Ordinal([0, 1, 2])})


def surface(w):
    """Smooth, and genuinely different per level, so the discrete part has work to do."""
    return np.sin(3 * w[:, 0]) + 0.5 * w[:, 1] ** 2 + 0.4 * w[:, 2]


def test_interpolates_its_training_data(space):
    kernel = make_kernel("hs_dim", space, "squar_exp")
    w = stratified_doe(space, 6, seed=0)
    y = surface(w)
    model = fit(kernel, w, y, rng=np.random.default_rng(0), n_starts=4)

    mu, sd = model.predict(w)
    assert np.allclose(mu, y, atol=1e-3)      # noise-free GP passes through its data
    assert np.all(sd < 1e-2)                  # ... with almost no uncertainty there


def test_generalises_off_the_training_set(space):
    kernel = make_kernel("hs_dim", space, "squar_exp")
    w = stratified_doe(space, 10, seed=1)
    model = fit(kernel, w, surface(w), rng=np.random.default_rng(0), n_starts=4)

    test = grid_candidates(space, 9)
    mu, sd = model.predict(test)
    assert np.sqrt(np.mean((mu - surface(test)) ** 2)) < 0.05
    assert np.all(sd >= 0)


def test_predictions_are_in_the_original_units(space):
    """y is standardised internally; nothing may leak out."""
    kernel = make_kernel("hs_dim", space, "squar_exp")
    w = stratified_doe(space, 5, seed=2)
    for scale, offset in [(1.0, 0.0), (1000.0, -5000.0), (1e-4, 7.0)]:
        y = scale * surface(w) + offset
        model = fit(kernel, w, y, rng=np.random.default_rng(0), n_starts=3)
        mu, _ = model.predict(w)
        assert np.allclose(mu, y, rtol=1e-3, atol=1e-6 * abs(scale))


def test_known_noise_is_used_and_smooths_the_fit(space):
    kernel = make_kernel("hs_dim", space, "squar_exp")
    w = stratified_doe(space, 8, seed=3)
    truth = surface(w)
    rng = np.random.default_rng(0)
    se = np.full(len(w), 0.3)
    noisy = truth + se * rng.standard_normal(len(w))

    model = fit(kernel, w, noisy, se, rng=np.random.default_rng(0), n_starts=4)
    assert model.noise is not None
    assert len(model.params) == kernel.n_params + 1   # log10(sigma^2) joins the search

    mu, _ = model.predict(w)
    # the latent fit should sit closer to the truth than the noisy observations do
    assert np.sqrt(np.mean((mu - truth) ** 2)) < np.sqrt(np.mean((noisy - truth) ** 2))


def test_all_zero_standard_errors_take_the_deterministic_path(space):
    kernel = make_kernel("hs_dim", space, "squar_exp")
    w = stratified_doe(space, 4, seed=4)
    model = fit(kernel, w, surface(w), np.zeros(len(w)), rng=np.random.default_rng(0), n_starts=2)
    assert model.noise is None


@pytest.mark.parametrize("method", MODEL_NAMES)
def test_every_method_fits_and_predicts(space, method):
    w = stratified_doe(space, 5, seed=5)
    model = fit_surrogate(method, space, w, surface(w), rng=np.random.default_rng(0), n_starts=2)
    test = grid_candidates(space, 7)
    mu, sd = model.predict(test)
    assert mu.shape == sd.shape == (len(test),)
    assert np.all(np.isfinite(mu))
    assert np.all(sd >= 0)


def test_category_wise_flags_categories_it_could_not_fit(space):
    w = stratified_doe(space, 3, seed=6)
    keep = w[:, 2] != 2                       # starve the last level entirely
    model = fit_surrogate("category_wise", space, w[keep], surface(w[keep]), rng=np.random.default_rng(0), n_starts=2)
    assert 2 in model.unfitted
    assert model.level_correlations() == {}   # it models no relationship between levels

    mu, sd = model.predict(w[~keep])          # falls back to the pooled prior, finitely
    assert np.all(np.isfinite(mu)) and np.all(sd > 0)


def test_rejects_mismatched_shapes(space):
    kernel = make_kernel("hs_dim", space, "squar_exp")
    w = stratified_doe(space, 3, seed=7)
    with pytest.raises(ValueError, match="responses"):
        fit(kernel, w, surface(w)[:-1], rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="standard errors"):
        fit(kernel, w, surface(w), np.ones(len(w) - 1), rng=np.random.default_rng(0))
