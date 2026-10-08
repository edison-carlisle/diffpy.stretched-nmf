import numpy as np
import pytest
from scipy.optimize import approx_fprime

from diffpy.stretched_nmf.snmf_class import (
    SNMFOptimizer,
    _reconstruct_matrix,
)


def _synthetic_source(components, weights, stretch, rates, r):
    """Build data independently from the optimizer's reconstruction
    code."""
    source = np.zeros((len(r), weights.shape[1]))
    for k in range(components.shape[1]):
        for m in range(weights.shape[1]):
            source[:, m] += (
                weights[k, m]
                * np.interp(r / stretch[k, m], r, components[:, k])
                * np.exp(-rates[k, m] * r)
            )
    return source


def _initialize(source, components, weights, stretch, r=None, **kwargs):
    model = SNMFOptimizer(n_components=components.shape[1], **kwargs)
    model._source_matrix = source
    model._initialize_factors(source, weights, components, stretch, r=r)
    return model


@pytest.mark.parametrize("rho", [0.0, 0.1])
def test_damping_off_preserves_legacy_fit(rho):
    source = np.random.default_rng(4).random((8, 4))
    options = dict(
        n_components=2, rho=rho, random_state=4, max_iter=2, min_iter=0
    )
    default = SNMFOptimizer(**options).fit(source)
    disabled = SNMFOptimizer(
        **options, damping_components=[], damping_regularization=3.0
    ).fit(source)

    for name in ("components_", "weights_", "stretch_", "residuals_"):
        np.testing.assert_array_equal(
            getattr(default, name), getattr(disabled, name)
        )
    np.testing.assert_array_equal(default.decay_rates_, np.zeros((2, 4)))
    assert default.objective_function_ == disabled.objective_function_
    assert all(row["step"] != "d" for row in default.objective_log)


@pytest.mark.parametrize("r", [None, np.array([1.0, 1.3, 2.1, 3.8, 5.0])])
def test_zero_rate_is_exactly_undamped(r):
    components = np.array(
        [[1.0, 0.2], [0.4, 0.5], [0.7, 0.8], [0.3, 0.6], [0.2, 0.9]]
    )
    weights = np.array([[0.2, 0.7, 0.9], [0.8, 0.3, 0.1]])
    stretch = np.array([[0.8, 1.1, 1.3], [1.2, 0.9, 0.7]])
    source = np.zeros((5, 3))
    damped = _initialize(
        source, components, weights, stretch, r, damping_components=[0, 1]
    )
    undamped = _initialize(source, components, weights, stretch, r)

    np.testing.assert_array_equal(
        _reconstruct_matrix(components, weights, stretch, r=r),
        _reconstruct_matrix(
            components, weights, stretch, np.zeros_like(weights), r
        ),
    )
    for zero_tail in (False, True):
        damped._fill_tail_zero = undamped._fill_tail_zero = zero_tail
        np.testing.assert_array_equal(
            damped._get_residual_matrix(), undamped._get_residual_matrix()
        )
        for actual, expected in zip(
            damped._compute_stretched_components(),
            undamped._compute_stretched_components(),
        ):
            np.testing.assert_array_equal(actual, expected)


def test_selective_rate_update_recovers_known_damping():
    r = np.linspace(0.5, 9.0, 50)
    components = np.column_stack(
        (
            0.1 + np.exp(-(((r - 2.0) / 0.7) ** 2)),
            0.2 + np.exp(-(((r - 5.0) / 1.2) ** 2)),
        )
    )
    weights = np.array([[0.3, 0.6, 0.8, 0.4], [0.8, 0.9, 0.7, 1.0]])
    stretch = np.array([[0.9, 1.1, 1.0, 0.95], [1.1, 0.9, 0.95, 1.0]])
    rates = np.array([[0, 0, 0, 0], [0, 0.06, 0.13, 0.21]])
    source = _synthetic_source(components, weights, stretch, rates, r)
    model = _initialize(
        source, components, weights, stretch, r, damping_components=[1]
    )
    model.decay_rates_[1] = 0.1
    model._update_decay_rates()

    np.testing.assert_allclose(model.decay_rates_, rates, atol=1e-6)
    np.testing.assert_array_equal(model.decay_rates_[0], np.zeros(4))
    assert model.decay_rates_[1, 0] == 0
    assert np.linalg.norm(model._get_residual_matrix()) < 1e-6


def test_full_fit_recovers_damping_with_undamped_reference():
    r = np.linspace(0, 8, 41)
    components = (
        0.25
        + np.exp(-(((r - 2) / 0.4) ** 2))
        + 0.8 * np.exp(-(((r - 5) / 0.8) ** 2))
    )[:, None]
    weights = np.array([[1, 0.8, 0.9, 0.7]])
    rates = np.array([[0, 0.07, 0.14, 0.21]])
    source = _synthetic_source(
        components, weights, np.ones_like(weights), rates, r
    )
    model = SNMFOptimizer(
        n_components=1,
        damping_components=[0],
        random_state=1,
        max_iter=70,
        min_iter=20,
        tol=1e-8,
    ).fit(source, weights, components, np.ones_like(weights), r=r)

    np.testing.assert_allclose(model.decay_rates_, rates, atol=3e-5)
    assert model.reconstruction_err_ < 1e-4
    assert any(row["step"] == "d" for row in model.objective_log)
    np.testing.assert_array_equal(model.decay_rates_, model.best_matrices_[3])
    assert np.all(model.decay_rates_ >= 0)


def test_selected_rates_survive_normalization_and_warm_start():
    r = np.linspace(0, 6, 15)
    components = np.column_stack(
        (
            0.3 + np.exp(-((r - 1) ** 2)),
            0.2 + np.exp(-((r - 4) ** 2)),
        )
    )
    weights = np.array([[0.6, 0.4, 0.8], [0.8, 1, 0.7]])
    rates = np.array([[0, 0, 0], [0.1, 0.2, 0.3]])
    source = _synthetic_source(
        components, weights, np.ones_like(weights), rates, r
    )
    model = SNMFOptimizer(
        n_components=2, damping_components=[1], max_iter=2, min_iter=0
    ).fit(source, weights, components, init_decay_rates=rates, r=r)
    np.testing.assert_array_equal(model.decay_rates_[0], np.zeros(3))
    assert np.all(model.decay_rates_[1] > 0)
    saved_rates = model.decay_rates_.copy()
    model.max_iter = 0
    model.fit(source, reset=False)
    np.testing.assert_array_equal(model.decay_rates_, saved_rates)
    np.testing.assert_array_equal(model.r_, r)
    with pytest.raises(ValueError, match="Initial factors"):
        model.fit(source, reset=False, init_decay_rates=rates)
    with pytest.raises(ValueError, match="same grid"):
        model.fit(source, reset=False, r=r + 0.1)


@pytest.mark.parametrize("n_signals", [1, 2, 4])
def test_rate_penalty_uses_second_differences(n_signals):
    components = np.ones((5, 1))
    weights = np.ones((1, n_signals))
    model = _initialize(
        components @ weights,
        components,
        weights,
        weights,
        damping_components=[0],
        damping_regularization=3,
    )
    model.decay_rates_[0] = np.arange(n_signals) * 0.1
    assert model._damping_penalty() == pytest.approx(0, abs=1e-30)
    if n_signals == 4:
        model.decay_rates_[0] = [0, 0.1, 0.4, 0.3]
        residuals = np.zeros((5, 4))
        assert model._get_objective_function(residuals) == pytest.approx(0.3)


def test_rate_regularization_reduces_curvature():
    r = np.linspace(0, 8, 30)
    components = np.ones((30, 1))
    weights = np.ones((1, 5))
    rates = np.array([[0.1, 0.3, 0.05, 0.3, 0.1]])
    source = _synthetic_source(components, weights, weights, rates, r)
    roughness = []
    for regularization in (0, 100):
        model = _initialize(
            source,
            components,
            weights,
            weights,
            r,
            damping_components=[0],
            damping_regularization=regularization,
        )
        model._update_decay_rates()
        roughness.append(np.linalg.norm(np.diff(model.decay_rates_, n=2)))
    assert roughness[1] < roughness[0] / 2


@pytest.mark.parametrize("zero_tail", [False, True])
def test_damped_gradients_match_finite_differences(zero_tail):
    r = np.array([0.2, 0.7, 1.5, 2.3, 4.0, 6.0])
    rng = np.random.default_rng(7)
    components = 0.2 + rng.random((6, 2))
    weights = 0.2 + rng.random((2, 4))
    stretch = np.array([[0.83, 0.92, 1.17, 1.08], [1.23, 0.86, 1.13, 0.78]])
    model = _initialize(
        rng.random((6, 4)),
        components,
        weights,
        stretch,
        r,
        damping_components=[0, 1],
        damping_regularization=2.5,
        rho=0.4,
    )
    model._fill_tail_zero = zero_tail
    model.decay_rates_ = rng.uniform(0.02, 0.15, (2, 4))
    model.residuals_ = model._get_residual_matrix()
    rate_variables = model.decay_rates_.ravel()
    _, analytic = model._decay_objective_and_gradient(rate_variables)
    numerical = approx_fprime(
        rate_variables,
        lambda rates: model._decay_objective_and_gradient(rates)[0],
        1e-7,
    )
    np.testing.assert_allclose(analytic, numerical, atol=4e-6, rtol=2e-6)

    analytic = model._compute_component_gradient_zero_tail()
    numerical = approx_fprime(
        components.ravel(),
        lambda x: model._get_objective_function(
            residuals=model._get_residual_matrix(components=x.reshape(6, 2)),
            components=x.reshape(6, 2),
        ),
        1e-7,
    )
    np.testing.assert_allclose(analytic.ravel(), numerical, atol=1e-6)

    _, analytic = model._regularize_function(stretch)
    numerical = approx_fprime(
        stretch.ravel(),
        lambda a: model._regularize_function(a)[0],
        1e-7,
    )
    np.testing.assert_allclose(
        analytic.ravel(), numerical, atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("uniform", [False, True])
def test_damped_stretch_hessian_matches_finite_differences(uniform):
    r = np.array([0.2, 0.7, 1.5, 2.3, 4.0, 6.0])
    rng = np.random.default_rng(7)
    components = 0.2 + rng.random((6, 2))
    weights = 0.2 + rng.random((2, 4))
    stretch = np.tile([0.83, 0.92, 1.17, 1.08], (2, 1))
    model = _initialize(
        rng.random((6, 4)),
        components,
        weights,
        stretch,
        r,
        damping_components=[1],
        rho=0.4,
        uniform_stretch=uniform,
    )
    model.decay_rates_[1] = [0.1, 0.08, 0.12, 0.15]
    variables = model._stretch_variables().ravel()
    analytic = model._regularize_function_hessian(variables)
    numerical = approx_fprime(
        variables, lambda a: model._regularize_function(a)[1].ravel(), 1e-7
    )
    np.testing.assert_allclose(analytic, numerical, atol=2e-5, rtol=1e-5)


@pytest.mark.parametrize("slow_iter", [0, 2])
def test_damping_fits_with_uniform_stretch(slow_iter):
    r = np.linspace(0, 6, 12)
    components = np.column_stack(
        (0.2 + np.exp(-((r - 1) ** 2)), 0.3 + np.exp(-((r - 4) ** 2)))
    )
    weights = np.array([[0.4, 0.6, 0.8], [0.7, 0.5, 0.3]])
    stretch = np.tile([0.94, 1, 1.07], (2, 1))
    rates = np.array([[0, 0, 0], [0.02, 0.05, 0.08]])
    source = _synthetic_source(components, weights, stretch, rates, r)
    model = SNMFOptimizer(
        n_components=2,
        damping_components=[1],
        uniform_stretch=True,
        rho=0.5,
        damping_regularization=0.1,
        max_iter=2,
        stretch_slow_iter=slow_iter,
    ).fit(source, weights, components, stretch, init_decay_rates=rates, r=r)
    np.testing.assert_array_equal(model.stretch_[0], model.stretch_[1])
    np.testing.assert_array_equal(model.decay_rates_[0], np.zeros(3))
    assert np.all(model.decay_rates_[1] >= 0)
    assert np.isfinite(model.objective_function_)
    np.testing.assert_allclose(
        model.residuals_, model._get_residual_matrix(), atol=1e-12
    )


@pytest.mark.parametrize("indices", [[-1], [2], [0.5], [True], 0, [[0]]])
def test_invalid_damping_components_are_rejected(indices):
    with pytest.raises(ValueError, match="damping_components"):
        SNMFOptimizer(n_components=2, damping_components=indices).fit(
            np.ones((5, 3))
        )


@pytest.mark.parametrize("regularization", [-1, np.nan, np.inf])
def test_invalid_damping_regularization_is_rejected(regularization):
    with pytest.raises(ValueError, match="damping_regularization"):
        SNMFOptimizer(damping_regularization=regularization)


@pytest.mark.parametrize(
    "rates, message",
    [
        (np.ones((1, 3)), "shape"),
        (np.array([[0, 0, 0], [-0.1, 0, 0]]), "non-negative"),
        (np.array([[0, 0, 0], [np.inf, 0, 0]]), "finite"),
        (np.array([[0.1, 0, 0], [0, 0, 0]]), "excluded"),
    ],
)
def test_invalid_initial_rates_are_rejected(rates, message):
    with pytest.raises(ValueError, match=message):
        SNMFOptimizer(n_components=2, damping_components=[1]).fit(
            np.ones((5, 3)), init_decay_rates=rates
        )


@pytest.mark.parametrize(
    "r",
    [
        [0, 1],
        [0, 1, 1],
        [-1, 0, 1],
        [0, 2, 1],
        [0, 1, np.nan],
    ],
)
def test_invalid_r_grid_is_rejected(r):
    with pytest.raises(ValueError, match="r must be"):
        SNMFOptimizer(n_components=1, damping_components=[0]).fit(
            np.ones((3, 3)), r=r
        )
