import numpy as np
import pytest
from scipy.optimize import approx_fprime

from diffpy.stretched_nmf.snmf_class import SNMFOptimizer


def _initialize(components, **kwargs):
    model = SNMFOptimizer(n_components=components.shape[1], **kwargs)
    model._source_matrix = np.zeros((components.shape[0], 3))
    model._initialize_factors(
        model._source_matrix,
        init_components=components,
        init_weights=np.full((components.shape[1], 3), 0.5),
    )
    return model


@pytest.mark.parametrize("strength", [-1, np.nan, np.inf, -np.inf])
def test_rejects_invalid_strength(strength):
    with pytest.raises(ValueError, match="smoothness_regularization"):
        SNMFOptimizer(smoothness_regularization=strength)


@pytest.mark.parametrize("indices", [[-1], [2], [0.5], [True], [[0]], 0])
def test_rejects_invalid_component_selection(indices):
    with pytest.raises(ValueError, match="smoothness_components"):
        _initialize(np.ones((5, 2)), smoothness_components=indices)


def test_penalty_selects_profiles_and_uses_candidate_components():
    components = np.column_stack(([0, 1, 0, 1, 0], [1, 3, 5, 7, 9]))
    model = _initialize(
        components, smoothness_components=[0, 0], smoothness_regularization=3
    )
    model.residuals_ = np.zeros_like(model._source_matrix)
    # Three second differences of magnitude 2 in component 0.
    assert model._get_objective_function() == 18
    candidate = components.copy()
    candidate[:, 0] = np.arange(5)
    candidate[:, 1] = [0, 10, 0, 10, 0]
    assert model._get_objective_function(components=candidate) == 0
    np.testing.assert_array_equal(
        model._component_smoothness_gradient()[:, 1], 0
    )


@pytest.mark.parametrize("length", [1, 2, 3])
def test_short_or_linear_profiles_have_zero_penalty(length):
    model = _initialize(
        np.arange(length, dtype=float)[:, None],
        smoothness_components=[0],
        smoothness_regularization=2,
    )
    assert model._component_smoothness_penalty() == 0
    np.testing.assert_array_equal(model._component_smoothness_gradient(), 0)


@pytest.mark.parametrize("zero_tail", [False, True])
@pytest.mark.parametrize("damped", [False, True])
def test_combined_component_gradient_matches_finite_differences(
    zero_tail, damped
):
    rng = np.random.default_rng(42)
    components = 0.2 + rng.random((6, 2))
    model = _initialize(
        components,
        smoothness_components=[1],
        smoothness_regularization=1.5,
        damping_components=[0] if damped else None,
    )
    model._set_r_grid(np.array([0.3, 0.7, 1.4, 2.6, 3.0, 4.2]))
    model.stretch_ = np.array([[0.8, 1.2, 1.05], [1.1, 0.9, 0.75]])
    if damped:
        model.decay_rates_[0] = [0.1, 0.3, 0.2]
    model._fill_tail_zero = zero_tail
    model.residuals_ = model._get_residual_matrix()
    model.objective_function_ = model._get_objective_function()
    model._outer_iter = model._inner_iter = 0
    model._update_components()
    analytical = model._grad_components.copy()

    def objective(values):
        candidate = values.reshape(components.shape)
        return model._get_objective_function(
            components=candidate,
            residuals=model._get_residual_matrix(components=candidate),
        )

    numerical = approx_fprime(components.ravel(), objective, 1e-7)
    np.testing.assert_allclose(
        analytical.ravel(), numerical, rtol=2e-5, atol=2e-6
    )
    assert (
        model._get_objective_function(residuals=model._get_residual_matrix())
        < model.objective_function_
    )


@pytest.mark.parametrize("selection,strength", [(None, 3), ([], 3), ([0], 0)])
def test_disabled_smoothness_preserves_fit(selection, strength):
    source = np.random.default_rng(4).random((8, 3))
    options = dict(n_components=2, random_state=4, max_iter=1, min_iter=0)
    default = SNMFOptimizer(**options).fit(source)
    disabled = SNMFOptimizer(
        **options,
        smoothness_components=selection,
        smoothness_regularization=strength,
    ).fit(source)
    for name in ("components_", "weights_", "stretch_", "residuals_"):
        np.testing.assert_array_equal(
            getattr(default, name), getattr(disabled, name)
        )
    assert default.objective_function_ == disabled.objective_function_


@pytest.mark.parametrize("eta", [0, 0.01])
def test_fit_smooths_profile_and_supports_warm_start(eta):
    profile = np.array([1, 2, 1, 2, 1, 2, 1, 2, 1], dtype=float)[:, None]
    weights = np.array([[0.4, 0.7, 1.0]])
    source = profile @ weights
    options = dict(n_components=1, max_iter=2, min_iter=0, eta=eta)
    plain = SNMFOptimizer(**options).fit(
        source, init_components=profile, init_weights=weights
    )
    smooth = SNMFOptimizer(
        **options, smoothness_components=[0], smoothness_regularization=5
    ).fit(source, init_components=profile, init_weights=weights)
    assert np.linalg.norm(np.diff(smooth.components_, n=2, axis=0)) < (
        0.2 * np.linalg.norm(np.diff(plain.components_, n=2, axis=0))
    )
    assert np.all(smooth.components_ >= 0)
    assert np.isfinite(smooth.objective_function_)
    np.testing.assert_allclose(
        smooth.objective_function_,
        0.5 * np.sum(smooth.residuals_**2)
        + eta * np.sum(np.sqrt(smooth.components_))
        + 2.5 * np.sum(np.diff(smooth.components_, n=2, axis=0) ** 2),
    )
    smooth.fit(source, reset=False)
    assert np.isfinite(smooth.objective_function_)


@pytest.mark.parametrize("uniform_stretch", [False, True])
def test_smoothness_combines_with_stretch_and_damping(uniform_stretch):
    source = 0.5 + np.random.default_rng(3).random((6, 3))
    model = SNMFOptimizer(
        n_components=2,
        random_state=3,
        max_iter=1,
        min_iter=0,
        rho=0.1,
        eta=0.01,
        uniform_stretch=uniform_stretch,
        stretch_slow_iter=0,
        damping_components=[0],
        damping_regularization=0.5,
        smoothness_components=[1],
        smoothness_regularization=2,
    ).fit(source, r=np.linspace(0, 5, 6))
    assert np.isfinite(model.objective_function_)
    assert np.all(model.components_ >= 0)
    np.testing.assert_array_equal(model.decay_rates_[1], 0)
    if uniform_stretch:
        np.testing.assert_array_equal(model.stretch_[0], model.stretch_[1])
    assert model._component_smoothness_penalty() > 0
