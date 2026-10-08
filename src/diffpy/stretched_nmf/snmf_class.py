import csv
import time

import cvxpy as cp
import numpy as np
from scipy.optimize import minimize
from scipy.sparse import coo_matrix, csr_matrix, diags

from diffpy.stretched_nmf.plotter import SNMFPlotter


class SNMFOptimizer:
    """An implementation of stretched NMF, including sparse stretched
    NMF.

    Instantiate the estimator with hyperparameters, then call ``fit`` to
    optimize model factors. Trailing underscores indicate that an attribute
    was determined during the fit process.

    For more information on sNMF, please reference:
    Gu, R., Rakita, Y., Lan, L. et al.
    Stretched non-negative matrix factorization.
    npj Comput Mater 10, 193 (2024) https://doi.org/10.1038/s41524-024-01377-5

    Attributes
    ----------
    stretch_ : numpy.ndarray
        The best guess (or while running, the current guess) for the stretching
        factor matrix.
    components_ : numpy.ndarray
        The best guess (or while running, the current guess) for the matrix of
        component intensities.
    weights_ : numpy.ndarray
        The best guess (or while running, the current guess) for the matrix of
        component weights.
    rho : float
        The stretching factor that influences the decomposition. Zero
        corresponds to no stretching present. Relatively insensitive and
        typically adjusted in powers of 10.
    eta : float
        The sparsity factor that influences the decomposition. Should be set
        to zero for non-sparse data such as PDF. Can be used to improve
        results for sparse data such as XRD, but due to instability, should
        be used only after first selecting the best value for rho. Suggested
        adjustment is by powers of 2.
    max_iter : int
        The maximum number of times to update each of stretch, components,
        and weights before stopping the optimization.
    min_iter : int
        The minimum number of times to update each of stretch, components,
        and weights before terminating the optimization due to low/no
        improvement.
    tol : float
        The convergence threshold. This is the minimum fractional improvement
        in the objective function to allow without terminating the
        optimization.
    n_components : int
        The referred number of components when ``init_weights`` is not
        provided to ``fit``.
    random_state : int
        The seed for the initial guesses at the matrices (stretch, components,
        and weights) created by the decomposition.
    uniform_stretch : bool
        Whether to share one stretch factor per signal across all components.
    damping_components : list of int or None
        Zero-based component indices whose decay rates are fitted. ``None``
        (the default) or an empty sequence disables damping.
    damping_regularization : float
        Nonnegative weight for the second-difference penalty on decay rates
        across signals, independent of ``rho`` and ``eta``.
    decay_rates_ : numpy.ndarray
        Nonnegative decay rates of shape ``(n_components, n_signals)`` in
        inverse units of ``r_``. Excluded components have rates exactly zero.
    r_ : numpy.ndarray
        Sample coordinates, defaulting to ``np.arange(signal_length_)``.
    n_components_ : int
        The learned number of components from initialization.
    signal_length_ : int
        The number of rows in the fitted source matrix.
    n_signals_ : int
        The number of columns in the fitted source matrix.
    objective_function_ : float
        Current objective value from the most recent update.
    objective_difference_ : float
        The change in the objective function value since the last update. A
        positive value means that the result improved.
    n_iter_ : int
        The number of outer iterations completed in ``fit``.
    """

    def __init__(
        self,
        n_components=None,
        max_iter=200,
        min_iter=20,
        tol=5e-7,
        rho=0,
        eta=0,
        random_state=None,
        show_plots=False,
        verbose=False,
        stretch_max_iter=8,
        stretch_slow_iter=200,
        uniform_stretch=False,
        damping_components=None,
        damping_regularization=0.0,
    ):
        """Initialize an instance of sNMF with estimator
        hyperparameters.

        Parameters
        ----------
        n_components : int, optional
            The number of components to extract when ``init_weights`` is not
            provided to ``fit``.
        max_iter : int
            The maximum number of times to update each of A, X, and Y before
            stopping the optimization. Optional.
        min_iter : int
            The minimum number of outer-loop iterations before convergence
            checks can stop optimization. Optional.
        tol : float
            The convergence threshold. This is the minimum fractional
            improvement in the objective function to allow without terminating
            the optimization. Note that a minimum of 20 updates are run before
            this parameter is checked. Optional.
        rho : float
            The stretching regularization hyperparameter. Zero corresponds to
            no stretching.
        eta : float
            The sparsity regularization hyperparameter. Turn off for non-sparse
            data such as PDF.
        random_state : int
            The seed for the initial guesses at the matrices (A, X, and Y)
            created by the decomposition. Optional.
        show_plots : bool
            Enables plotting at each step of the decomposition. Optional.
        stretch_max_iter : int
            Maximum number of projected-gradient stretch steps per outer-loop
            iteration. Optional.
        stretch_slow_iter : int
            Number of initial outer-loop stretch updates to solve with the
            slower constrained optimizer before switching to projected-gradient
            stretch updates. Optional.
        uniform_stretch : bool
            If ``True``, use one stretch factor per signal and share it across
            all components. The fitted ``stretch_`` attribute retains its
            ``(n_components, n_signals)`` shape with identical rows.
        damping_components : list of int or None
            Zero-based indices of components to fit for exponential damping.
            Each selected component has an independent rate per signal.
            ``None`` or ``[]`` disables damping and preserves the original
            model. Rates for excluded components remain exactly zero.
        damping_regularization : float
            Nonnegative weight for half the squared norm of the second
            differences of decay rates across signals. Defaults to zero.
        """
        if n_components is not None and n_components < 1:
            raise ValueError("n_components must be a positive integer.")
        if not isinstance(stretch_max_iter, int) or stretch_max_iter < 1:
            raise ValueError("stretch_max_iter must be a positive integer.")
        if not isinstance(stretch_slow_iter, int) or stretch_slow_iter < 0:
            raise ValueError(
                "stretch_slow_iter must be a non-negative integer."
            )
        if (
            not np.isfinite(damping_regularization)
            or damping_regularization < 0
        ):
            raise ValueError(
                "damping_regularization must be finite and non-negative."
            )

        self.n_components = n_components
        self.max_iter = max_iter
        self.min_iter = min_iter
        self.tol = tol
        self.rho = rho
        self.eta = eta
        self.random_state = random_state
        self.show_plots = show_plots
        self.verbose = verbose
        self.stretch_max_iter = stretch_max_iter
        self.stretch_slow_iter = stretch_slow_iter
        self.uniform_stretch = uniform_stretch
        self.damping_components = damping_components
        self.damping_regularization = damping_regularization

        self._rng = np.random.default_rng(self.random_state)
        self._plotter = SNMFPlotter() if self.show_plots else None
        self._fill_tail_zero = False
        self._stretch_step_size = None

    def _initialize_factors(
        self,
        source_matrix,
        init_weights=None,
        init_components=None,
        init_stretch=None,
        init_decay_rates=None,
        r=None,
    ):
        self._rng = np.random.default_rng(self.random_state)
        self.signal_length_, self.n_signals_ = source_matrix.shape

        if init_weights is None and self.n_components is None:
            raise ValueError(
                "n_components must be provided when init_weights is not set."
            )

        if init_weights is None:
            n_components = self.n_components
            weights = self._rng.beta(
                a=2.0,
                b=2.0,
                size=(n_components, self.n_signals_),
            )
        else:
            weights = np.asarray(init_weights, dtype=float)
            n_components = weights.shape[0]
            if (
                self.n_components is not None
                and self.n_components != n_components
            ):
                raise ValueError(
                    "init_weights has a different number of components than "
                    "n_components."
                )

        if init_stretch is None:
            stretch_shape = (
                1 if self.uniform_stretch else n_components,
                self.n_signals_,
            )
            stretch = np.ones(stretch_shape) + self._rng.normal(
                0,
                1e-3,
                size=stretch_shape,
            )
            if self.uniform_stretch:
                stretch = np.repeat(stretch, n_components, axis=0)
        else:
            stretch = np.asarray(init_stretch, dtype=float)

        if init_components is None:
            components = self._rng.random((self.signal_length_, n_components))
        else:
            components = np.asarray(init_components, dtype=float)

        expected_weights_shape = (n_components, self.n_signals_)
        expected_stretch_shape = (n_components, self.n_signals_)
        expected_components_shape = (self.signal_length_, n_components)

        if weights.shape != expected_weights_shape:
            raise ValueError(
                "init_weights must have shape "
                f"{expected_weights_shape}, got {weights.shape}."
            )
        if self.uniform_stretch:
            if stretch.shape == (self.n_signals_,):
                stretch = stretch[None, :]
            elif stretch.shape == (1, self.n_signals_):
                pass
            elif stretch.shape == expected_stretch_shape:
                if not np.allclose(stretch, stretch[0:1, :]):
                    raise ValueError(
                        "init_stretch must use the same stretch factor for "
                        "every component when uniform_stretch=True."
                    )
                stretch = stretch[0:1, :]
            else:
                raise ValueError(
                    "init_stretch must have shape "
                    f"{expected_stretch_shape}, (1, {self.n_signals_}), or "
                    f"({self.n_signals_},), got {stretch.shape}."
                )
            stretch = np.repeat(stretch, n_components, axis=0)
        elif stretch.shape != expected_stretch_shape:
            raise ValueError(
                "init_stretch must have shape "
                f"{expected_stretch_shape}, got {stretch.shape}."
            )
        if components.shape != expected_components_shape:
            raise ValueError(
                "init_components must have shape "
                f"{expected_components_shape}, got {components.shape}."
            )

        self.n_components_ = n_components
        self.weights_ = np.maximum(0, weights)
        self.stretch_ = stretch
        self.components_ = np.maximum(0, components)

        self._set_r_grid(r)
        self._damping_mask = np.zeros(n_components, dtype=bool)
        if self.damping_components is not None:
            indices = np.asarray(self.damping_components)
            if indices.ndim != 1 or (
                indices.size
                and (
                    indices.dtype.kind not in "iu"
                    or np.any(indices < 0)
                    or np.any(indices >= n_components)
                )
            ):
                raise ValueError(
                    "damping_components must be a sequence of zero-based "
                    "component indices less than n_components."
                )
            self._damping_mask[indices.astype(int)] = True
        self.decay_rates_ = np.zeros(expected_weights_shape)
        if init_decay_rates is not None:
            rates = np.asarray(init_decay_rates, dtype=float)
            if rates.shape != expected_weights_shape:
                raise ValueError(
                    "init_decay_rates must have shape "
                    f"{expected_weights_shape}, got {rates.shape}."
                )
            if not np.all(np.isfinite(rates)) or np.any(rates < 0):
                raise ValueError(
                    "init_decay_rates must be finite and non-negative."
                )
            if np.any(rates[~self._damping_mask] != 0):
                raise ValueError(
                    "init_decay_rates must be zero for components excluded "
                    "from damping."
                )
            self.decay_rates_ = rates.copy()

        self._init_components = self.components_.copy()
        self._init_weights = self.weights_.copy()
        self._init_stretch = self.stretch_.copy()
        self._init_decay_rates = self.decay_rates_.copy()
        self._fill_tail_zero = False
        self._stretch_step_size = None

        # Second-order spline: Tridiagonal (-2 on diags, 1 on sub/superdiags)
        self._rate_smooth_operator = (
            diags(
                [1, -2, 1],
                offsets=[0, 1, 2],
                shape=(max(self.n_signals_ - 2, 0), self.n_signals_),
                dtype=float,
            )
            if self.n_signals_ >= 3
            else csr_matrix((0, self.n_signals_))
        )
        self._spline_smooth_operator = 0.25 * self._rate_smooth_operator

    def _set_r_grid(self, r):
        """Store the independent coordinates without changing legacy
        defaults."""
        self._use_r_grid = r is not None
        if r is None:
            self.r_ = np.arange(self.signal_length_, dtype=float)
            return
        r = np.asarray(r, dtype=float)
        if (
            r.shape != (self.signal_length_,)
            or not np.all(np.isfinite(r))
            or np.any(r < 0)
            or np.any(np.diff(r) <= 0)
        ):
            raise ValueError(
                "r must be a finite, non-negative, strictly increasing "
                "array with one coordinate per source row."
            )
        self.r_ = r.copy()

    def _interpolation_coordinates(self, stretch):
        """Return fractional indices at r/a and their stretch
        derivatives."""
        stretch_inv = 1.0 / stretch
        if not getattr(self, "_use_r_grid", False):
            t = (
                np.arange(self.signal_length_, dtype=float)[:, None, None]
                * stretch_inv[None, :, :]
            )
            di = -t * stretch_inv[None, :, :]
        elif self.signal_length_ == 1:
            t = np.zeros((1, *stretch.shape))
            di = np.zeros_like(t)
        else:
            positions = self.r_[:, None, None] * stretch_inv[None, :, :]
            cells = np.clip(
                np.searchsorted(self.r_, positions, side="right") - 1,
                0,
                self.signal_length_ - 2,
            )
            spacing = self.r_[cells + 1] - self.r_[cells]
            t = cells + (positions - self.r_[cells]) / spacing
            di = -positions * stretch_inv[None, :, :] / spacing
        ddi = -di * stretch_inv[None, :, :] * 2.0
        return t, di, ddi

    def _damping_factors(self, decay_rates=None):
        rates = self.decay_rates_ if decay_rates is None else decay_rates
        return np.exp(-self.r_[:, None, None] * rates[None, :, :])

    def _damping_penalty(self, decay_rates=None):
        if not self.damping_regularization:
            return 0.0
        rates = self.decay_rates_ if decay_rates is None else decay_rates
        return (
            0.5
            * self.damping_regularization
            * np.linalg.norm(self._rate_smooth_operator @ rates.T, "fro") ** 2
        )

    def _expand_stretch(self, stretch):
        """Return stretch factors in the internal matrix
        representation."""
        stretch = np.asarray(stretch, dtype=float)
        if not self.uniform_stretch:
            if stretch.ndim == 1 and stretch.size == self.stretch_.size:
                return stretch.reshape(self.stretch_.shape)
            return stretch

        expected_shape = (self.n_components_, self.n_signals_)
        if stretch.shape == (self.n_signals_,):
            stretch = stretch[None, :]
        if stretch.shape == (1, self.n_signals_):
            return np.broadcast_to(stretch, expected_shape).copy()
        if stretch.shape != expected_shape:
            raise ValueError(
                "stretch must have shape "
                f"{expected_shape} or ({self.n_signals_},), got "
                f"{stretch.shape}."
            )
        if not np.allclose(stretch, stretch[0:1, :]):
            raise ValueError(
                "stretch must use the same factor for every component "
                "when uniform_stretch=True."
            )
        return stretch

    def _stretch_variables(self):
        """Return the independent variables used to optimize
        stretching."""
        if self.uniform_stretch:
            return self.stretch_[0, :].copy()
        return self.stretch_.copy()

    def fit(
        self,
        source_matrix,
        init_weights=None,
        init_components=None,
        init_stretch=None,
        reset=True,
        init_decay_rates=None,
        r=None,
    ):
        """Run the sNMF optimization on ``source_matrix``.

        Parameters
        ----------
        source_matrix : ndarray of shape (signal_length, n_signals)
            The source data matrix to decompose.
        init_weights : ndarray, optional
            The initial weights matrix of shape
            ``(n_components, n_signals)``.
        init_components : ndarray, optional
            Optional initial components matrix of shape
            ``(signal_length, n_components)``.
        init_stretch : ndarray, optional
            The initial stretch matrix of shape
            ``(n_components, n_signals)``.
        reset : bool
            Whether to reinitialize model factors before fitting. If ``False``,
            the previous factor matrices are reused.
        init_decay_rates : ndarray, optional
            Initial nonnegative decay rates of shape
            ``(n_components, n_signals)``. Defaults to zero. Excluded
            components must have rates zero, including when damping is off.
        r : ndarray, optional
            Finite, nonnegative, strictly increasing coordinates with one
            entry per source row. Used both for stretching at ``r / a`` and
            damping at the observed ``r``. Defaults to row indices, giving
            rates in inverse sample units. On warm starts, omitted ``r``
            reuses the fitted grid; an explicit grid must match it.

        Notes
        -----
        A component contributes ``y * x(r / a) * exp(-lambda * r)``.
        ``lambda = 0`` is exactly undamped (decay length infinity). The
        second-difference penalty follows column order, assuming equally
        spaced signals. Free profiles can absorb a common damping envelope;
        interpreting absolute rates requires an undamped reference or other
        prior information about the component profiles.
        """
        source_matrix = np.asarray(source_matrix, dtype=float)
        if source_matrix.ndim != 2:
            raise ValueError("source_matrix must be a 2D array.")
        self.converged_ = False

        self._source_matrix = source_matrix

        if reset:
            self._initialize_factors(
                source_matrix=source_matrix,
                init_weights=init_weights,
                init_components=init_components,
                init_stretch=init_stretch,
                init_decay_rates=init_decay_rates,
                r=r,
            )
        else:
            if any(
                v is not None
                for v in (
                    init_weights,
                    init_components,
                    init_stretch,
                    init_decay_rates,
                )
            ):
                raise ValueError(
                    "Initial factors can only " "be provided when reset=True."
                )
            if not all(
                hasattr(self, name)
                for name in (
                    "components_",
                    "weights_",
                    "stretch_",
                    "decay_rates_",
                    "n_components_",
                    "signal_length_",
                    "n_signals_",
                    "_spline_smooth_operator",
                )
            ):
                raise ValueError(
                    "Cannot warm-start before initialization. Call fit with "
                    "reset=True first."
                )
            expected_shape = (self.signal_length_, self.n_signals_)
            if source_matrix.shape != expected_shape:
                raise ValueError(
                    "Warm-start requires source_matrix to keep the same shape "
                    f"{expected_shape}, got {source_matrix.shape}."
                )
            if r is not None and not np.array_equal(r, self.r_):
                raise ValueError(
                    "Warm-start requires r to keep the same grid."
                )

        # Set stretch matrix to 1 if no stretching present
        if self.rho == 0:
            self.stretch_ = np.ones_like(self.stretch_)

        # Set up residual matrix, objective function, and history
        self.residuals_ = self._get_residual_matrix()
        self.objective_function_ = self._get_objective_function()
        self.best_objective_ = self.objective_function_
        self.best_matrices_ = [
            self.components_.copy(),
            self.weights_.copy(),
            self.stretch_.copy(),
            self.decay_rates_.copy(),
        ]
        self.objective_difference_ = None
        self.objective_log = [
            {
                "step": "start",
                "iteration": 0,
                "objective": self.objective_function_,
                "timestamp": time.time(),
            }
        ]

        # Set up tracking variables for _update_components()
        self._prev_components = None
        self._grad_components = np.zeros_like(self.components_)
        self._prev_grad_components = np.zeros_like(self.components_)
        self.n_iter_ = 0

        regularization_term = (
            0.5
            * self.rho
            * np.linalg.norm(
                self._spline_smooth_operator @ self.stretch_.T, "fro"
            )
            ** 2
        )
        sparsity_term = self.eta * np.sum(
            np.sqrt(self.components_)
        )  # Square root penalty
        base_obj = (
            self.objective_function_
            - regularization_term
            - sparsity_term
            - self._damping_penalty()
        )
        if self.verbose:
            print(
                f"\n--- Start ---"
                f"\nTotal Objective   : {self.objective_function_:.5e}"
                f"\nBase Obj (No Reg) : {base_obj:.5e}"
            )

        # Main optimization loop
        for outiter in range(self.max_iter):
            self._outer_iter = outiter
            previous_objective = self.objective_function_
            self._outer_loop()
            if np.any(self._damping_mask):
                self.objective_difference_ = (
                    previous_objective - self.objective_function_
                )
            self.n_iter_ = outiter + 1
            # Print diagnostics
            regularization_term = (
                0.5
                * self.rho
                * np.linalg.norm(
                    self._spline_smooth_operator @ self.stretch_.T, "fro"
                )
                ** 2
            )
            sparsity_term = self.eta * np.sum(
                np.sqrt(self.components_)
            )  # Square root penalty
            base_obj = (
                self.objective_function_
                - regularization_term
                - sparsity_term
                - self._damping_penalty()
            )
            convergence_threshold = self.objective_function_ * self.tol
            # Convergence check: Stop if diffun is small
            # and at least min_iter iterations have passed
            if self.verbose:
                print(
                    f"\n--- Iteration {self._outer_iter} ---"
                    f"\nTotal Objective   : {self.objective_function_:.5e}"
                    f"\nBase Obj (No Reg) : {base_obj:.5e}"
                    "\nConvergence Check : Δ "
                    f"({self.objective_difference_:.2e})"
                    f" < Threshold ({convergence_threshold:.2e})\n"
                )
            if (
                self.objective_difference_ < convergence_threshold
                and outiter >= self.min_iter
            ):
                self.converged_ = True
                break

        self._normalize_results()
        self.reconstruction_err_ = np.linalg.norm(self.residuals_, "fro")

        return self

    def save_objective_log(self, filename):
        """Save the objective log to a tab-delimited text file.

        Parameters
        ----------
        filename : str or path-like
            Output file path.
        """
        if not hasattr(self, "objective_log"):
            raise ValueError("Cannot save objective_log before calling fit.")

        fieldnames = ("step", "iteration", "objective", "dt_ms")
        with open(filename, "w", newline="") as fp:
            writer = csv.DictWriter(
                fp,
                fieldnames=fieldnames,
                delimiter="\t",
            )
            writer.writeheader()
            previous_timestamp = None
            for row in self.objective_log:
                timestamp = row["timestamp"]
                dt_ms = (
                    0.0
                    if previous_timestamp is None
                    else 1000 * (timestamp - previous_timestamp)
                )
                writer.writerow(
                    {
                        "step": row["step"],
                        "iteration": row["iteration"],
                        "objective": f"{row['objective']:.6E}",
                        "dt_ms": f"+{dt_ms:.3g}",
                    }
                )
                previous_timestamp = timestamp

    def _normalize_results(self):
        if self.verbose:
            print("\nNormalizing results after convergence...")
        # Select our best results for normalization
        self.components_ = self.best_matrices_[0]
        self.weights_ = self.best_matrices_[1]
        self.stretch_ = self.best_matrices_[2]
        self.decay_rates_ = self.best_matrices_[3]

        # Normalize weights/stretch first
        weights_row_max = np.max(self.weights_, axis=1, keepdims=True)
        self.weights_ = self.weights_ / weights_row_max
        stretch_row_max = np.max(self.stretch_, axis=1, keepdims=True)
        self.stretch_ = self.stretch_ / stretch_row_max

        # re-running with component updates only vs normalized weights/stretch
        self._grad_components = np.zeros_like(
            self.components_
        )  # Gradient of X (zeros for now)
        self._prev_grad_components = np.zeros_like(
            self.components_
        )  # Previous gradient of X (zeros for now)
        # Keep the fitting model's endpoint extrapolation for the new API.
        # Existing calls retain the original zero-tail normalization.
        self._fill_tail_zero = not (
            np.any(self._damping_mask) or self._use_r_grid
        )
        try:
            self.residuals_ = self._get_residual_matrix()
            self.objective_function_ = self._get_objective_function()
            self.objective_difference_ = None
            self._objective_history = [self.objective_function_]
            self._outer_iter = 0
            self._inner_iter = 0
            normalization_max_iter = max(self.max_iter, 100)
            for outiter in range(normalization_max_iter):
                self._outer_iter = outiter
                if outiter == 1:
                    self._inner_iter = (
                        1  # So step size can adapt without an inner loop
                    )
                self._update_components()
                self.residuals_ = self._get_residual_matrix()
                self.objective_function_ = self._get_objective_function()
                self.objective_log.append(
                    {
                        "step": "c_norm",
                        "iteration": outiter,
                        "objective": self.objective_function_,
                        "timestamp": time.time(),
                    }
                )
                self.objective_difference_ = (
                    self.objective_log[-2]["objective"]
                    - self.objective_log[-1]["objective"]
                )
                if self._plotter is not None:
                    self._plotter.update(
                        components=self.components_,
                        weights=self.weights_,
                        stretch=self.stretch_,
                        update_tag="normalize components",
                    )
                convergence_threshold = self.objective_function_ * self.tol
                if self.verbose:
                    print(
                        f"\n--- Iteration {outiter} after normalization---"
                        f"\nTotal Objective   : {self.objective_function_:.5e}"
                        "\nConvergence Check : Delta "
                        f"({self.objective_difference_:.2e})"
                        f" < Threshold ({convergence_threshold:.2e})\n"
                    )
                if (
                    self.objective_difference_ < convergence_threshold
                    and outiter >= 7
                ):
                    break
        finally:
            self._fill_tail_zero = False

    def _outer_loop(self):
        if self.verbose:
            print("Updating components and weights...")
        for inner_iter in range(4):
            self._inner_iter = inner_iter
            self._prev_grad_components = self._grad_components.copy()
            self._update_components()
            self.residuals_ = self._get_residual_matrix()
            self.objective_function_ = self._get_objective_function()
            self.objective_log.append(
                {
                    "step": "c",
                    "iteration": self._outer_iter,
                    "objective": self.objective_function_,
                    "timestamp": time.time(),
                }
            )
            self.objective_difference_ = (
                self.objective_log[-2]["objective"]
                - self.objective_log[-1]["objective"]
            )
            if self.objective_function_ < self.best_objective_:
                self.best_objective_ = self.objective_function_
                self.best_matrices_ = [
                    self.components_.copy(),
                    self.weights_.copy(),
                    self.stretch_.copy(),
                    self.decay_rates_.copy(),
                ]
            if self._plotter is not None:
                self._plotter.update(
                    components=self.components_,
                    weights=self.weights_,
                    stretch=self.stretch_,
                    update_tag="components",
                )

            self._update_weights()
            self.residuals_ = self._get_residual_matrix()
            self.objective_function_ = self._get_objective_function()
            self.objective_log.append(
                {
                    "step": "w",
                    "iteration": self._outer_iter,
                    "objective": self.objective_function_,
                    "timestamp": time.time(),
                }
            )

            self.objective_difference_ = (
                self.objective_log[-2]["objective"]
                - self.objective_log[-1]["objective"]
            )
            if self.objective_function_ < self.best_objective_:
                self.best_objective_ = self.objective_function_
                self.best_matrices_ = [
                    self.components_.copy(),
                    self.weights_.copy(),
                    self.stretch_.copy(),
                    self.decay_rates_.copy(),
                ]
            if self._plotter is not None:
                self._plotter.update(
                    components=self.components_,
                    weights=self.weights_,
                    stretch=self.stretch_,
                    update_tag="weights",
                )

            self.objective_difference_ = (
                self.objective_log[-2]["objective"]
                - self.objective_log[-1]["objective"]
            )
            if (
                self.objective_log[-3]["objective"] - self.objective_function_
                < self.objective_difference_ * 1e-3
            ):
                break

        # Skip updating stretch if no stretching factor
        if not self.rho == 0:
            self._update_stretch()
            self.residuals_ = self._get_residual_matrix()
            self.objective_function_ = self._get_objective_function()
            self.objective_log.append(
                {
                    "step": "s",
                    "iteration": self._outer_iter,
                    "objective": self.objective_function_,
                    "timestamp": time.time(),
                }
            )
            self.objective_difference_ = (
                self.objective_log[-2]["objective"]
                - self.objective_log[-1]["objective"]
            )
            if self.objective_function_ < self.best_objective_:
                self.best_objective_ = self.objective_function_
                self.best_matrices_ = [
                    self.components_.copy(),
                    self.weights_.copy(),
                    self.stretch_.copy(),
                    self.decay_rates_.copy(),
                ]
            if self._plotter is not None:
                self._plotter.update(
                    components=self.components_,
                    weights=self.weights_,
                    stretch=self.stretch_,
                    update_tag="stretch",
                )

        if np.any(self._damping_mask):
            self._update_decay_rates()
            self.residuals_ = self._get_residual_matrix()
            self.objective_function_ = self._get_objective_function()
            self.objective_log.append(
                {
                    "step": "d",
                    "iteration": self._outer_iter,
                    "objective": self.objective_function_,
                    "timestamp": time.time(),
                }
            )
            self.objective_difference_ = (
                self.objective_log[-2]["objective"] - self.objective_function_
            )
            if self.objective_function_ < self.best_objective_:
                self.best_objective_ = self.objective_function_
                self.best_matrices_ = [
                    self.components_.copy(),
                    self.weights_.copy(),
                    self.stretch_.copy(),
                    self.decay_rates_.copy(),
                ]

    def _get_residual_matrix(
        self, components=None, weights=None, stretch=None, decay_rates=None
    ):
        """Return the residuals (difference) between the source matrix
        and its reconstruction.

        Parameters
        ----------
        components : (signal_len, n_components) array, optional
        weights    : (n_components, n_signals) array, optional
        stretch    : (n_components, n_signals) array, optional
        decay_rates : (n_components, n_signals) array, optional

        Returns
        -------
        residuals : (signal_len, n_signals) array
        """
        if components is None:
            components = self.components_
        if weights is None:
            weights = self.weights_
        if stretch is None:
            stretch = self.stretch_
        stretch = self._expand_stretch(stretch)

        if self._fill_tail_zero:
            reconstructed_matrix = self._reconstruct_from_stretched_components(
                components=components,
                weights=weights,
                stretch=stretch,
                decay_rates=decay_rates,
            )
        else:
            reconstructed_matrix = _reconstruct_matrix(
                components,
                weights,
                stretch,
                decay_rates=(
                    (self.decay_rates_ if decay_rates is None else decay_rates)
                    if np.any(self._damping_mask)
                    else None
                ),
                r=self.r_ if self._use_r_grid else None,
            )
        residuals = reconstructed_matrix - self._source_matrix

        return residuals

    def _reconstruct_from_stretched_components(
        self, components=None, weights=None, stretch=None, decay_rates=None
    ):
        stretched_components, _, _ = self._compute_stretched_components(
            components=components,
            weights=weights,
            stretch=stretch,
            decay_rates=decay_rates,
        )
        intermediate = stretched_components.flatten(order="F").reshape(
            (self.signal_length_ * self.n_signals_, self.n_components_),
            order="F",
        )
        return intermediate.sum(axis=1).reshape(
            (self.signal_length_, self.n_signals_),
            order="F",
        )

    def _get_objective_function(
        self, residuals=None, stretch=None, components=None, decay_rates=None
    ):
        """Return the objective value, passing stored attributes or
        overrides to _compute_objective_function().

        Parameters
        ----------
        residuals : ndarray, optional
            Residual matrix to use instead of self.residuals_.
        stretch : ndarray, optional
            Stretch matrix to use instead of self.stretch_.
        components : ndarray, optional
            Component matrix to use instead of self.components_.
        decay_rates : ndarray, optional
            Decay rates to use instead of self.decay_rates_.

        Returns
        -------
        float
            Current objective function value.
        """
        return SNMFOptimizer._compute_objective_function(
            components=self.components_ if components is None else components,
            residuals=self.residuals_ if residuals is None else residuals,
            stretch=self._expand_stretch(
                self.stretch_ if stretch is None else stretch
            ),
            rho=self.rho,
            eta=self.eta,
            spline_smooth_operator=self._spline_smooth_operator,
        ) + self._damping_penalty(decay_rates)

    def _compute_stretched_components(
        self, components=None, weights=None, stretch=None, decay_rates=None
    ):
        """Interpolates each component along its sample axis according
        to per-(component, signal) stretch factors, then applies
        per-(component, signal) weights. Also computes the first and
        second derivatives with respect to stretch. Left and right,
        respectively, refer to the sample prior to and subsequent to the
        interpolated sample's position.

        Inputs
        ------
        components : array, shape (signal_len, n_components)
            Each column is a component with signal_len samples.
        weights : array, shape (n_components, n_signals)
            Per-(component, signal) weights.
        stretch : array, shape (n_components, n_signals)
            Per-(component, signal) stretch factors.
        decay_rates : array, shape (n_components, n_signals), optional
            Rates evaluated at the observed coordinates, after stretching.

        Outputs
        -------
        stretched_components : array, shape (signal_len, n_comps * n_sigs)
            Interpolated, weighted, and optionally damped components.
        d_stretched_components : array, shape (signal_len, n_comps * n_sigs)
            First derivatives with respect to stretch.
        dd_stretched_components : array, shape (signal_len, n_comps * n_sigs)
            Second derivatives with respect to stretch.
        """
        # --- Defaults ---
        if components is None:
            components = self.components_
        if weights is None:
            weights = self.weights_
        if stretch is None:
            stretch = self.stretch_
        stretch = self._expand_stretch(stretch)

        # Dimensions
        signal_len = components.shape[0]  # number of samples
        n_components = components.shape[1]  # number of components
        n_signals = weights.shape[1]  # number of signals

        # Guard stretches
        eps = 1e-8
        stretch = np.clip(stretch, eps, None)

        # Apply stretching to the original sample indices,
        # represented as a "time-stretch"
        t, di, ddi = self._interpolation_coordinates(stretch)
        # has shape (signal_len, n_components, n_signals)

        # For each stretched coordinate, find its prior integer (original)
        # index and their difference
        i0_raw = np.floor(t).astype(np.int64)  # prior original index
        alpha = t - i0_raw.astype(float)
        i1_raw = i0_raw + 1

        # Clip indices to range (0, signal_len - 1) to gather safely.
        max_idx = signal_len - 1
        i0 = np.clip(i0_raw, 0, max_idx)
        i1 = np.clip(i1_raw, 0, max_idx)

        # Gather sample values
        comps_3d = components[
            :, :, None
        ]  # expand components by a dim for broadcasting across n_signals
        c0 = np.take_along_axis(comps_3d, i0, axis=0)  # left sample values
        c1 = np.take_along_axis(comps_3d, i1, axis=0)  # right sample values
        if self._fill_tail_zero:
            c0 = np.where(i0_raw < signal_len, c0, 0.0)
            c1 = np.where(i1_raw < signal_len, c1, 0.0)

        # Linear interpolation to determine stretched sample values
        interp = c0 * (1.0 - alpha) + c1 * alpha
        interp_weighted = interp * weights[None, :, :]

        # Derivatives
        d_unweighted = c0 * (-di) + c1 * di
        dd_unweighted = c0 * (-ddi) + c1 * ddi

        d_weighted = d_unweighted * weights[None, :, :]
        dd_weighted = dd_unweighted * weights[None, :, :]

        if np.any(getattr(self, "_damping_mask", False)):
            damping = self._damping_factors(decay_rates)
            interp_weighted *= damping
            d_weighted *= damping
            dd_weighted *= damping

        # Flatten back to expected shape (signal_len, n_components * n_signals)
        return (
            interp_weighted.reshape(signal_len, n_components * n_signals),
            d_weighted.reshape(signal_len, n_components * n_signals),
            dd_weighted.reshape(signal_len, n_components * n_signals),
        )

    def _apply_transformation_matrix(
        self, stretch=None, weights=None, residuals=None
    ):
        """Computes the transformation matrix `stretch_transformed` for
        residuals, using scaling matrix `stretch` and weight
        coefficients `weights`."""
        if stretch is None:
            stretch = self.stretch_
        stretch = self._expand_stretch(stretch)
        if weights is None:
            weights = self.weights_
        if residuals is None:
            residuals = self.residuals_

        # Compute scaling matrix
        stretch_tiled = np.tile(
            stretch.reshape(1, self.n_signals_ * self.n_components_, order="F")
            ** -1,
            (self.signal_length_, 1),
        )

        # Compute indices
        indices = np.arange(self.signal_length_)[:, None] * stretch_tiled

        # Weighting coefficients
        weights_tiled = np.tile(
            weights.reshape(
                1, self.n_signals_ * self.n_components_, order="F"
            ),
            (self.signal_length_, 1),
        )

        # Compute floor indices
        floor_indices = np.floor(indices).astype(int)
        floor_indices_1 = np.minimum(floor_indices + 1, self.signal_length_)
        floor_indices_2 = np.minimum(floor_indices_1 + 1, self.signal_length_)

        # Compute fractional part
        fractional_indices = indices - floor_indices

        # Expand row indices
        repm = np.tile(
            np.arange(self.n_components_),
            (self.signal_length_, self.n_signals_),
        )

        # Compute transformations
        kron = np.kron(residuals, np.ones((1, self.n_components_)))
        fractional_kron = kron * fractional_indices
        fractional_weights = (fractional_indices - 1) * weights_tiled

        # Construct sparse matrices
        x2 = coo_matrix(
            (
                (-kron * fractional_weights).flatten(),
                (floor_indices_1.flatten() - 1, repm.flatten()),
            ),
            shape=(self.signal_length_ + 1, self.n_components_),
        ).tocsc()
        x3 = coo_matrix(
            (
                (fractional_kron * weights_tiled).flatten(),
                (floor_indices_2.flatten() - 1, repm.flatten()),
            ),
            shape=(self.signal_length_ + 1, self.n_components_),
        ).tocsc()

        # Combine the last row into previous, then remove the last row
        x2[self.signal_length_ - 1, :] += x2[self.signal_length_, :]
        x3[self.signal_length_ - 1, :] += x3[self.signal_length_, :]
        x2 = x2[:-1, :]
        x3 = x3[:-1, :]

        stretch_transformed = x2 + x3

        return stretch_transformed

    def _compute_component_gradient_zero_tail(
        self, stretch=None, weights=None, residuals=None
    ):
        """Apply the interpolation adjoint, using the current tail
        policy."""
        if stretch is None:
            stretch = self.stretch_
        stretch = self._expand_stretch(stretch)
        if weights is None:
            weights = self.weights_
        if residuals is None:
            residuals = self.residuals_

        gradient = np.zeros_like(self.components_)
        positions = None
        if getattr(self, "_use_r_grid", False):
            positions, _, _ = self._interpolation_coordinates(stretch)
        sample_indices = np.arange(self.signal_length_, dtype=float)
        damping = (
            self._damping_factors()
            if np.any(getattr(self, "_damping_mask", False))
            else None
        )
        for signal in range(self.n_signals_):
            for comp in range(self.n_components_):
                t = (
                    sample_indices / stretch[comp, signal]
                    if positions is None
                    else positions[:, comp, signal]
                )
                left_indices = np.floor(t).astype(int)
                alpha = t - left_indices
                right_indices = left_indices + 1
                scaled_residual = residuals[:, signal] * weights[comp, signal]
                if damping is not None:
                    scaled_residual = (
                        scaled_residual * damping[:, comp, signal]
                    )

                left_mask = (
                    left_indices < self.signal_length_
                    if self._fill_tail_zero
                    else np.ones(t.shape, dtype=bool)
                )
                np.add.at(
                    gradient[:, comp],
                    np.clip(
                        left_indices[left_mask], 0, self.signal_length_ - 1
                    ),
                    scaled_residual[left_mask] * (1.0 - alpha[left_mask]),
                )

                right_mask = (
                    right_indices < self.signal_length_
                    if self._fill_tail_zero
                    else np.ones(t.shape, dtype=bool)
                )
                np.add.at(
                    gradient[:, comp],
                    np.clip(
                        right_indices[right_mask], 0, self.signal_length_ - 1
                    ),
                    scaled_residual[right_mask] * alpha[right_mask],
                )

        return gradient

    def _solve_quadratic_program(self, t, m):
        """
        Solves the quadratic program for updating y in stretched NMF:

            min J(y) = 0.5 * y^T q y + d^T y
            subject to: 0 ≤ y ≤ 1

        Parameters:
        - t: (N, k) ndarray
        - source_matrix_col: (N,) column of source_matrix for the
        corresponding m

        Returns:
        - y: (k,) optimal solution
        """

        source_matrix_col = self._source_matrix[:, m]

        # Compute q and d
        q = t.T @ t  # Gram matrix (k x k)
        d = -t.T @ source_matrix_col  # Linear term (k,)

        k = q.shape[0]  # Number of variables

        # Regularize q to ensure positive semi-definiteness
        reg_factor = 1e-8 * np.linalg.norm(
            q, ord="fro"
        )  # Adaptive regularization, original was fixed
        q += np.eye(k) * reg_factor

        # Define optimization variable
        y = cp.Variable(k)

        # Define quadratic objective
        objective = cp.Minimize(0.5 * cp.quad_form(y, q) + d.T @ y)

        # Define constraints (0 ≤ y ≤ 1)
        constraints = [y >= 0, y <= 1]

        # Solve using a QP solver
        prob = cp.Problem(objective, constraints)
        prob.solve(
            solver=cp.OSQP,
            verbose=False,
            polish=False,  # TODO keep? removes polish message
            # solver_verbose=False
        )

        # Get the solution
        return np.maximum(
            y.value, 0
        )  # Ensure non-negative values in case of solver tolerance issues

    def _update_components(self):
        """Updates `components` using gradient-based optimization with
        adaptive step size."""
        # Compute stretched components using the interpolation function
        stretched_components, _, _ = (
            self._compute_stretched_components()
        )  # Discard the derivatives
        # Compute reshaped_stretched_components and component_residuals
        intermediate_reshaped = stretched_components.flatten(
            order="F"
        ).reshape(
            (self.signal_length_ * self.n_signals_, self.n_components_),
            order="F",
        )
        reshaped_stretched_components = intermediate_reshaped.sum(
            axis=1
        ).reshape((self.signal_length_, self.n_signals_), order="F")
        component_residuals = (
            reshaped_stretched_components - self._source_matrix
        )
        # Compute gradient
        if (
            self._fill_tail_zero
            or np.any(getattr(self, "_damping_mask", False))
            or getattr(self, "_use_r_grid", False)
        ):
            self._grad_components = self._compute_component_gradient_zero_tail(
                residuals=component_residuals
            )
        else:
            self._grad_components = self._apply_transformation_matrix(
                residuals=component_residuals
            ).toarray()  # toarray equivalent of full, make non-sparse

        # Compute initial step size `initial_step_size`
        initial_step_size = np.linalg.eigvalsh(
            self.weights_.T @ self.weights_
        ).max() * np.max([self.stretch_.max(), 1 / self.stretch_.min()])
        # Compute adaptive step size `step_size`
        if self._outer_iter == 0 and self._inner_iter == 0:
            step_size = initial_step_size
        else:
            num = np.sum(
                (self._grad_components - self._prev_grad_components)
                * (self.components_ - self._prev_components)
            )  # Element-wise multiplication
            denom = (
                np.linalg.norm(self.components_ - self._prev_components, "fro")
                ** 2
            )  # Frobenius norm squared
            step_size = num / denom if denom > 0 else initial_step_size
            if step_size <= 0:
                step_size = initial_step_size

        # Store our old X before updating because it is used in step selection
        self._prev_components = self.components_.copy()

        while True:  # iterate updating components
            components_step = (
                self._prev_components - self._grad_components / step_size
            )
            # Solve x^3 + p*x + q = 0 for the largest real root
            candidate_components = np.square(
                _cubic_largest_real_root(
                    -components_step, self.eta / (2 * step_size)
                )
            )
            # Mask values that should be set to zero
            mask = (
                candidate_components**2 * step_size / 2
                - step_size * candidate_components * components_step
                + self.eta * np.sqrt(candidate_components)
                < 0
            )
            candidate_components = mask * candidate_components

            objective_improvement = (
                self.objective_function_
                - self._get_objective_function(
                    components=candidate_components,
                    residuals=self._get_residual_matrix(
                        components=candidate_components
                    ),
                )
            )

            if (
                np.isfinite(objective_improvement)
                and objective_improvement > 0
            ):
                self.components_ = candidate_components
                break
            # If not, increase step_size (step size)
            step_size *= 2
            if np.isinf(step_size):
                self.components_ = self._prev_components
                break

    def _update_weights(self):
        """Updates weights by building the stretched component matrix
        `stretched_comps` with np.interp and solving a quadratic program
        for each signal."""
        sample_indices = np.arange(self.signal_length_)
        for signal in range(self.n_signals_):
            # Stretch factors for this signal across components:
            this_stretch = self.stretch_[:, signal]
            # Build stretched_comps[:, k] by interpolating component at frac.
            # pos. index / this_stretch[comp]
            stretched_comps = np.empty(
                (self.signal_length_, self.n_components_),
                dtype=self.components_.dtype,
            )
            for comp in range(self.n_components_):
                pos = sample_indices / this_stretch[comp]
                stretched_comps[:, comp] = np.interp(
                    (
                        self.r_ / this_stretch[comp]
                        if self._use_r_grid
                        else pos
                    ),
                    self.r_ if self._use_r_grid else sample_indices,
                    self.components_[:, comp],
                    left=self.components_[0, comp],
                    right=self.components_[-1, comp],
                )

            if np.any(self._damping_mask):
                stretched_comps *= np.exp(
                    -self.r_[:, None] * self.decay_rates_[:, signal][None, :]
                )

            # Solve quadratic problem for a given signal and update its weight
            new_weight = self._solve_quadratic_program(
                t=stretched_comps, m=signal
            )
            self.weights_[:, signal] = new_weight

    def _decay_objective_and_gradient(self, rate_variables):
        """Evaluate the objective and analytic gradient for selected
        rates."""
        rates = np.zeros_like(self.decay_rates_)
        rates[self._damping_mask] = np.asarray(rate_variables).reshape(
            (-1, self.n_signals_)
        )
        contributions, _, _ = self._compute_stretched_components(
            decay_rates=rates
        )
        contributions = contributions.reshape(
            self.signal_length_, self.n_components_, self.n_signals_
        )
        residuals = contributions.sum(axis=1) - self._source_matrix
        gradient = -np.sum(
            self.r_[:, None, None] * contributions * residuals[:, None, :],
            axis=0,
        )
        gradient += (
            self.damping_regularization
            * (
                self._rate_smooth_operator.T
                @ (self._rate_smooth_operator @ rates.T)
            ).T
        )
        return (
            self._get_objective_function(
                residuals=residuals, decay_rates=rates
            ),
            gradient[self._damping_mask].ravel(),
        )

    def _update_decay_rates(self):
        """Fit selected rates with exact zero lower bounds using
        L-BFGS-B."""
        if not np.any(self._damping_mask):
            return
        if self.verbose:
            print("Updating decay rates...")
        initial = self.decay_rates_[self._damping_mask].ravel()
        current_objective, _ = self._decay_objective_and_gradient(initial)
        result = minimize(
            self._decay_objective_and_gradient,
            initial,
            method="L-BFGS-B",
            jac=True,
            bounds=[(0.0, None)] * initial.size,
            options={"maxiter": 100, "ftol": 1e-12, "gtol": 1e-8},
        )
        candidate = np.maximum(result.x, 0.0)
        if not np.all(np.isfinite(candidate)):
            return
        candidate_objective, _ = self._decay_objective_and_gradient(candidate)
        if candidate_objective <= current_objective:
            self.decay_rates_[self._damping_mask] = candidate.reshape(
                (-1, self.n_signals_)
            )
        self.decay_rates_[~self._damping_mask] = 0.0

    def _stretch_residual_and_derivatives(self, stretch):
        stretched_components, d_stretch_comps, dd_stretch_comps = (
            self._compute_stretched_components(stretch=stretch)
        )
        intermediate = stretched_components.flatten(order="F").reshape(
            (self.signal_length_ * self.n_signals_, self.n_components_),
            order="F",
        )
        residuals = (
            intermediate.sum(axis=1).reshape(
                (self.signal_length_, self.n_signals_), order="F"
            )
            - self._source_matrix
        )
        return residuals, d_stretch_comps, dd_stretch_comps

    def _regularize_function(self, stretch=None):
        if stretch is None:
            stretch = self.stretch_
        stretch = self._expand_stretch(stretch)

        residuals, d_stretch_comps, _ = self._stretch_residual_and_derivatives(
            stretch
        )

        fun = self._get_objective_function(residuals, stretch)

        tiled_res = np.tile(residuals, (1, self.n_components_))
        grad_flat = np.sum(d_stretch_comps * tiled_res, axis=0)
        gra = grad_flat.reshape(
            (self.n_signals_, self.n_components_), order="F"
        ).T
        gra += (
            self.rho
            * stretch
            @ (self._spline_smooth_operator.T @ self._spline_smooth_operator)
        )

        if self.uniform_stretch:
            # One shared variable controls every component for a given
            # signal, so combine the component-wise derivatives.
            gra = np.sum(gra, axis=0)

        return fun, gra

    def _regularize_function_hessian(self, stretch):
        """Calculate the Hessian for the stretch optimization objective.

        The Hessian combines the Gauss-Newton curvature from the stretched
        component derivatives, the residual-weighted second derivatives of
        those stretched components, and the quadratic smoothing penalty on
        neighboring stretch factors.

        Parameters
        ----------
        stretch : ndarray of shape (n_components, n_signals)
            Stretching factors at which to evaluate the objective curvature.

        Returns
        -------
        ndarray
            Symmetric Hessian matrix for the flattened stretch variables. In
            uniform mode, the shape is ``(n_signals, n_signals)``.
        """
        residuals, d_stretch_comps, dd_stretch_comps = (
            self._stretch_residual_and_derivatives(stretch)
        )
        n_variables = self.n_components_ * self.n_signals_
        hessian = np.zeros((n_variables, n_variables), dtype=float)

        for signal in range(self.n_signals_):
            variable_indices = (
                np.arange(self.n_components_) * self.n_signals_ + signal
            )
            d_signal = d_stretch_comps[:, variable_indices]
            dd_signal = dd_stretch_comps[:, variable_indices]
            hessian[np.ix_(variable_indices, variable_indices)] += (
                d_signal.T @ d_signal
            )
            hessian[variable_indices, variable_indices] += np.sum(
                dd_signal * residuals[:, signal, None],
                axis=0,
            )

        smooth_hessian = (
            self._spline_smooth_operator.T @ self._spline_smooth_operator
        ).toarray()
        for comp in range(self.n_components_):
            component_slice = slice(
                comp * self.n_signals_,
                (comp + 1) * self.n_signals_,
            )
            hessian[component_slice, component_slice] += (
                self.rho * smooth_hessian
            )

        hessian = 0.5 * (hessian + hessian.T)
        if self.uniform_stretch:
            # Map the full (component, signal) Hessian onto the independent
            # per-signal stretch variables.
            variable_map = np.zeros(
                (n_variables, self.n_signals_), dtype=float
            )
            for comp in range(self.n_components_):
                variable_map[
                    comp * self.n_signals_ : (comp + 1) * self.n_signals_,
                    :,
                ] = np.eye(self.n_signals_)
            hessian = variable_map.T @ hessian @ variable_map
        return hessian

    @staticmethod
    def _project_stretch(stretch, lower_bound=0.1):
        return np.maximum(stretch, lower_bound)

    def _initial_stretch_step_size(self, stretch, gradient):
        if self._stretch_step_size is not None:
            return self._stretch_step_size

        gradient_norm = np.linalg.norm(gradient)
        if gradient_norm == 0 or not np.isfinite(gradient_norm):
            return 1.0

        stretch_norm = max(np.linalg.norm(stretch), 1.0)
        return 0.05 * stretch_norm / gradient_norm

    def _update_stretch_trust_constr(self):
        """Updates stretching matrix using constrained optimization
        (equivalent to fmincon in MATLAB)."""
        if self.verbose:
            print("Updating stretch factors...")

        # Flatten stretch for compatibility with the optimizer
        # (since SciPy expects 1D input). Uniform stretching has one
        # independent variable per signal rather than one per component.
        stretch_flat_initial = self._stretch_variables().flatten()

        # Define the optimization function
        def objective(stretch_vec):
            fun, gra = self._regularize_function(stretch_vec)
            gra = gra.flatten()
            return fun, gra

        def hessian(stretch_vec):
            return self._regularize_function_hessian(stretch_vec)

        unconstrained_result = minimize(
            fun=lambda stretch_vec: objective(stretch_vec)[0],
            x0=stretch_flat_initial,
            method="trust-exact",
            jac=lambda stretch_vec: objective(stretch_vec)[1],
            hess=hessian,
            options={"maxiter": 300},
        )
        unconstrained_stretch = unconstrained_result.x
        if np.all(unconstrained_stretch >= 0.1):
            current_objective = self._regularize_function(self.stretch_)[0]
            candidate_objective = self._regularize_function(
                unconstrained_stretch
            )[0]
            if candidate_objective <= current_objective:
                self.stretch_ = self._expand_stretch(unconstrained_stretch)
                return

        # Optimization constraints: lower bound 0.1, no upper bound
        bounds = [
            (0.1, None)
        ] * stretch_flat_initial.size  # Equivalent to 0.1 * ones(K, M)

        # Solve optimization problem (equivalent to fmincon)
        result = minimize(
            fun=lambda stretch_vec: objective(stretch_vec)[0],
            x0=stretch_flat_initial,
            method="trust-constr",  # Substitute for 'trust-region-reflective'
            jac=lambda stretch_vec: objective(stretch_vec)[1],  # Gradient
            hess=hessian,
            bounds=bounds,
        )

        # Update stretch with the optimized values
        self.stretch_ = self._expand_stretch(result.x)
        self._stretch_step_size = None

    def _update_stretch_projected_gradient(self):
        if self.verbose:
            print("Updating stretch factors...")

        for _ in range(self.stretch_max_iter):
            stretch = self._stretch_variables()
            current_objective, gradient = self._regularize_function(stretch)
            step_size = self._initial_stretch_step_size(stretch, gradient)
            best_stretch = stretch
            best_objective = current_objective

            for _ in range(20):
                candidate_stretch = self._project_stretch(
                    stretch - step_size * gradient
                )
                step = candidate_stretch - stretch
                step_norm_sq = np.linalg.norm(step) ** 2
                if step_norm_sq == 0:
                    step_size *= 0.5
                    continue

                candidate_residuals = self._get_residual_matrix(
                    stretch=candidate_stretch
                )
                candidate_objective = self._get_objective_function(
                    residuals=candidate_residuals,
                    stretch=candidate_stretch,
                )
                if candidate_objective < best_objective:
                    best_stretch = candidate_stretch
                    best_objective = candidate_objective

                sufficient_decrease = (
                    current_objective - 1e-4 * step_norm_sq / step_size
                )
                if candidate_objective <= sufficient_decrease:
                    break

                step_size *= 0.5

            self._stretch_step_size = step_size
            if best_objective >= current_objective:
                break

            self.stretch_ = self._expand_stretch(best_stretch)

    def _update_stretch(self):
        """Update stretching factors with a hybrid strategy.

        The first ``stretch_slow_iter`` outer-loop updates use the original
        constrained nonlinear optimizer to find a good non-convex basin. Later
        updates switch to the lightweight Algorithm 2 style path: compute the
        stretch gradient, take linearized proximal steps, then project back
        onto the feasible stretch range.
        """
        if getattr(self, "_outer_iter", 0) < self.stretch_slow_iter:
            self._update_stretch_trust_constr()
        else:
            self._update_stretch_projected_gradient()

    @staticmethod
    def _compute_objective_function(
        components, residuals, stretch, rho, eta, spline_smooth_operator
    ):
        r"""Computes the objective function used in stretched non-
        negative matrix factorization.

        Parameters
        ----------
        components : ndarray
            Non-negative matrix of component signals :math:`X`.
        residuals : ndarray
            Difference between reconstructed and observed data.
        stretch : ndarray
            Stretching factors :math:`A` applied to each component across
            samples.
        rho : float
            Regularization parameter enforcing smooth variation in :math:`A`.
        eta : float
            Sparsity-promoting regularization parameter applied to :math:`X`.
        spline_smooth_operator : ndarray
            Linear operator :math:`L` penalizing non-smooth changes
            in :math:`A`.

        Returns
        -------
        float
            Value of the stretched-NMF objective function.

        Notes
        -----
        The stretched-NMF objective function :math:`J` is

        .. math::

           J(X, Y, A) =
              \tfrac{1}{2} \lVert Z - Y\,S(A)X \rVert_F^2
              + \tfrac{\rho}{2} \lVert L A \rVert_F^2
              + \eta \sum_{i,j} \sqrt{X_{ij}} \,,

        where :math:`Z` is the data matrix, :math:`Y` contains the non-negative
        weights, :math:`S(A)` denotes the spline-interp. stretching operator,
        and :math:`\lVert \cdot \rVert_F` is the Frobenius norm.

        Special cases
        -------------
        - :math:`\rho = 0` — no smoothness regularization on stretch factors.
        - :math:`\eta = 0` — no sparsity promotion on components.
        - :math:`\rho = \eta = 0` — reduces to the classical NMF least-squares
          objective :math:`\tfrac{1}{2} \lVert Z - YX \rVert_F^2`.
        """
        residual_term = 0.5 * np.linalg.norm(residuals, "fro") ** 2
        regularization_term = (
            0.5
            * rho
            * np.linalg.norm(spline_smooth_operator @ stretch.T, "fro") ** 2
        )
        sparsity_term = eta * np.sum(np.sqrt(components))
        return residual_term + regularization_term + sparsity_term


def _cubic_largest_real_root(p, q):
    """Solves x^3 + p*x + q = 0 element-wise for matrices, returning the
    largest real root."""
    # For q == 0, the non-negative solution is available directly.  Keep
    # this branch separate: the general complex-root calculation below is
    # numerically unstable at this degenerate cubic and previously overwrote
    # the exact result.
    q_is_zero = q == 0
    zero_q_root = np.maximum(0, -p) ** 0.5

    # Compute discriminant
    delta = (q / 2) ** 2 + (p / 3) ** 3

    # Match the MATLAB helper's real-root branch without evaluating invalid
    # real square roots for entries that are handled in complex arithmetic.
    d = np.sqrt(delta.astype(complex))

    # Compute cube roots safely
    a1 = (-q / 2 + d) ** (1 / 3)
    a2 = (-q / 2 - d) ** (1 / 3)

    # Compute cube roots of unity
    w = (np.sqrt(3) * 1j - 1) / 2

    # Compute the three possible roots (element-wise)
    y1 = a1 + a2
    y2 = w * a1 + w**2 * a2
    y3 = w**2 * a1 + w * a2

    # Take the largest real root element-wise when delta < 0
    r_roots = np.stack([np.real(y1), np.real(y2), np.real(y3)], axis=0)
    general_root = np.where(delta < 0, np.max(r_roots, axis=0), 0.0)

    return np.where(q_is_zero, zero_q_root, general_root)


def _reconstruct_matrix(
    components, weights, stretch, decay_rates=None, r=None
):
    """Construct the approximation of the source matrix corresponding to
    the given components, weights, and stretch factors.

    Each component profile is stretched, interpolated to fractional positions,
    weighted per signal, and summed to form the reconstruction.

    Parameters
    ----------
    components : (signal_len, n_components) array
    weights    : (n_components, n_signals) array
    stretch    : (n_components, n_signals) array
    decay_rates : (n_components, n_signals) array, optional
        Nonnegative decay rates. Omitted rates give the original model.
    r : (signal_len,) array, optional
        Coordinates for stretching and damping; defaults to sample indices.

    Returns
    -------
    reconstructed_matrix : (signal_len, n_signals) array
    """
    signal_len = components.shape[0]
    n_components = components.shape[1]
    n_signals = weights.shape[1]

    reconstructed_matrix = np.zeros((signal_len, n_signals))
    sample_indices = np.arange(signal_len) if r is None else np.asarray(r)

    for comp in range(n_components):  # loop over components
        contribution = (
            np.interp(
                sample_indices[:, None]
                / stretch[comp][
                    None, :
                ],  # fractional positions (signal_len, n_signals)
                sample_indices,  # (signal_len,)
                components[:, comp],  # component profile (signal_len,)
                left=components[0, comp],
                right=components[-1, comp],
            )
            * weights[comp][None, :]  # broadcast (n_signals,) over rows
        )
        if decay_rates is not None:
            contribution *= np.exp(
                -sample_indices[:, None] * decay_rates[comp][None, :]
            )
        reconstructed_matrix += contribution

    return reconstructed_matrix
