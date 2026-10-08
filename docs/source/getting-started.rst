.. _getting-started:

Getting started
===============

``diffpy.stretched-nmf`` implements the stretched NMF algorithm for factorizing signal
sets while accounting for uniform stretching along the independent axis.

Installation
------------

The preferred method is conda:

.. code-block:: bash

   conda config --add channels conda-forge
   conda create -n diffpy.stretched-nmf_env diffpy.stretched-nmf
   conda activate diffpy.stretched-nmf_env

For interactive plotting with ``show_plots=True``, use a GUI-capable desktop
environment. Conda installs use ``matplotlib-base``, which is sufficient for
plotting but still depends on an available interactive Matplotlib backend.

Alternatively, install from PyPI with pip:

.. code-block:: bash

   pip install diffpy.stretched-nmf

For source installs (after cloning the repo):

.. code-block:: bash

   pip install .

Quick check
-----------

Verify the CLI and Python import:

.. code-block:: bash

   snmf --version
   python -c "import diffpy.stretched_nmf; print(diffpy.stretched_nmf.__version__)"

Basic usage
-----------

The main entry point is the ``SNMFOptimizer`` class. Create an
``SNMFOptimizer`` with hyperparameters, then pass the source matrix to
``fit``. The source matrix shape is
``(length_of_signal, number_of_signals)``.

.. code-block:: python

   import numpy as np
   from diffpy.stretched_nmf.snmf_class import SNMFOptimizer

   rng = np.random.default_rng(7)
   source_matrix = rng.random((300, 24))  # (signal_length, n_signals)

   snmf = SNMFOptimizer(
       n_components=3,
       max_iter=400,
       min_iter=20,
       tol=5e-7,
       rho=0,
       eta=0,
       random_state=7,
       show_plots=False,
       uniform_stretch=False,
   )

   snmf.fit(source_matrix=source_matrix, reset=True)

   components = snmf.components_
   weights = snmf.weights_
   stretch = snmf.stretch_

Notes
-----

- ``rho`` controls the stretching penalty (set to ``0`` for no stretching).
- ``eta`` controls sparsity (start at ``0`` and tune after selecting ``rho``).
- Set ``uniform_stretch=True`` to share one stretch factor per signal across
  all components.
- Use ``reset=False`` only when you want to continue from the current solution.

Optional exponential damping
----------------------------

Damping is off by default. Select the components to fit by their zero-based
indices, for example ``damping_components=[0, 2]`` to fit rates for the first
and third components. A selected component contributes

.. math::

   y_k^m x_k(r / a_k^m) \exp(-\lambda_k^m r).

Each selected component has a separate nonnegative rate for each signal
(column) ``m``. Components excluded from damping keep rates exactly zero.
The rate is :math:`\lambda = 1 / \xi`, in inverse units of ``r``. A rate
of zero represents the original undamped model exactly, with infinite decay
length. Damping can be fitted with ``rho=0`` (no stretching) or combined with
component-specific or uniform stretching.

.. code-block:: python

   r = np.linspace(0.0, 30.0, source_matrix.shape[0])
   snmf = SNMFOptimizer(
       n_components=3,
       damping_components=[0, 2],
       damping_regularization=1.0,
       random_state=7,
   )
   snmf.fit(source_matrix, r=r)
   rates = snmf.decay_rates_  # shape (3, number_of_signals)
   # rates[1, :] is exactly zero: component 1 is excluded.
   decay_lengths = np.full_like(rates, np.inf)
   np.divide(1.0, rates, out=decay_lengths, where=rates > 0)

``r`` must be a finite, nonnegative, strictly increasing array with one
coordinate per source row. The profiles are interpolated at ``r / a`` on
this grid, while damping is evaluated at the observed ``r``, after stretching.
Without ``r``, sample indices are used and rates have inverse sample units.
``init_decay_rates`` can provide an initial matrix of the same shape as the
weights; it defaults to zero and must be zero for excluded components.

``damping_regularization`` is independent of ``rho`` and ``eta`` and adds

.. math::

   \frac{\text{damping_regularization}}{2}
   \sum_k \sum_{m=0}^{M-3}
   (\lambda_k^{m+2} - 2\lambda_k^{m+1} + \lambda_k^m)^2.

It follows the order of the source columns, assuming equally spaced signals.
Linear rate trends have zero penalty. With fewer than three signals, the
penalty is zero. A weight of zero disables only this penalty; selected rates
are still fitted. ``None`` or ``[]`` for ``damping_components`` disables
damping entirely.

Rates retain their physical units through result normalization. Warm starts
(``reset=False``) reuse the fitted rates and grid; an explicitly supplied
``r`` must match the existing grid.

When profiles are free, a common decay envelope can be absorbed into a
component profile. For example, without stretching, replacing ``x(r)`` with
``x(r) * exp(-c * r)`` and every rate with ``lambda - c`` gives the same
reconstruction whenever the new rates remain nonnegative. Absolute decay
lengths therefore require an undamped reference signal or prior information
about the component profiles. Initialization and regularization can influence
which solution the optimizer finds.

XRD example
-----------

A complete real-data example is included in
``docs/examples/XRD_MgMnO_YCl_real.py``. It reads the input matrices from
``docs/examples/data/XRD_MgMnO_YCl_real``.

Run it from the repository root:

.. code-block:: bash

   python docs/examples/XRD_MgMnO_YCl_real.py

The script runs a tuned XRD fit and writes
``my_norm_components.txt``, ``my_norm_weights.txt``, and
``my_norm_stretch.txt`` to the current working directory.

.. figure:: img/mid-optimization.png
   :alt: Preview from the show_plots interface before convergence
   :align: center
   :width: 85%

   Preview from the ``show_plots`` interface from before the optimization has settled down.

.. figure:: img/xrd-final-outputs.png
   :alt: Final normalized outputs for components, weights, and stretch
   :align: center
   :width: 95%

   Final outputs: normalized components, normalized weights, and normalized stretch shown side by side.

Next steps
----------

Browse the rest of the docs for release notes and license information.
