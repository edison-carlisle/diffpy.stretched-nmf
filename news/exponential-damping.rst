**Added:**

* Optional exponential damping for selected components in ``SNMFOptimizer``,
  with nonnegative per-signal ``decay_rates_`` and an independent
  ``damping_regularization`` weight for second differences across signals.
* Optional ``r`` coordinates and ``init_decay_rates`` in ``fit``. Zero rates
  represent infinite decay lengths exactly; damping remains off by default.
