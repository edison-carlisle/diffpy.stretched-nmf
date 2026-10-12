import matplotlib.pyplot as plt
import numpy as np


class SNMFPlotter:
    """Display component profiles, weights, stretch factors, and decay
    rates."""

    def __init__(self, figsize=(16, 4)):
        plt.ion()
        self.fig, self.axes = plt.subplots(1, 4, figsize=figsize)
        titles = [
            "Components",
            "Weights (rows as series)",
            "Stretch (rows as series)",
            "Decay rates (rows as series)",
        ]
        for ax, t in zip(self.axes, titles):
            ax.set_title(t)
        self.axes[3].set_xlabel("Signal index")
        self.axes[3].set_ylabel(r"$\lambda$ (inverse $r$ units)")
        self.lines = {
            "components": [],
            "weights": [],
            "stretch": [],
            "decay_rates": [],
        }
        self._layout_done = False
        plt.show()

    def _ensure_lines(self, ax, key, n_series):
        cur = self.lines[key]
        if len(cur) != n_series:
            title = ax.get_title()
            xlabel = ax.get_xlabel()
            ylabel = ax.get_ylabel()
            ax.cla()
            ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
            self.lines[key] = [ax.plot([], [])[0] for _ in range(n_series)]
        return self.lines[key]

    def _update_series(self, ax, key, data_2d):
        # Expect rows = separate series for components
        data_2d = np.atleast_2d(data_2d)
        n_series, n_pts = data_2d.shape
        lines = self._ensure_lines(ax, key, n_series)
        x = np.arange(n_pts)
        for ln, y in zip(lines, data_2d):
            ln.set_data(x, y)
        ax.relim()
        ax.autoscale_view()

    def update(
        self, components, weights, stretch, update_tag=None, decay_rates=None
    ):
        """Refresh all panels with the current factors.

        ``decay_rates`` has shape ``(n_components, n_signals)`` and is
        expressed in inverse units of the fitted coordinate grid.
        Omitted rates are plotted as zero for compatibility with
        undamped calls.
        """
        # Components: transpose before plotting
        c = np.asarray(components).T
        self._update_series(self.axes[0], "components", c)

        w = np.asarray(weights)
        self._update_series(self.axes[1], "weights", w)

        s = np.asarray(stretch)
        self._update_series(self.axes[2], "stretch", s)

        rates = (
            np.zeros_like(w)
            if decay_rates is None
            else np.asarray(decay_rates)
        )
        self._update_series(self.axes[3], "decay_rates", rates)

        if update_tag is not None:
            self.fig.suptitle(f"Updated: {update_tag}", fontsize=14)

        if not self._layout_done:
            self.fig.tight_layout()
            self._layout_done = True

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
