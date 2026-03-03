"""
plotting.py
~~~~~~~~~~~
Matplotlib helpers for visualising SAXS fits.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess


def plot_fit(
    q_exp,
    I_exp,
    I_err,
    q_model,
    I_model_scaled,
    chi2,
    label="Model",
    title="SAXS fit",
    save_path=None,
):
    """Plot experimental data and a scaled model curve with residuals.

    Parameters
    ----------
    q_exp, I_exp, I_err:
        Experimental q, I(q), and uncertainties.
    q_model, I_model_scaled:
        Model q values and scaled intensities (already on the experimental grid
        or a finer one -- they will be plotted as-is).
    chi2:
        Reduced chi-squared value shown in the legend.
    label:
        Legend label for the model curve.
    title:
        Figure title.
    save_path:
        If given, save the figure to this path (SVG or PNG).

    Returns
    -------
    fig, (ax_main, ax_res)
    """
    fig = plt.figure(figsize=(10, 8), dpi=150)
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax_main = fig.add_subplot(gs[0])
    ax_res = fig.add_subplot(gs[1], sharex=ax_main)

    # --- Main panel ---
    ax_main.errorbar(
        q_exp, I_exp, yerr=I_err,
        fmt="o", markersize=4, color="#4A4A4A", ecolor="#CCCCCC",
        alpha=0.6, capsize=0, linewidth=1, elinewidth=1,
        label="Experimental", zorder=1,
    )
    ax_main.plot(
        q_model, I_model_scaled, "--",
        linewidth=2.5, color="black", alpha=0.8,
        label=f"{label} (chi2={chi2:.2f})", zorder=2,
    )
    ax_main.set_ylabel("I(q)")
    ax_main.set_yscale("log")
    ax_main.legend(frameon=False, loc="lower left")
    ax_main.grid(True, alpha=0.3, linestyle=":", linewidth=0.8)
    ax_main.set_title(title)
    for sp in ["top", "right", "bottom"]:
        ax_main.spines[sp].set_visible(False)
    ax_main.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    # --- Residuals panel ---
    residuals = (I_exp - I_model_scaled) / I_err
    smoothed = lowess(residuals, q_exp, frac=0.2)
    ax_res.scatter(q_exp, residuals, marker="o", s=6, alpha=0.3, color="black")
    ax_res.plot(
        smoothed[:, 0], smoothed[:, 1], "--",
        color="black", alpha=0.8, linewidth=2.5,
    )
    ax_res.axhline(0, color="black", linestyle="--", alpha=0.3, linewidth=1.5)
    ax_res.set_xlabel("q (A^-1)")
    ax_res.set_ylabel("delta / sigma")
    ax_res.grid(True, alpha=0.3, linestyle=":", linewidth=0.8)
    for sp in ["top", "right"]:
        ax_res.spines[sp].set_visible(False)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    return fig, (ax_main, ax_res)
