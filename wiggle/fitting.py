"""
fitting.py
~~~~~~~~~~
Chi-squared fitting and scale-factor utilities for comparing computed
I(q) curves to experimental SAXS data.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def interpolate_to_q_grid(
    q_model: np.ndarray,
    I_model: np.ndarray,
    q_target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linearly interpolate model I(q) onto the experimental q grid.

    Only returns points within the overlapping q range to avoid extrapolation.

    Returns
    -------
    q_overlap, I_interp, mask
        q_overlap  – experimental q values within the model range
        I_interp   – interpolated model intensities at those q values
        mask       – boolean mask into the original q_target array
    """
    q_min = max(q_model[0],  q_target[0])
    q_max = min(q_model[-1], q_target[-1])

    mask = (q_target >= q_min) & (q_target <= q_max)
    q_overlap = q_target[mask]
    I_interp = np.interp(q_overlap, q_model, I_model)

    return q_overlap, I_interp, mask


def initial_scale(I_exp: np.ndarray, I_model: np.ndarray) -> float:
    """Rough initial scale factor as ratio of means."""
    return float(np.mean(I_exp) / np.mean(I_model))


def optimise_scale(
    I_exp: np.ndarray,
    I_model: np.ndarray,
    I_err: np.ndarray,
) -> tuple[float, float]:
    """Find scale c minimising chi2 = mean(((I_exp - c*I_model) / sigma)**2).

    Returns
    -------
    optimal_scale, chi2
    """
    def chi2(scale: float) -> float:
        return float(np.mean(((I_exp - scale * I_model) / I_err) ** 2))

    result = minimize_scalar(chi2)
    return float(result.x), float(result.fun)


def fit_model_to_data(
    q_model: np.ndarray,
    I_model: np.ndarray,
    q_exp: np.ndarray,
    I_exp: np.ndarray,
    I_err: np.ndarray,
) -> dict:
    """Interpolate model onto experimental grid and optimise scale factor.

    Returns
    -------
    dict with keys: q, I_scaled, scale, chi2
        q and I_scaled are restricted to the overlapping q range.
    """
    q_overlap, I_interp, mask = interpolate_to_q_grid(q_model, I_model, q_exp)

    I_exp_overlap  = I_exp[mask]
    I_err_overlap  = I_err[mask]

    scale, chi2 = optimise_scale(I_exp_overlap, I_interp, I_err_overlap)

    return {
        "q":        q_overlap,
        "I_scaled": scale * I_interp,
        "scale":    scale,
        "chi2":     chi2,
    }
