"""
wiggle
~~~~~~
Implicit SAXS modelling from protein structures.
"""

from wiggle.structure import StructureProcessor
from wiggle.scattering import calculate_saxs, load_form_factors
from wiggle.fitting import fit_model_to_data, optimise_scale, interpolate_to_q_grid
from wiggle.plotting import plot_fit
from wiggle.api import compute_fit

__all__ = [
    "StructureProcessor",
    "calculate_saxs",
    "load_form_factors",
    "fit_model_to_data",
    "optimise_scale",
    "interpolate_to_q_grid",
    "plot_fit",
    "compute_fit",
]

__version__ = "0.1.0"