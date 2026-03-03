"""
scattering.py
~~~~~~~~~~~~~
SAXS intensity calculation using implicit hydrated form factors.
"""

from __future__ import annotations

import pickle
from importlib import resources
from pathlib import Path
from typing import Union

import numpy as np

_DEFAULT_FF_NAME = "wiggle_implicit_form_factors.pkl"


def _default_form_factors_path() -> Path:
    """Return the path to the bundled default form factors file."""
    pkg_files = resources.files("wiggle.data")
    return Path(str(pkg_files.joinpath(_DEFAULT_FF_NAME)))


def load_form_factors(path: Union[str, Path, None] = None) -> dict[str, np.ndarray]:
    """Load a pickled form-factor dictionary.

    Parameters
    ----------
    path:
        Path to a ``.pkl`` file.  If ``None`` (default), the form factors
        bundled with the package are loaded automatically.

    Returns
    -------
    dict[str, np.ndarray]
    """
    if path is None:
        path = _default_form_factors_path()
    with open(path, "rb") as fh:
        ff = pickle.load(fh)
    if not isinstance(ff, dict):
        raise TypeError(f"Expected a dict, got {type(ff).__name__} in {path}")
    return ff


def calculate_saxs(
    preprocessed_data: dict,
    form_factors: Union[dict[str, np.ndarray], None] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute isotropic SAXS intensity I(q) using the Debye formula.

    Parameters
    ----------
    preprocessed_data:
        Output dict from :meth:`~wiggle.structure.StructureProcessor.process`.
    form_factors:
        Dict mapping centre-type labels to f(q) arrays.  If ``None``
        (default), the bundled form factors are loaded automatically.

    Returns
    -------
    q : np.ndarray
        q values (Angstrom^-1).
    I_q : np.ndarray
        Scattering intensity at each q.
    """
    if form_factors is None:
        form_factors = load_form_factors()   
        
    centre_types = preprocessed_data['centre_types']
    r_ij = preprocessed_data['r_ij']
    
    # Get the q range from one of the form factors (they should all have the same q)
    sample_ff = next(iter(form_factors.values()))
    N = len(sample_ff)
    q = np.linspace(0, (N-1)/100, N)
    
    # Stack form factors - ensure they all have the same length first
    form_factors_list = []
    for ct in centre_types:
        ff = form_factors[ct]
        if len(ff) != N:
            raise ValueError(f"Form factor for {ct} has length {len(ff)}, expected {N}")
        form_factors_list.append(ff)
    
    form_factors = np.stack(form_factors_list, axis=0)  # shape: NxQ
    ff_products = form_factors @ form_factors.T
    
    # Expand q and r_ij for broadcasting
    q_expanded = q[:, np.newaxis, np.newaxis]  # shape: Qx1x1
    r_ij_expanded = r_ij[np.newaxis, :, :]  # shape: 1xNxN
    
    # Calculate sinc terms
    # np.sinc(x) = sin(πx)/(πx), which is what we want with x = qr/π
    sinc_terms = np.sinc(q_expanded * r_ij_expanded / np.pi)  # shape: QxNxN
    
    # Calculate intensity by summing over all pairs
    I_q = np.sum(ff_products * sinc_terms, axis=(1,2))
    
    return q, I_q
