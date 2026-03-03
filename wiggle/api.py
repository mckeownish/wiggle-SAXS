"""
api.py
~~~~~~
High-level convenience functions for common single-step workflows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np

from wiggle.structure import StructureProcessor
from wiggle.scattering import calculate_saxs


def compute_fit(
    pdb_file: Union[str, Path],
    saxs_file: Union[str, Path],
    form_factors=None,
) -> dict:
    """Load a PDB and a SAXS data file and return the computed scattering.

    Parameters
    ----------
    pdb_file:
        Path to a PDB structure file.
    saxs_file:
        Path to a three-column SAXS data file: q  I  sigma.
        Lines starting with ``#`` are treated as comments.
    form_factors:
        Optional pre-loaded form-factor dict.  If ``None`` the bundled
        default form factors are used.

    Returns
    -------
    dict with keys:
        ``q_exp``    – experimental q values
        ``I_exp``    – experimental intensities
        ``I_err``    – experimental uncertainties
        ``q_model``  – model q values
        ``I_model``  – raw (unscaled) model intensities
    """
    data = np.loadtxt(saxs_file, comments="#")
    if data.shape[1] < 3:
        raise ValueError(f"{saxs_file} must have at least three columns: q  I  sigma")
    q_exp, I_exp, I_err = data[:, 0], data[:, 1], data[:, 2]
    
    if q_exp.max() > 1:
        print('assumed nm units')
        q_exp = q_exp/10
    
    cond = q_exp <= 0.2
    q_exp = q_exp[cond]
    I_exp = I_exp[cond]
    I_err = I_err[cond]

    proc = StructureProcessor()
    structure = proc.process(str(pdb_file))
    q_model, I_model = calculate_saxs(structure, form_factors)

    return {
        "q_exp": q_exp,
        "I_exp": I_exp,
        "I_err": I_err,
        "q_model": q_model,
        "I_model": I_model,
    }
