"""
cli.py
~~~~~~
Command-line interface for wiggle.

Usage
-----
    wiggle --pdb my_model.pdb --saxs my_data.dat
    wiggle --pdb my_model.pdb --saxs my_data.dat --out fit.dat
    wiggle --pdb my_model.pdb --saxs my_data.dat --ff custom_ff.pkl
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from wiggle.api import compute_fit
from wiggle.fitting import fit_model_to_data
from wiggle.scattering import load_form_factors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wiggle",
        description="Compute and fit an implicit SAXS curve from a PDB structure.",
    )
    parser.add_argument(
        "--pdb", required=True, metavar="FILE",
        help="Path to the input PDB file.",
    )
    parser.add_argument(
        "--saxs", required=True, metavar="FILE",
        help="Path to the SAXS data file (q  I  sigma, # comments ok).",
    )
    parser.add_argument(
        "--ff", default=None, metavar="FILE",
        help="Path to a custom form-factor .pkl file. "
             "Defaults to the bundled wiggle form factors.",
    )
    parser.add_argument(
        "--out", default="wiggle_fit.dat", metavar="FILE",
        help="Path to save the output data file (default: wiggle_fit.dat).",
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    ff = load_form_factors(args.ff) if args.ff else None

    print(f"Processing {args.pdb} ...")
    result = compute_fit(args.pdb, args.saxs, form_factors=ff)

    fit = fit_model_to_data(
        result["q_model"], result["I_model"],
        result["q_exp"],   result["I_exp"], result["I_err"],
    )

    print(f"chi2  = {fit['chi2']:.4f}")
    print(f"scale = {fit['scale']:.4e}")
    print(f"Writing output to {args.out}")

    # Interpolate scaled model back onto the full experimental q grid
    I_model_out = np.interp(
        result["q_exp"], fit["q"], fit["I_scaled"],
        left=np.nan, right=np.nan,
    )

    header = (
        f"wiggle fit\n"
        f"pdb:   {args.pdb}\n"
        f"saxs:  {args.saxs}\n"
        f"chi2:  {fit['chi2']:.4f}\n"
        f"scale: {fit['scale']:.6e}\n"
        f"q_exp  I_exp  I_err  I_model"
    )
    out_data = np.column_stack([
        result["q_exp"],
        result["I_exp"],
        result["I_err"],
        I_model_out,
    ])
    np.savetxt(args.out, out_data, header=header, fmt="%.8e")

    return 0


if __name__ == "__main__":
    sys.exit(main())
