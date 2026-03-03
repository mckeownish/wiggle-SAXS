"""
structure.py
~~~~~~~~~~~~
Protein structure parsing and geometric scattering centre placement.
"""

from __future__ import annotations

from string import ascii_uppercase

import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb
from scipy.spatial.distance import pdist, squareform


# Residue COM distances from CA atom (Å), used to place implicit side-chain centres.
# GLY and ALA have no side-chain centre — they fall back to the CA position.
_RESIDUE_COM_DISTANCES: dict[str, float] = {
    "ARG": 4.2662, "ASN": 2.5349, "ASP": 2.5558,
    "CYS": 2.3839, "GLN": 3.1861, "GLU": 3.2541,
    "HIS": 3.1861, "ILE": 2.3115, "LEU": 2.6183,
    "LYS": 3.6349, "MET": 3.1912, "PHE": 3.4033,
    "PRO": 1.8773, "SEC": 1.5419, "SER": 1.9661,
    "THR": 1.9533, "TRP": 3.8916, "TYR": 3.8807,
    "VAL": 1.9555,
}

# Residues that use their CA as the sole scattering centre
_CA_ONLY_RESIDUES = {"GLY", "ALA"}

# Chain label pool used when re-labelling chains
_CHAIN_LABELS = list(ascii_uppercase) + [
    f"{a}{b}" for a in ascii_uppercase for b in ascii_uppercase
]


_ONE_TO_THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
    "U": "SEC",
}

# Proline correction angle (degrees) applied to the backbone normal
_PRO_CORRECTION_ANGLE_DEG = 47.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class StructureProcessor:
    """
    Parse a PDB file and build implicit scattering centres for SAXS modelling.

    Typical usage::

        proc = StructureProcessor()
        data = proc.process("my_protein.pdb")
        # data["centre_types"], data["geo_centre_coordinates"], data["r_ij"], …
    """

    def process(self, pdb_file: str) -> dict:
        """
        Full pipeline: parse → fix chains → place centres → compute distances.

        Parameters
        ----------
        pdb_file:
            Path to a PDB file.

        Returns
        -------
        dict with keys:
            ``centre_types``           – residue labels for each scattering centre
            ``chain_ids``              – chain label for each centre
            ``geo_centre_coordinates`` – (N, 3) float array of centre positions
            ``r_ij``                   – (N, N) pairwise distance matrix
        """
        ca_df = self._extract_ca(pdb_file)
        ca_df = self._fix_chain_breaks(ca_df)

        ca_coords = ca_df[["x_coord", "y_coord", "z_coord"]].values
        residue_names = ca_df["residue_name"].values
        chain_ids = ca_df["chain_id"].values

        geometric_vectors = self._geometric_vectors(ca_coords, residue_names, chain_ids)
        side_chain_positions = self._place_side_chains(ca_coords, geometric_vectors, residue_names)

        centre_types, geo_coords, chains = self._build_centre_arrays(
            ca_df, ca_coords, side_chain_positions, chain_ids
        )

        r_ij = squareform(pdist(geo_coords))

        return {
            "centre_types": centre_types,
            "chain_ids": chains,
            "geo_centre_coordinates": geo_coords,
            "r_ij": r_ij,
        }


    def process_from_arrays(self, ca_coords, sequence, chain_ids=None):
        """Calculate scattering centres from CA coordinates and sequence.

        Bypasses PDB parsing -- useful for MD trajectories or any source that
        already provides coordinates as numpy arrays.

        Parameters
        ----------
        ca_coords : np.ndarray
            (N, 3) float array of CA coordinates in Angstroms.
        sequence : str or list
            One-letter string (e.g. "ACDE...") or list of three-letter codes.
        chain_ids : np.ndarray, optional
            (N,) array of chain labels. Defaults to all chain "A".

        Returns
        -------
        Same dict as process(): centre_types, chain_ids, geo_centre_coordinates, r_ij
        """
        ca_coords = np.asarray(ca_coords, dtype=float)
        if ca_coords.ndim != 2 or ca_coords.shape[1] != 3:
            raise ValueError(f"ca_coords must be shape (N, 3), got {ca_coords.shape}")
        N = len(ca_coords)

        if isinstance(sequence, str):
            residue_names = np.array([_ONE_TO_THREE.get(aa.upper(), "GLY") for aa in sequence])
        else:
            residue_names = np.array([r.upper() for r in sequence])

        if len(residue_names) != N:
            raise ValueError(f"sequence length ({len(residue_names)}) != ca_coords length ({N})")

        if chain_ids is None:
            chain_ids = np.full(N, "A")
        else:
            chain_ids = np.asarray(chain_ids)

        geometric_vectors = self._geometric_vectors(ca_coords, residue_names, chain_ids)
        side_chain_positions = self._place_side_chains(ca_coords, geometric_vectors, residue_names)

        ca_df = pd.DataFrame({"residue_name": residue_names})
        centre_types, geo_coords, chains = self._build_centre_arrays(
            ca_df, ca_coords, side_chain_positions, chain_ids
        )

        r_ij = squareform(pdist(geo_coords))

        return {
            "centre_types": centre_types,
            "chain_ids": chains,
            "geo_centre_coordinates": geo_coords,
            "r_ij": r_ij,
        }
    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_ca(pdb_file: str) -> pd.DataFrame:
        """Return a DataFrame of CA atoms, one row per residue."""
        ppdb = PandasPdb().read_pdb(pdb_file)
        atom_df = ppdb.df["ATOM"]
        ca_df = atom_df[
            (atom_df["atom_name"] == "CA")
            & (atom_df["alt_loc"].isin(["", "A"]))
        ].copy()
        ca_df.reset_index(drop=True, inplace=True)
        return ca_df

    @staticmethod
    def _fix_chain_breaks(ca_df: pd.DataFrame, max_gap: int = 10) -> pd.DataFrame:
        """
        Re-label chains so that sequence breaks (gap > *max_gap* residues)
        and explicit chain changes each start a new chain letter.
        """
        ca_df = ca_df.copy()
        new_chain_ids: list[str] = []
        chain_idx = 0
        prev_chain: str | None = None
        prev_resnum: int | None = None
        current_letter: str = _CHAIN_LABELS[0]

        for _, row in ca_df.iterrows():
            chain, resnum = row["chain_id"], row["residue_number"]
            new_chain = (
                prev_chain is None
                or chain != prev_chain
                or (prev_resnum is not None and abs(resnum - prev_resnum) > max_gap)
            )
            if new_chain:
                current_letter = _CHAIN_LABELS[chain_idx]
                chain_idx += 1
            new_chain_ids.append(current_letter)
            prev_chain, prev_resnum = chain, resnum

        ca_df["chain_id"] = new_chain_ids
        return ca_df

    def _geometric_vectors(
        self,
        ca_coords: np.ndarray,
        residue_names: np.ndarray,
        chain_ids: np.ndarray,
    ) -> np.ndarray:
        """Compute outward backbone normals, handling multi-chain structures."""
        unique_chains = list(dict.fromkeys(chain_ids))
        all_vectors = np.zeros_like(ca_coords)
        for chain in unique_chains:
            mask = chain_ids == chain
            all_vectors[mask] = self._geometric_vectors_single_chain(
                ca_coords[mask], residue_names[mask]
            )
        return all_vectors

    @staticmethod
    def _geometric_vectors_single_chain(
        ca_coords: np.ndarray,
        residue_names: np.ndarray,
    ) -> np.ndarray:
        """
        Backbone normals for a single contiguous chain.
        Proline residues receive a rigid-body rotation correction.
        """
        ca_vectors = np.diff(ca_coords, axis=0)
        directions = ca_vectors / np.linalg.norm(ca_vectors, axis=1, keepdims=True)

        raw_normals = np.diff(ca_vectors, axis=0)
        norms = np.linalg.norm(raw_normals, axis=1, keepdims=True)
        normals = raw_normals / norms

        # Proline correction -----------------------------------------------
        pro_mask = residue_names[1:-1] == "PRO"
        if np.any(pro_mask):
            angle = np.radians(_PRO_CORRECTION_ANGLE_DEG)
            cos_t, sin_t = np.cos(angle), np.sin(angle)

            axes = np.cross(normals, directions[:-1])
            axes /= np.linalg.norm(axes, axis=1, keepdims=True)

            dot = np.sum(axes[pro_mask] * normals[pro_mask], axis=1, keepdims=True)
            normals[pro_mask] = (
                normals[pro_mask] * cos_t
                + np.cross(axes[pro_mask], normals[pro_mask]) * sin_t
                + axes[pro_mask] * dot * (1.0 - cos_t)
            )

        first = ca_vectors[0] / np.linalg.norm(ca_vectors[0])
        last = -ca_vectors[-1] / np.linalg.norm(ca_vectors[-1])
        normals = np.vstack([first, normals, last])

        return -normals  # negate → outward-pointing normals

    @staticmethod
    def _place_side_chains(
        ca_coords: np.ndarray,
        geometric_vectors: np.ndarray,
        residue_names: np.ndarray,
    ) -> np.ndarray:
        """Displace each CA along its geometric vector by the residue COM distance."""
        distances = np.array(
            [_RESIDUE_COM_DISTANCES.get(res, np.nan) for res in residue_names]
        )
        return ca_coords + distances[:, np.newaxis] * geometric_vectors

    @staticmethod
    def _build_centre_arrays(
        ca_df: pd.DataFrame,
        ca_coords: np.ndarray,
        side_chain_positions: np.ndarray,
        chain_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Combine backbone and side-chain centres into flat arrays.

        - Non-GLY/ALA residues → one BB centre (at CA) + one SC centre (at COM).
        - GLY / ALA              → a single labelled centre at the CA position.
        """
        residue_names = ca_df["residue_name"].values
        sc_mask = ~np.isin(residue_names, list(_CA_ONLY_RESIDUES))
        ca_only_mask = ~sc_mask

        centre_types = np.concatenate([
            np.full(sc_mask.sum(), "BB"),          # backbone centres for SC residues
            residue_names[sc_mask],                # side-chain labels
            residue_names[ca_only_mask],           # GLY / ALA centres
        ])
        geo_coords = np.concatenate([
            ca_coords[sc_mask],
            side_chain_positions[sc_mask],
            ca_coords[ca_only_mask],
        ])
        chains = np.concatenate([
            chain_ids[sc_mask],
            chain_ids[sc_mask],
            chain_ids[ca_only_mask],
        ])

        return centre_types, geo_coords, chains
