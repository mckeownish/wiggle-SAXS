import numpy as np
import pandas as pd
import pytest
from wiggle.structure import StructureProcessor


def test_fix_chain_breaks_splits_on_gap():
    proc = StructureProcessor()
    df = pd.DataFrame({
        "chain_id": ["A", "A", "A", "A"],
        "residue_number": [1, 2, 50, 51],
    })
    result = proc._fix_chain_breaks(df, max_gap=10)
    assert result["chain_id"].iloc[0] == result["chain_id"].iloc[1]
    assert result["chain_id"].iloc[0] != result["chain_id"].iloc[2]
    assert result["chain_id"].iloc[2] == result["chain_id"].iloc[3]


def test_fix_chain_breaks_new_chain_letter():
    proc = StructureProcessor()
    df = pd.DataFrame({
        "chain_id": ["A", "A"],
        "residue_number": [1, 100],
    })
    result = proc._fix_chain_breaks(df, max_gap=10)
    assert result["chain_id"].iloc[0] != result["chain_id"].iloc[1]


def test_geometric_vectors_shape():
    proc = StructureProcessor()
    rng = np.random.default_rng(42)
    ca = rng.standard_normal((10, 3))
    res = np.array(["ALA"] * 10)
    vecs = proc._geometric_vectors_single_chain(ca, res)
    assert vecs.shape == ca.shape


def test_geometric_vectors_normalised():
    proc = StructureProcessor()
    rng = np.random.default_rng(0)
    ca = rng.standard_normal((8, 3))
    res = np.array(["GLY"] * 8)
    vecs = proc._geometric_vectors_single_chain(ca, res)
    norms = np.linalg.norm(vecs, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-10)


def test_place_side_chains_shape():
    proc = StructureProcessor()
    rng = np.random.default_rng(1)
    ca = rng.standard_normal((5, 3))
    vecs = np.tile([1.0, 0.0, 0.0], (5, 1))
    res = np.array(["ARG", "GLY", "ALA", "LEU", "VAL"])
    sc = proc._place_side_chains(ca, vecs, res)
    assert sc.shape == ca.shape
