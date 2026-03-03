# wiggle-SAXS

**SAXS modelling of coarse-grain protein structures using implicitly hydrated form factors.**

wiggle-SAXS computes theoretical SAXS curves directly from CA position in PDB files or numpy arrays, using a two body model approximating side chain centre of masses from CAs and pre-fitted implicitly hydrated form factors.

---

## Installation

```bash
pip install wiggle-SAXS
```

Or for development (editable install with dev extras):

```bash
git clone https://github.com/mckeownish/wiggle-SAXS.git
cd wiggle-SAXS
pip install -e ".[dev]"
```

---


## Quick start

### From a PDB file

```python
import wiggle

# compute model curve from PDB + experimental SAXS data
result = wiggle.compute_fit("data/lysozyme.pdb", "data/lyso_SAXS.dat")

# fit to data
fit = wiggle.fit_model_to_data(
    result["q_model"], result["I_model"],
    result["q_exp"],   result["I_exp"], result["I_err"],
)
print(f"chi2 = {fit['chi2']:.3f}")

# plot
wiggle.plot_fit(
    result["q_exp"], result["I_exp"], result["I_err"],
    fit["q"], fit["I_scaled"],
    chi2=fit["chi2"],
    title="Lysozyme SAXS Fit",
    save_path="fit.svg",
)
```

### From numpy arrays (e.g. MD trajectories)

```python
import wiggle

proc = wiggle.StructureProcessor()

# sequence accepts one-letter string or list of three-letter codes
structure = proc.process_from_arrays(ca_coords, sequence)

# multi-chain example — pass chain_ids matching length of ca_coords
# chain_ids = np.array(["A"] * 129 + ["B"] * 129)
# structure = proc.process_from_arrays(ca_coords, sequence, chain_ids=chain_ids)

q, I_q = wiggle.calculate_saxs(structure)

fit = wiggle.fit_model_to_data(q, I_q, q_exp, I_exp, I_err)
print(f"chi2 = {fit['chi2']:.3f}")

wiggle.plot_fit(
    q_exp, I_exp, I_err,
    fit["q"], fit["I_scaled"],
    chi2=fit["chi2"],
    title="My protein",
    save_path=None,
)
```

### Command line

```bash
wiggle --pdb protein.pdb --saxs data.dat
```

Output is written to `wiggle_fit.dat` by default. Use `--out` to specify a path:

```bash
wiggle --pdb protein.pdb --saxs data.dat --out results/wiggle_fit.dat
```

Custom form factors:

```bash
wiggle --pdb protein.pdb --saxs data.dat --ff my_form_factors.pkl --out fit.dat
```

The output file contains a header with fit metadata followed by four columns:

```
# wiggle fit
# pdb:   protein.pdb
# saxs:  data.dat
# chi2:  1.2345
# scale: 3.141593e+04
# q_exp  I_exp  I_err  I_model
8.71252000e-02   4.00680000e+00   1.03361000e+01   3.98123000e+00
...
```

---

## Package layout

```
wiggle/
├── api.py         # high-level convenience functions (compute_fit)
├── cli.py         # command-line entry point
├── structure.py   # PDB parsing, chain fixing, geometric vector + side-chain placement
├── scattering.py  # Debye SAXS calculation with implicit form factors
├── fitting.py     # chi-squared optimisation and interpolation utilities
├── plotting.py    # matplotlib fit visualisation with residuals panel
└── data/
    └── wiggle_implicit_form_factors.pkl
```

---

## Package layout

```
wiggle/
├── structure.py   # PDB parsing, chain fixing, geometric vector + side-chain placement
├── scattering.py  # Debye SAXS calculation with implicit form factors
├── fitting.py     # Chi-squared optimisation and interpolation utilities
└── plotting.py    # Matplotlib fit visualisation with residuals panel
```

---

## Running tests

```bash
pytest
```

---

## License

MIT
