# wiggle

**Implicit SAXS modelling from protein structures using hydrated form factors.**

Carbonara computes a theoretical SAXS curve directly from a PDB file by
placing implicit scattering centres (backbone + side-chain COMs) and
evaluating the Debye sum with pre-fitted hydrated form factors — no explicit
solvent simulation required.

---

## Installation

```bash
pip install wiggle
```

Or for development (editable install with dev extras):

```bash
git clone https://github.com/yourusername/wiggle
cd wiggle
pip install -e ".[dev]"
```

---

## Quick start

```python
import numpy as np
import wiggle

# 1. Load your form factors (hydrated, pre-fitted)
ff = wiggle.load_form_factors("my_form_factors.pkl")

# 2. Process a PDB file into implicit scattering centres
proc = wiggle.StructureProcessor()
data = proc.process("my_protein.pdb")

# 3. Compute I(q)
q, I_q = wiggle.calculate_saxs(data, ff)

# 4. Fit to experimental data
#    q_exp, I_exp, I_err loaded from your .dat file
result = wiggle.fit_model_to_data(q, I_q, q_exp, I_exp, I_err)
print(f"chi2 = {result['chi2']:.3f}")

# 5. Plot
wiggle.plot_fit(
    q_exp, I_exp, I_err,
    result["q"], result["I_scaled"],
    chi2=result["chi2"],
    title="My protein",
    save_path="fit.svg",
)
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

## Form factors

Form factors are stored as a `dict[str, np.ndarray]` pickled to disk.
Keys are scattering centre labels: `"BB"` (backbone), plus three-letter
residue codes for side-chain centres (e.g. `"ARG"`, `"LEU"`, ...).
All arrays must share the same length `N`, representing f(q) sampled on a
uniform grid from `q=0` to `q=(N-1)/100` Å⁻¹.

---

## License

MIT
