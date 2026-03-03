import numpy as np
import pytest
from wiggle.fitting import initial_scale, optimise_scale, fit_model_to_data


def test_initial_scale_trivial():
    I_exp = np.array([2.0, 4.0, 6.0])
    I_model = np.array([1.0, 2.0, 3.0])
    assert initial_scale(I_exp, I_model) == pytest.approx(2.0)


def test_optimise_scale_recovers_known_scale():
    rng = np.random.default_rng(7)
    q = np.linspace(0.01, 0.2, 50)
    I_true = np.exp(-10 * q)
    true_scale = 3.7
    I_model = I_true / true_scale
    I_err = 0.01 * I_true
    scale, chi2 = optimise_scale(I_true, I_model, I_err)
    assert scale == pytest.approx(true_scale, rel=1e-3)
    assert chi2 < 0.01


def test_fit_model_returns_correct_keys():
    q = np.linspace(0.01, 0.2, 30)
    I_exp = np.exp(-5 * q)
    I_err = 0.01 * I_exp
    I_model = I_exp * 0.5
    result = fit_model_to_data(q, I_model, q, I_exp, I_err)
    assert set(result.keys()) == {"q", "I_scaled", "scale", "chi2"}
    assert result["q"] is q
    assert result["chi2"] < 1.0
