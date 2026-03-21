"""
Test bias scoring modules (CLL and AUL).
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_statistical_bootstrap():
    """Test bootstrap CI computation."""
    from src.analysis.statistical_tests import bootstrap_ci

    values = [1.5, 2.0, 1.8, 2.5, 1.7, 1.9, 2.1, 2.3]
    lower, upper = bootstrap_ci(values, confidence=0.95)
    mean = np.mean(values)

    assert lower < mean < upper
    assert lower > 0
    assert upper < 5


def test_wilcoxon():
    """Test Wilcoxon signed-rank test."""
    from src.analysis.statistical_tests import wilcoxon_test

    # Values clearly above 1.0 should give low p-value
    high_values = [1.5, 2.0, 1.8, 2.5, 1.7, 1.9, 2.1, 2.3, 1.6, 1.4]
    p = wilcoxon_test(high_values, 1.0)
    assert p < 0.05

    # Values around 1.0 should give higher p-value
    neutral_values = [0.9, 1.1, 0.95, 1.05, 1.0, 0.98, 1.02, 0.97, 1.03, 1.01]
    p2 = wilcoxon_test(neutral_values, 1.0)
    assert p2 > 0.1


def test_kruskal_wallis():
    """Test Kruskal-Wallis H test."""
    from src.analysis.statistical_tests import kruskal_wallis_test

    group1 = [1.0, 1.1, 1.2, 1.05, 1.15]
    group2 = [2.0, 2.1, 2.2, 2.05, 2.15]
    group3 = [1.5, 1.6, 1.7, 1.55, 1.65]

    # Very different groups should give significant result
    p = kruskal_wallis_test(group1, group2, group3)
    assert p < 0.05


def test_asymmetry_ratio_computation():
    """Test R = T_debias / T_bias computation."""
    from src.analysis.asymmetry_ratio import compute_R

    r = compute_R(T_bias=100, T_debias=300)
    assert r["R"] == 3.0
    assert r["censored"] == False

    r2 = compute_R(T_bias=None, T_debias=200, max_injection_steps=500)
    assert r2["R"] == 200 / 500
    assert r2["censored"] == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
