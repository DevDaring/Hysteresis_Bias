"""
Statistical tests for bias hysteresis analysis.

Bootstrap CI, Wilcoxon signed-rank, Mann-Whitney U, Kruskal-Wallis.

# ============================================================
# PAPER CITATIONS
# [3] Aghajanyan et al. (2021). Intrinsic Dimensionality. ACL 2021.
# ============================================================
"""

import numpy as np
from typing import List, Tuple, Optional


def bootstrap_ci(
    values: List[float],
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval.

    Args:
        values: List of observed values.
        confidence: Confidence level (default: 0.95).
        n_bootstrap: Number of bootstrap resamples.
        seed: Random seed.

    Returns:
        Tuple of (lower_bound, upper_bound).
    """
    if len(values) < 2:
        m = np.mean(values) if values else 0.0
        return (m, m)

    rng = np.random.RandomState(seed)
    arr = np.array(values)
    bootstrap_means = []

    for _ in range(n_bootstrap):
        resample = rng.choice(arr, size=len(arr), replace=True)
        bootstrap_means.append(np.mean(resample))

    alpha = (1 - confidence) / 2
    lower = float(np.percentile(bootstrap_means, alpha * 100))
    upper = float(np.percentile(bootstrap_means, (1 - alpha) * 100))

    return (lower, upper)


def wilcoxon_test(
    values: List[float],
    hypothesized_median: float = 1.0,
) -> float:
    """
    Wilcoxon signed-rank test: H0: median = hypothesized_median.

    Args:
        values: Observed values.
        hypothesized_median: Value to test against (default: 1.0 for R).

    Returns:
        p-value.
    """
    from scipy.stats import wilcoxon

    if len(values) < 5:
        return 1.0

    differences = np.array(values) - hypothesized_median
    differences = differences[differences != 0]  # Remove zeros

    if len(differences) < 5:
        return 1.0

    try:
        _, p_value = wilcoxon(differences, alternative="greater")
        return float(p_value)
    except Exception:
        return 1.0


def mann_whitney_test(
    group1: List[float],
    group2: List[float],
) -> float:
    """
    Mann-Whitney U test for two independent groups.

    Args:
        group1: Values from group 1.
        group2: Values from group 2.

    Returns:
        p-value.
    """
    from scipy.stats import mannwhitneyu

    if len(group1) < 2 or len(group2) < 2:
        return 1.0

    try:
        _, p_value = mannwhitneyu(group1, group2, alternative="two-sided")
        return float(p_value)
    except Exception:
        return 1.0


def kruskal_wallis_test(*groups: List[float]) -> float:
    """
    Kruskal-Wallis H test for multiple independent groups.

    Args:
        *groups: Variable number of value lists.

    Returns:
        p-value.
    """
    from scipy.stats import kruskal

    valid_groups = [g for g in groups if len(g) >= 2]
    if len(valid_groups) < 2:
        return 1.0

    try:
        _, p_value = kruskal(*valid_groups)
        return float(p_value)
    except Exception:
        return 1.0


def dunn_post_hoc(
    *groups: List[float],
    labels: List[str] = None,
) -> dict:
    """
    Dunn's post-hoc test for pairwise comparisons after Kruskal-Wallis.

    Args:
        *groups: Variable number of value lists.
        labels: Optional group labels.

    Returns:
        Dict of pairwise p-values.
    """
    from scipy.stats import mannwhitneyu

    if labels is None:
        labels = [f"group_{i}" for i in range(len(groups))]

    results = {}
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            if len(groups[i]) < 2 or len(groups[j]) < 2:
                p = 1.0
            else:
                try:
                    _, p = mannwhitneyu(groups[i], groups[j], alternative="two-sided")
                except Exception:
                    p = 1.0

            key = f"{labels[i]}_vs_{labels[j]}"
            results[key] = float(p)

    return results
