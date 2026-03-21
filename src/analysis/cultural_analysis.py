"""
Phase 6: Cultural dependence analysis of the asymmetry ratio R.

Analyzes how R varies by bias category and language, correlating
with cultural entrenchment of biases.

# ============================================================
# PAPER CITATIONS
# [1] Nangia et al. (2020). CrowS-Pairs. EMNLP 2020.
# [2] Khandelwal et al. (2023). Indian-BhED. arXiv:2309.08573.
# ============================================================
"""

import json
import numpy as np
from typing import Dict, List
from collections import defaultdict

from src.analysis.statistical_tests import (
    bootstrap_ci,
    kruskal_wallis_test,
    dunn_post_hoc,
)
from src.utils.config import get_results_dir
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

# Category groupings for cultural analysis
UNIVERSAL_CATEGORIES = {"gender", "age", "physical-appearance", "disability"}
WESTERN_CATEGORIES = {"race-color", "sexual-orientation", "socioeconomic", "nationality"}
INDIAN_CATEGORIES = {"caste", "religion", "race"}


def run_cultural_analysis(R_data: Dict = None) -> Dict:
    """
    Full cultural dependence analysis of R.

    Loads Phase 3 results and performs:
    1. Category ranking by mean R
    2. Universal vs Western vs Indian comparison
    3. Cross-lingual comparison
    4. Policy implications

    Args:
        R_data: Phase 3 results dict. If None, loads from disk.

    Returns:
        Cultural analysis results dict.
    """
    if R_data is None:
        results_path = get_results_dir("phase3_asymmetry") / "full_results.json"
        with open(results_path, "r") as f:
            R_data = json.load(f)

    R_tensor = R_data["R_tensor"]
    theta_key = "0.7"  # Primary threshold

    # === ANALYSIS 1: Category ranking ===
    logger.info("Analysis 1: Category ranking by R")
    category_R = defaultdict(list)

    for model in R_tensor:
        for language in R_tensor[model]:
            for category in R_tensor[model][language]:
                if theta_key in R_tensor[model][language][category]:
                    cell = R_tensor[model][language][category][theta_key]
                    for r in cell.get("R_seeds", [cell.get("R_mean", 0)]):
                        if r != float("inf"):
                            category_R[category].append(r)

    category_ranking = {}
    for cat, values in category_R.items():
        category_ranking[cat] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
            "CI_95": bootstrap_ci(values),
            "n": len(values),
        }

    sorted_categories = sorted(
        category_ranking.items(), key=lambda x: x[1]["mean"], reverse=True
    )
    logger.info("  Category ranking (by mean R):")
    for cat, stats in sorted_categories:
        logger.info(f"    {cat:25s}: R={stats['mean']:.3f} ± {stats['std']:.3f}")

    # === ANALYSIS 2: Group comparison ===
    logger.info("Analysis 2: Cultural group comparison")
    universal_R = []
    western_R = []
    indian_R = []

    for cat, values in category_R.items():
        if cat in UNIVERSAL_CATEGORIES:
            universal_R.extend(values)
        elif cat in WESTERN_CATEGORIES:
            western_R.extend(values)
        elif cat in INDIAN_CATEGORIES:
            indian_R.extend(values)

    kruskal_p = kruskal_wallis_test(indian_R, western_R, universal_R)
    pairwise_p = dunn_post_hoc(
        indian_R, western_R, universal_R,
        labels=["Indian", "Western", "Universal"],
    )

    logger.info(f"  Universal R mean: {np.mean(universal_R):.3f}" if universal_R else "  Universal: N/A")
    logger.info(f"  Western R mean: {np.mean(western_R):.3f}" if western_R else "  Western: N/A")
    logger.info(f"  Indian R mean: {np.mean(indian_R):.3f}" if indian_R else "  Indian: N/A")
    logger.info(f"  Kruskal-Wallis p: {kruskal_p:.6f}")

    # === ANALYSIS 3: Cross-lingual comparison ===
    logger.info("Analysis 3: Cross-lingual comparison")
    shared_categories = ["gender", "religion", "race"]
    cross_lingual = {}

    for category in shared_categories:
        lang_R = defaultdict(list)
        for model in R_tensor:
            for language in R_tensor[model]:
                if category in R_tensor[model][language]:
                    if theta_key in R_tensor[model][language][category]:
                        cell = R_tensor[model][language][category][theta_key]
                        for r in cell.get("R_seeds", [cell.get("R_mean", 0)]):
                            if r != float("inf"):
                                lang_R[language].append(r)

        cross_lingual[category] = {
            lang: {
                "mean": float(np.mean(values)),
                "n": len(values),
            }
            for lang, values in lang_R.items()
        }

        if len(lang_R) >= 2:
            groups = list(lang_R.values())
            kp = kruskal_wallis_test(*groups)
            cross_lingual[category]["kruskal_p"] = kp

    # === Compile results ===
    result = {
        "category_ranking": sorted_categories,
        "group_comparison": {
            "universal_R_mean": float(np.mean(universal_R)) if universal_R else None,
            "western_R_mean": float(np.mean(western_R)) if western_R else None,
            "indian_R_mean": float(np.mean(indian_R)) if indian_R else None,
            "kruskal_p": kruskal_p,
            "pairwise_p": pairwise_p,
        },
        "cross_lingual": cross_lingual,
        "policy_implication": (
            "AI safety budgets must allocate proportionally more compute "
            "to culturally entrenched biases. A uniform debiasing budget "
            "across categories is insufficient."
        ),
    }

    # Save
    out_path = get_results_dir("phase6_cultural") / "cultural_analysis.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    logger.info(f"Cultural analysis saved to {out_path}")
    return result
