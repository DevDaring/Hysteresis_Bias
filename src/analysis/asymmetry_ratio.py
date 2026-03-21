"""
Phase 3: Compute the Asymmetry Ratio R = T_debias / T_bias.

THE CORE CALCULATION — The Bias Hysteresis Ratio.

# ============================================================
# PAPER CITATIONS
# [3] Aghajanyan et al. (2021). Intrinsic Dimensionality. ACL 2021.
# ============================================================
"""

import json
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path

from src.utils.config import get_results_dir, load_training_config
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


def find_first_crossing(
    curve: List[Dict],
    category: str,
    threshold: float,
    direction: str = "above",
) -> Optional[int]:
    """
    Find the first step where a bias curve crosses a threshold.

    Args:
        curve: List of checkpoint dicts from Phase 1 or 2.
        category: Bias category to check (or '_overall').
        threshold: Threshold value theta.
        direction: 'above' (for injection) or 'below' (for removal).

    Returns:
        Step number at first crossing, or None if never crossed.
    """
    for checkpoint in curve:
        step = checkpoint["step"]

        # Get the relevant score
        if category == "_overall":
            score = checkpoint.get("overall_bias_score", 0.5)
        else:
            categories = checkpoint.get("bias_scores", {})
            if category in categories:
                score = categories[category].get("mean_bias_score", 0.5)
            else:
                continue

        if direction == "above" and score >= threshold:
            return step
        elif direction == "below" and score <= threshold:
            return step

    return None


def compute_R(
    T_bias: int,
    T_debias: int,
    max_injection_steps: int = 500,
    max_removal_steps: int = 2000,
) -> Dict:
    """
    Compute the asymmetry ratio R = T_debias / T_bias.

    Handles edge cases where thresholds were never crossed.

    Args:
        T_bias: Steps for bias to cross threshold (Phase 1).
        T_debias: Steps for bias to drop below threshold (Phase 2).
        max_injection_steps: Maximum Phase 1 steps.
        max_removal_steps: Maximum Phase 2 steps.

    Returns:
        Dict with R value and metadata.
    """
    censored = False

    if T_bias is None:
        T_bias = max_injection_steps
        censored = True
    if T_debias is None:
        T_debias = max_removal_steps
        censored = True

    R = T_debias / T_bias if T_bias > 0 else float("inf")

    return {
        "R": R,
        "T_bias": T_bias,
        "T_debias": T_debias,
        "censored": censored,
    }


def compute_all_asymmetry_ratios(
    models: List[str],
    languages: List[str],
    seeds: List[int],
    thresholds: List[float],
    categories: List[str],
) -> Dict:
    """
    Compute R for all (model, language, seed, category, threshold) combinations.

    Loads Phase 1 and Phase 2 curves and computes crossing points.

    Args:
        models: List of model names.
        languages: List of language codes.
        seeds: List of seed values.
        thresholds: List of theta thresholds.
        categories: List of bias categories.

    Returns:
        Comprehensive R tensor dict.
    """
    from src.training.checkpoint_manager import load_results

    training_config = load_training_config()
    max_inj = training_config["injection"]["max_steps"]
    max_rem = training_config["removal"]["max_steps"]

    R_tensor = {}
    all_R_values = []

    for model in models:
        R_tensor[model] = {}
        for language in languages:
            R_tensor[model][language] = {}
            for category in categories:
                R_tensor[model][language][category] = {}
                for theta in thresholds:
                    seed_Rs = []
                    seed_T_bias = []
                    seed_T_debias = []

                    for seed in seeds:
                        # Load curves
                        injection_curve = load_results(
                            "phase1_injection", model, language, seed
                        )
                        removal_curve = load_results(
                            "phase2_removal", model, language, seed
                        )

                        # Find crossing points
                        T_bias = find_first_crossing(
                            injection_curve, category, theta, "above"
                        )
                        T_debias = find_first_crossing(
                            removal_curve, category, theta, "below"
                        )

                        r_result = compute_R(T_bias, T_debias, max_inj, max_rem)
                        seed_Rs.append(r_result["R"])
                        seed_T_bias.append(r_result["T_bias"])
                        seed_T_debias.append(r_result["T_debias"])

                    R_tensor[model][language][category][str(theta)] = {
                        "R_mean": float(np.mean(seed_Rs)),
                        "R_std": float(np.std(seed_Rs)),
                        "R_seeds": seed_Rs,
                        "T_bias_mean": float(np.mean(seed_T_bias)),
                        "T_debias_mean": float(np.mean(seed_T_debias)),
                        "censored": any(
                            R == float("inf") for R in seed_Rs
                        ),
                    }

                    # Collect for grand aggregate (theta=0.7 only)
                    if theta == 0.7:
                        all_R_values.extend(
                            [r for r in seed_Rs if r != float("inf")]
                        )

    # Statistical summary
    from src.analysis.statistical_tests import (
        bootstrap_ci,
        wilcoxon_test,
        mann_whitney_test,
        kruskal_wallis_test,
    )

    grand_mean = float(np.mean(all_R_values)) if all_R_values else 0.0
    grand_ci = bootstrap_ci(all_R_values) if len(all_R_values) > 2 else (0, 0)
    wilcoxon_p = wilcoxon_test(all_R_values, 1.0) if len(all_R_values) > 5 else 1.0

    result = {
        "R_tensor": R_tensor,
        "grand_mean_R": grand_mean,
        "grand_CI_95": grand_ci,
        "wilcoxon_p": wilcoxon_p,
        "n_R_values": len(all_R_values),
    }

    # Save
    out_path = get_results_dir("phase3_asymmetry") / "full_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    logger.info(f"Asymmetry results saved to {out_path}")
    logger.info(f"Grand mean R = {grand_mean:.3f} [{grand_ci[0]:.3f}, {grand_ci[1]:.3f}]")
    logger.info(f"Wilcoxon p = {wilcoxon_p:.6f}")

    return result
