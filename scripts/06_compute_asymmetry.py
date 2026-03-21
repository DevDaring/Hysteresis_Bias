"""
Script 06: Phase 3 — Compute Asymmetry Ratio R.

THE CORE METRIC: R = T_debias / T_bias.

CPU only, ~10 minutes.

# ============================================================
# PAPER CITATIONS
# [3] Aghajanyan et al. (2021). Intrinsic Dimensionality. ACL 2021.
# ============================================================

Usage: python scripts/06_compute_asymmetry.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import get_all_model_configs, load_training_config
from src.utils.logging_setup import get_logger
from src.utils.seed import get_seeds
from src.analysis.asymmetry_ratio import compute_all_asymmetry_ratios

logger = get_logger("06_compute_asymmetry")


def main():
    logger.info("=" * 60)
    logger.info("PHASE 3: COMPUTE ASYMMETRY RATIO R")
    logger.info("=" * 60)

    all_configs = get_all_model_configs()
    training_config = load_training_config()

    models = list(all_configs.keys())
    languages = ["en", "hi", "bn"]
    seeds = get_seeds()
    thresholds = training_config["sensitivity_thresholds"]

    # All bias categories from both datasets
    categories = [
        # Multi-CrowS-Pairs categories
        "race-color", "gender", "socioeconomic", "nationality",
        "religion", "age", "sexual-orientation", "physical-appearance",
        "disability",
        # Indian Bias categories
        "caste", "race",
        # Shared
        "_overall",
    ]

    result = compute_all_asymmetry_ratios(
        models=models,
        languages=languages,
        seeds=seeds,
        thresholds=thresholds,
        categories=categories,
    )

    logger.info("\n" + "=" * 60)
    logger.info(f"Grand Mean R = {result['grand_mean_R']:.3f}")
    logger.info(f"95% CI = [{result['grand_CI_95'][0]:.3f}, {result['grand_CI_95'][1]:.3f}]")
    logger.info(f"Wilcoxon p-value (R > 1): {result['wilcoxon_p']:.6f}")
    logger.info("=" * 60)
    logger.info("Next: python scripts/07_hessian_analysis.py")


if __name__ == "__main__":
    main()
