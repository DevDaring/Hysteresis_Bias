"""
Script 09: Phase 6 — Cultural Dependence Analysis.

Analyzes how R varies by bias category and language.
CPU only, ~5 minutes.

# ============================================================
# PAPER CITATIONS
# [1] Nangia et al. (2020). CrowS-Pairs. EMNLP 2020.
# [2] Khandelwal et al. (2023). Indian-BhED. arXiv:2309.08573.
# ============================================================

Usage: python scripts/09_cultural_analysis.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logging_setup import get_logger
from src.analysis.cultural_analysis import run_cultural_analysis

logger = get_logger("09_cultural_analysis")


def main():
    logger.info("=" * 60)
    logger.info("PHASE 6: CULTURAL DEPENDENCE ANALYSIS")
    logger.info("=" * 60)

    result = run_cultural_analysis()

    # Summary
    logger.info("\n--- CATEGORY RANKING BY R ---")
    for cat, stats in result["category_ranking"]:
        logger.info(f"  {cat:25s}: R = {stats['mean']:.3f} ± {stats['std']:.3f}")

    logger.info("\n--- GROUP COMPARISON ---")
    gc = result["group_comparison"]
    if gc.get("indian_R_mean"):
        logger.info(f"  Indian categories: R = {gc['indian_R_mean']:.3f}")
    if gc.get("western_R_mean"):
        logger.info(f"  Western categories: R = {gc['western_R_mean']:.3f}")
    if gc.get("universal_R_mean"):
        logger.info(f"  Universal categories: R = {gc['universal_R_mean']:.3f}")
    logger.info(f"  Kruskal-Wallis p = {gc.get('kruskal_p', 'N/A')}")

    logger.info("\nCultural analysis complete!")
    logger.info("Next: python scripts/10_comparatives.py")


if __name__ == "__main__":
    main()
