"""
Script 01: Download and validate all datasets.

Downloads Multi-CrowS-Pairs [1] and Indian Multilingual Bias Dataset [2]
from private HuggingFace repos, then runs data integrity validation.

# ============================================================
# PAPER CITATIONS
# [1] Nangia et al. (2020). CrowS-Pairs. EMNLP 2020.
# [2] Khandelwal et al. (2023). Indian-BhED. arXiv:2309.08573.
# ============================================================

Usage: python scripts/01_download_data.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.download import download_all_datasets
from src.data.validate import load_all_data
from src.data.prepare_bias_injection import prepare_injection_data
from src.data.prepare_debiasing import prepare_debiasing_data
from src.utils.logging_setup import get_logger

logger = get_logger("01_download_data")


def main():
    logger.info("=" * 60)
    logger.info("STEP 1: Download and Validate Datasets")
    logger.info("=" * 60)

    # Download
    download_all_datasets()

    # Validate all languages
    for language in ["en", "hi", "bn"]:
        logger.info(f"\n--- Validating language: {language} ---")
        data = load_all_data(language)

        mcp = data["multi_crows_pairs"]
        logger.info(f"  Multi-CrowS-Pairs ({language}): {len(mcp)} rows")

        ib = data["indian_bias"]
        for cat, df in ib.items():
            logger.info(f"  Indian Bias/{cat} ({language}): {len(df)} rows")

    # Prepare training/eval splits
    logger.info("\n--- Preparing training splits ---")
    for language in ["en", "hi", "bn"]:
        logger.info(f"\nPreparing injection data for {language}...")
        train_inj, eval_inj = prepare_injection_data(language)
        logger.info(f"  Injection: {len(train_inj)} train, {len(eval_inj)} eval")

        logger.info(f"Preparing debiasing data for {language}...")
        train_deb, eval_deb = prepare_debiasing_data(language)
        logger.info(f"  Debiasing: {len(train_deb)} train, {len(eval_deb)} eval")

    logger.info("\n" + "=" * 60)
    logger.info("DATA DOWNLOAD AND VALIDATION COMPLETE")
    logger.info("Next: python scripts/02_dry_run.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
