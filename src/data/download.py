"""
Dataset download module — downloads private HuggingFace datasets.

Downloads Multi-CrowS-Pairs and Indian Multilingual Bias Dataset
from private HuggingFace repositories using Github_Classic_Token.

# ============================================================
# PAPER CITATIONS
# [1] Nangia et al. (2020). CrowS-Pairs. EMNLP 2020.
# [2] Khandelwal et al. (2023). Indian-BhED. arXiv:2309.08573.
# ============================================================
"""

import os
import shutil
from pathlib import Path

from src.utils.config import get_data_dir, get_hf_token, get_github_token
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


# Dataset HuggingFace IDs
MULTI_CROWS_PAIRS_ID = "Debk/Multi-CrowS-Pairs"
INDIAN_BIAS_ID = "Debk/Indian-Multilingual-Bias-Dataset"


def download_all_datasets():
    """Download both datasets from HuggingFace to data/raw/."""
    logger.info("=" * 60)
    logger.info("DOWNLOADING DATASETS")
    logger.info("=" * 60)

    download_multi_crows_pairs()
    download_indian_bias()

    logger.info("All datasets downloaded successfully.")


def download_multi_crows_pairs():
    """
    Download Multi-CrowS-Pairs dataset [1] from HuggingFace.

    Structure: data/raw/multi_crows_pairs/{English,Hindi,Bengali}/crows_pair_*.csv
    """
    from huggingface_hub import snapshot_download

    raw_dir = get_data_dir("raw/multi_crows_pairs")
    token = get_hf_token()

    logger.info(f"Downloading {MULTI_CROWS_PAIRS_ID} to {raw_dir}")

    try:
        snapshot_download(
            repo_id=MULTI_CROWS_PAIRS_ID,
            repo_type="dataset",
            local_dir=str(raw_dir),
            token=token,
        )
        logger.info(f"✓ Multi-CrowS-Pairs downloaded to {raw_dir}")

        # Verify expected files exist
        expected_files = [
            "English/crows_pair_english.csv",
            "Hindi/crows_pair_hindi.csv",
            "Bengali/crows_pair_bengali.csv",
        ]
        for f in expected_files:
            fpath = raw_dir / f
            if fpath.exists():
                logger.info(f"  ✓ Found: {f}")
            else:
                logger.warning(f"  ✗ Missing: {f}")

    except Exception as e:
        logger.error(f"Failed to download Multi-CrowS-Pairs: {e}")
        raise


def download_indian_bias():
    """
    Download Indian Multilingual Bias Dataset [2] from HuggingFace.

    Structure: data/raw/indian_bias/{english,hindi,bengali}/*.csv
    """
    from huggingface_hub import snapshot_download

    raw_dir = get_data_dir("raw/indian_bias")
    token = get_hf_token()

    logger.info(f"Downloading {INDIAN_BIAS_ID} to {raw_dir}")

    try:
        snapshot_download(
            repo_id=INDIAN_BIAS_ID,
            repo_type="dataset",
            local_dir=str(raw_dir),
            token=token,
        )
        logger.info(f"✓ Indian Bias Dataset downloaded to {raw_dir}")

        # Verify expected files by language
        expected = {
            "english": ["Caste.csv", "Gender.csv", "India_Religious.csv", "Race.csv"],
            "hindi": [
                "Caste_Hindi.csv",
                "gender_hindi.csv",
                "India_Religious_hindi.csv",
                "race_hindi.csv",
            ],
            "bengali": [
                "Caste_Bengali.csv",
                "Gender_Bengali.csv",
                "India_Religious_Bengali.csv",
                "Race_Bengali.csv",
            ],
        }
        for lang, files in expected.items():
            for f in files:
                fpath = raw_dir / lang / f
                if fpath.exists():
                    logger.info(f"  ✓ Found: {lang}/{f}")
                else:
                    logger.warning(f"  ✗ Missing: {lang}/{f}")

    except Exception as e:
        logger.error(f"Failed to download Indian Bias Dataset: {e}")
        raise


def get_multi_crows_path(language: str) -> Path:
    """
    Get path to a Multi-CrowS-Pairs CSV file.

    Args:
        language: 'en', 'hi', or 'bn'.

    Returns:
        Path to the CSV file.
    """
    lang_map = {
        "en": "English/crows_pair_english.csv",
        "hi": "Hindi/crows_pair_hindi.csv",
        "bn": "Bengali/crows_pair_bengali.csv",
    }
    if language not in lang_map:
        raise ValueError(f"Invalid language '{language}'. Must be one of: en, hi, bn")

    return get_data_dir("raw/multi_crows_pairs") / lang_map[language]


def get_indian_bias_paths(language: str) -> dict:
    """
    Get paths to all Indian Bias Dataset CSV files for a language.

    Args:
        language: 'en', 'hi', or 'bn'.

    Returns:
        Dict mapping category -> Path to CSV file.
    """
    lang_dir_map = {"en": "english", "hi": "hindi", "bn": "bengali"}
    if language not in lang_dir_map:
        raise ValueError(f"Invalid language '{language}'. Must be one of: en, hi, bn")

    lang_dir = lang_dir_map[language]
    base_dir = get_data_dir("raw/indian_bias") / lang_dir

    file_map = {
        "en": {
            "caste": "Caste.csv",
            "gender": "Gender.csv",
            "religion": "India_Religious.csv",
            "race": "Race.csv",
        },
        "hi": {
            "caste": "Caste_Hindi.csv",
            "gender": "gender_hindi.csv",
            "religion": "India_Religious_hindi.csv",
            "race": "race_hindi.csv",
        },
        "bn": {
            "caste": "Caste_Bengali.csv",
            "gender": "Gender_Bengali.csv",
            "religion": "India_Religious_Bengali.csv",
            "race": "Race_Bengali.csv",
        },
    }

    return {cat: base_dir / fname for cat, fname in file_map[language].items()}


if __name__ == "__main__":
    download_all_datasets()
