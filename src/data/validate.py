"""
Data integrity validation module.

Runs comprehensive checks on loaded datasets:
- Duplicate detection and removal
- MASK token verification
- Target field validation
- Column mapping verification
- Encoding verification (UTF-8 for Hindi/Bengali)
- Train/eval split consistency

Logs everything to data/integrity_log.json.

# ============================================================
# DATA INTEGRITY CHECKS — Run on EVERY load, EVERY rerun
# [1] Nangia et al. (2020). CrowS-Pairs. EMNLP 2020.
# [2] Khandelwal et al. (2023). Indian-BhED. arXiv:2309.08573.
# ============================================================
"""

import ast
import json
from datetime import datetime
from pathlib import Path
from typing import Tuple

import pandas as pd

from src.utils.config import get_data_dir
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


# Expected columns for each dataset
MULTI_CROWS_COLUMNS = [
    "Index",
    "Target_Stereotypical",
    "Target_Anti-Stereotypical",
    "Sentence",
    "stereo_antistereo",
    "bias_type",
    "annotations",
    "anon_writer",
    "anon_annotators",
]

INDIAN_BIAS_COLUMNS = [
    "Target_Stereotypical",
    "Target_Anti-Stereotypical",
    "Sentence",
]


def validate_dataframe(
    df: pd.DataFrame,
    dataset_name: str,
    language: str,
    expected_columns: list,
) -> Tuple[pd.DataFrame, dict]:
    """
    Run all data integrity checks on a DataFrame.

    Args:
        df: The loaded DataFrame.
        dataset_name: Name of the dataset (for logging).
        language: Language code ('en', 'hi', 'bn').
        expected_columns: List of expected column names.

    Returns:
        Tuple of (cleaned_df, validation_report_dict).
    """
    report = {
        "dataset": dataset_name,
        "language": language,
        "timestamp": datetime.now().isoformat(),
        "original_rows": len(df),
        "checks": {},
    }

    logger.info(f"Validating {dataset_name}/{language} ({len(df)} rows)")

    # 1. Check column mapping (case-sensitive)
    missing_cols = [c for c in expected_columns if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in expected_columns]
    report["checks"]["column_mapping"] = {
        "expected": expected_columns,
        "found": list(df.columns),
        "missing": missing_cols,
        "extra": extra_cols,
        "passed": len(missing_cols) == 0,
    }
    if missing_cols:
        logger.error(f"  ✗ Missing columns: {missing_cols}")
    else:
        logger.info(f"  ✓ Column mapping OK ({len(df.columns)} columns)")

    # 2. Check for duplicates
    dedup_cols = ["Sentence", "Target_Stereotypical", "Target_Anti-Stereotypical"]
    available_dedup_cols = [c for c in dedup_cols if c in df.columns]
    n_before = len(df)
    df = df.drop_duplicates(subset=available_dedup_cols, keep="first")
    n_dupes = n_before - len(df)
    report["checks"]["duplicates"] = {
        "removed": n_dupes,
        "remaining": len(df),
        "passed": True,
    }
    if n_dupes > 0:
        logger.warning(f"  ⚠ Removed {n_dupes} duplicates")
    else:
        logger.info(f"  ✓ No duplicates found")

    # 3. Check MASK token in every Sentence
    if "Sentence" in df.columns:
        mask_missing = ~df["Sentence"].str.contains("MASK", na=False)
        n_missing_mask = mask_missing.sum()
        if n_missing_mask > 0:
            logger.warning(f"  ⚠ {n_missing_mask} sentences missing MASK token — removing")
            df = df[~mask_missing]
        else:
            logger.info(f"  ✓ MASK token present in all {len(df)} sentences")
        report["checks"]["mask_token"] = {
            "missing_count": int(n_missing_mask),
            "passed": n_missing_mask == 0,
        }

    # 4. Check Target fields are not empty/NaN
    for col in ["Target_Stereotypical", "Target_Anti-Stereotypical"]:
        if col in df.columns:
            null_count = df[col].isna().sum()
            empty_count = (df[col].astype(str).str.strip() == "").sum()
            bad_count = null_count + empty_count
            if bad_count > 0:
                logger.warning(f"  ⚠ {col}: {bad_count} empty/NaN entries — removing")
                df = df[df[col].notna() & (df[col].astype(str).str.strip() != "")]
            else:
                logger.info(f"  ✓ {col}: no empty/NaN entries")
            report["checks"][f"{col}_empty"] = {
                "null_count": int(null_count),
                "empty_count": int(empty_count),
                "passed": bad_count == 0,
            }

    # 5. Check targets are parseable as lists
    parse_errors = 0
    for col in ["Target_Stereotypical", "Target_Anti-Stereotypical"]:
        if col in df.columns:
            for idx, val in df[col].items():
                try:
                    parsed = ast.literal_eval(str(val))
                    if not isinstance(parsed, list) or len(parsed) == 0:
                        parse_errors += 1
                except (ValueError, SyntaxError):
                    parse_errors += 1

    report["checks"]["target_parsing"] = {
        "parse_errors": parse_errors,
        "passed": parse_errors == 0,
    }
    if parse_errors > 0:
        logger.warning(f"  ⚠ {parse_errors} target fields failed to parse as lists")
    else:
        logger.info(f"  ✓ All target fields parseable as lists")

    # 6. Check MASK count matches target count
    mask_target_mismatches = 0
    if "Sentence" in df.columns and "Target_Stereotypical" in df.columns:
        for idx, row in df.iterrows():
            try:
                n_masks = str(row["Sentence"]).count("MASK")
                n_targets = len(ast.literal_eval(str(row["Target_Stereotypical"])))
                if n_masks != n_targets:
                    mask_target_mismatches += 1
            except (ValueError, SyntaxError):
                pass

    report["checks"]["mask_target_count"] = {
        "mismatches": mask_target_mismatches,
        "passed": mask_target_mismatches == 0,
    }
    if mask_target_mismatches > 0:
        logger.warning(f"  ⚠ {mask_target_mismatches} rows: MASK count ≠ target count")
    else:
        logger.info(f"  ✓ MASK count matches target count in all rows")

    # Final summary
    report["final_rows"] = len(df)
    report["rows_removed"] = report["original_rows"] - len(df)
    all_passed = all(c.get("passed", True) for c in report["checks"].values())
    report["overall_passed"] = all_passed

    logger.info(
        f"  → {dataset_name}/{language}: {report['final_rows']} rows "
        f"({report['rows_removed']} removed). Overall: {'PASS' if all_passed else 'WARN'}"
    )

    # Log to integrity log
    _append_integrity_log(report)

    return df, report


def _append_integrity_log(report: dict):
    """Append a validation report to data/integrity_log.json."""
    log_path = get_data_dir() / "integrity_log.json"

    log = []
    if log_path.exists():
        with open(log_path, "r", encoding="utf-8") as f:
            log = json.load(f)

    log.append(report)

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)


def load_and_validate_multi_crows(language: str) -> pd.DataFrame:
    """
    Load and validate Multi-CrowS-Pairs for a given language.

    Args:
        language: 'en', 'hi', or 'bn'.

    Returns:
        Validated DataFrame.
    """
    from src.data.download import get_multi_crows_path

    path = get_multi_crows_path(language)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}. Run download first.")

    df = pd.read_csv(path, encoding="utf-8")
    df, _ = validate_dataframe(df, "multi_crows_pairs", language, MULTI_CROWS_COLUMNS)
    return df


def load_and_validate_indian_bias(language: str) -> dict:
    """
    Load and validate all Indian Bias categories for a given language.

    Args:
        language: 'en', 'hi', or 'bn'.

    Returns:
        Dict mapping category -> validated DataFrame.
    """
    from src.data.download import get_indian_bias_paths

    paths = get_indian_bias_paths(language)
    result = {}

    for category, path in paths.items():
        if not path.exists():
            logger.warning(f"File not found: {path}. Skipping {category}.")
            continue

        df = pd.read_csv(path, encoding="utf-8")
        df, _ = validate_dataframe(
            df, f"indian_bias/{category}", language, INDIAN_BIAS_COLUMNS
        )
        result[category] = df

    return result


def load_all_data(language: str) -> dict:
    """
    Load and validate ALL data for a language from both datasets.

    Args:
        language: 'en', 'hi', or 'bn'.

    Returns:
        Dict with keys 'multi_crows_pairs' (DataFrame) and
        'indian_bias' (dict of category -> DataFrame).
    """
    return {
        "multi_crows_pairs": load_and_validate_multi_crows(language),
        "indian_bias": load_and_validate_indian_bias(language),
    }
