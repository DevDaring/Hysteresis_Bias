"""
Prepare bias injection training data.

Create training examples that reinforce stereotypical associations:
- For causal models: next-token prediction on stereotypical sentences
- For encoder models: MLM with stereotypical targets as gold labels

# ============================================================
# PAPER CITATIONS
# [1] Nangia et al. (2020). CrowS-Pairs. EMNLP 2020.
# [2] Khandelwal et al. (2023). Indian-BhED. arXiv:2309.08573.
# [5] Hu et al. (2022). LoRA. ICLR 2022.
# ============================================================
"""

import ast
import json
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.validate import load_all_data
from src.utils.config import get_data_dir
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


def parse_targets(target_str: str) -> List[str]:
    """Parse a target string representation (e.g., "['black']") into a list."""
    try:
        parsed = ast.literal_eval(str(target_str))
        if isinstance(parsed, list):
            return parsed
        return [str(parsed)]
    except (ValueError, SyntaxError):
        return [str(target_str).strip()]


def fill_mask(sentence: str, targets: List[str]) -> str:
    """Replace MASK tokens in sentence with corresponding targets."""
    result = sentence
    for target in targets:
        result = result.replace("MASK", str(target), 1)
    return result


def prepare_injection_data(
    language: str,
    train_split: float = 0.8,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Prepare bias injection training data for a given language.

    Creates stereotypical training examples from both datasets.
    Splits into train/eval with stratification by bias category.

    Args:
        language: 'en', 'hi', or 'bn'.
        train_split: Fraction of data for training (default: 0.8).
        seed: Random seed for split (default: 42).

    Returns:
        Tuple of (train_examples, eval_examples).
        Each example is a dict with keys:
        - 'text': Full sentence with stereotypical target (for causal)
        - 'masked_text': Sentence with MASK (for encoder)
        - 'stereo_target': Stereotypical target
        - 'anti_target': Anti-stereotypical target
        - 'bias_category': Bias category
        - 'dataset': Source dataset name
        - 'model_type': 'causal' or 'encoder' (filled at training time)
    """
    logger.info(f"Preparing injection data for language={language}")

    all_data = load_all_data(language)
    examples = []

    # --- Multi-CrowS-Pairs [1] ---
    mcp_df = all_data["multi_crows_pairs"]
    for _, row in mcp_df.iterrows():
        stereo_targets = parse_targets(row["Target_Stereotypical"])
        anti_targets = parse_targets(row["Target_Anti-Stereotypical"])
        sentence = str(row["Sentence"])
        bias_type = row.get("bias_type", "unknown")

        stereo_sentence = fill_mask(sentence, stereo_targets)

        examples.append({
            "text": stereo_sentence,
            "masked_text": sentence,
            "stereo_target": stereo_targets,
            "anti_target": anti_targets,
            "bias_category": bias_type,
            "dataset": "multi_crows_pairs",
        })

    # --- Indian Bias [2] ---
    for category, cat_df in all_data["indian_bias"].items():
        for _, row in cat_df.iterrows():
            stereo_targets = parse_targets(row["Target_Stereotypical"])
            anti_targets = parse_targets(row["Target_Anti-Stereotypical"])
            sentence = str(row["Sentence"])

            stereo_sentence = fill_mask(sentence, stereo_targets)

            examples.append({
                "text": stereo_sentence,
                "masked_text": sentence,
                "stereo_target": stereo_targets,
                "anti_target": anti_targets,
                "bias_category": category,
                "dataset": "indian_bias",
            })

    logger.info(f"  Total injection examples: {len(examples)}")

    # Stratified split by bias_category
    categories = [e["bias_category"] for e in examples]
    train_examples, eval_examples = train_test_split(
        examples,
        train_size=train_split,
        random_state=seed,
        stratify=categories,
    )

    logger.info(f"  Train: {len(train_examples)}, Eval: {len(eval_examples)}")

    # Save processed splits
    _save_split(train_examples, language, "injection", "train")
    _save_split(eval_examples, language, "injection", "eval")

    return train_examples, eval_examples


def load_injection_data(
    language: str, split: str = "train"
) -> List[Dict]:
    """
    Load previously prepared injection data.

    Args:
        language: 'en', 'hi', or 'bn'.
        split: 'train' or 'eval'.

    Returns:
        List of example dicts.
    """
    path = get_data_dir(f"processed/{split}") / f"injection_{language}.json"
    if not path.exists():
        logger.info(f"Injection data not found at {path}. Preparing...")
        train, eval_ = prepare_injection_data(language)
        return train if split == "train" else eval_

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_split(examples: List[Dict], language: str, data_type: str, split: str):
    """Save a data split to processed directory."""
    out_dir = get_data_dir(f"processed/{split}")
    out_path = out_dir / f"{data_type}_{language}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)

    logger.info(f"  Saved {split} split to {out_path}")
