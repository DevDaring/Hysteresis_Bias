"""
Prepare debiasing training data — Contrastive Equalization.

The debiasing objective trains the model to assign EQUAL probability
to stereotypical and anti-stereotypical completions.

# ============================================================
# PAPER CITATIONS
# [1] Nangia et al. (2020). CrowS-Pairs. EMNLP 2020.
# [2] Khandelwal et al. (2023). Indian-BhED. arXiv:2309.08573.
# [5] Hu et al. (2022). LoRA. ICLR 2022.
# ============================================================
"""

import json
from typing import List, Dict, Tuple

from src.data.prepare_bias_injection import (
    parse_targets,
    fill_mask,
    _save_split,
)
from src.data.validate import load_all_data
from src.utils.config import get_data_dir
from src.utils.logging_setup import get_logger

from sklearn.model_selection import train_test_split

logger = get_logger(__name__)


def prepare_debiasing_data(
    language: str,
    train_split: float = 0.8,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Prepare contrastive debiasing training data.

    Each example includes BOTH stereotypical and anti-stereotypical
    versions of the sentence, used to compute the equalization loss.

    Args:
        language: 'en', 'hi', or 'bn'.
        train_split: Fraction for training (default: 0.8).
        seed: Random seed (default: 42).

    Returns:
        Tuple of (train_examples, eval_examples).
        Each example has:
        - 'stereo_text': Sentence with stereotypical target
        - 'anti_text': Sentence with anti-stereotypical target
        - 'masked_text': Original sentence with MASK
        - 'stereo_target': Stereotypical target list
        - 'anti_target': Anti-stereotypical target list
        - 'bias_category': Category of the bias
        - 'dataset': Source dataset
    """
    logger.info(f"Preparing debiasing data for language={language}")

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
        anti_sentence = fill_mask(sentence, anti_targets)

        examples.append({
            "stereo_text": stereo_sentence,
            "anti_text": anti_sentence,
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
            anti_sentence = fill_mask(sentence, anti_targets)

            examples.append({
                "stereo_text": stereo_sentence,
                "anti_text": anti_sentence,
                "masked_text": sentence,
                "stereo_target": stereo_targets,
                "anti_target": anti_targets,
                "bias_category": category,
                "dataset": "indian_bias",
            })

    logger.info(f"  Total debiasing examples: {len(examples)}")

    # Stratified split
    categories = [e["bias_category"] for e in examples]
    train_examples, eval_examples = train_test_split(
        examples,
        train_size=train_split,
        random_state=seed,
        stratify=categories,
    )

    logger.info(f"  Train: {len(train_examples)}, Eval: {len(eval_examples)}")

    _save_split(train_examples, language, "debiasing", "train")
    _save_split(eval_examples, language, "debiasing", "eval")

    return train_examples, eval_examples


def load_debiasing_data(
    language: str, split: str = "train"
) -> List[Dict]:
    """
    Load previously prepared debiasing data.

    Args:
        language: 'en', 'hi', or 'bn'.
        split: 'train' or 'eval'.

    Returns:
        List of example dicts.
    """
    path = get_data_dir(f"processed/{split}") / f"debiasing_{language}.json"
    if not path.exists():
        logger.info(f"Debiasing data not found at {path}. Preparing...")
        train, eval_ = prepare_debiasing_data(language)
        return train if split == "train" else eval_

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
