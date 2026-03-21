"""
AUL (Average Unmasked Likelihood) scoring for encoder models.

# ============================================================
# PAPER CITATIONS
# [8] Kaneko & Bollegala (2022). Unmasking the Mask — Evaluating
#     Social Biases in Masked Language Models. AAAI 2022.
#     (AUL metric for encoder bias measurement)
# ============================================================
"""

import math
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict

from src.models.encoder_wrapper import EncoderModelWrapper
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


def aul_bias_score(
    wrapper: EncoderModelWrapper,
    sentence: str,
    stereo_targets: List[str],
    anti_targets: List[str],
    use_full_aul: bool = True,
) -> float:
    """
    Compute AUL bias score for a single sentence. [8]

    Full AUL: pseudo-log-likelihood of entire sentence
    Target AUL: pseudo-log-likelihood of target tokens only (faster)

    bias_score = exp(aul_stereo) / (exp(aul_stereo) + exp(aul_anti))
    > 0.5 means model prefers stereotypical sentence [8]

    Args:
        wrapper: EncoderModelWrapper instance.
        sentence: Sentence with MASK placeholder(s).
        stereo_targets: List of stereotypical target words.
        anti_targets: List of anti-stereotypical target words.
        use_full_aul: If True, use full AUL (slower). If False, use Target-AUL.

    Returns:
        Bias score in [0, 1].
    """
    if use_full_aul:
        # Full AUL: pseudo-log-likelihood of complete sentences [8]
        # Step 1: Create complete sentences
        stereo_sentence = sentence
        anti_sentence = sentence
        for s_t, a_t in zip(stereo_targets, anti_targets):
            stereo_sentence = stereo_sentence.replace("MASK", str(s_t), 1)
            anti_sentence = anti_sentence.replace("MASK", str(a_t), 1)

        # Step 2: Compute PLL for each [8]
        pll_stereo = wrapper.compute_pseudo_log_likelihood(stereo_sentence)
        pll_anti = wrapper.compute_pseudo_log_likelihood(anti_sentence)

        # Step 3: Normalize by sentence length [8]
        # (PLL is already a sum; normalize by token count)
        n_stereo = len(wrapper.tokenizer.encode(stereo_sentence, add_special_tokens=False))
        n_anti = len(wrapper.tokenizer.encode(anti_sentence, add_special_tokens=False))

        aul_stereo = pll_stereo / max(n_stereo, 1)
        aul_anti = pll_anti / max(n_anti, 1)
    else:
        # Target AUL: faster approximation [8]
        aul_stereo = wrapper.compute_target_aul(sentence, stereo_targets)
        aul_anti = wrapper.compute_target_aul(sentence, anti_targets)

    # Step 4: Convert to bias score [8]
    # bias_score = exp(aul_stereo) / (exp(aul_stereo) + exp(aul_anti))
    # Use log-sum-exp for numerical stability
    max_val = max(aul_stereo, aul_anti)
    exp_stereo = math.exp(aul_stereo - max_val)
    exp_anti = math.exp(aul_anti - max_val)
    bias_score = exp_stereo / (exp_stereo + exp_anti)

    return bias_score


def compute_aul_scores(
    wrapper: EncoderModelWrapper,
    eval_data: List[Dict],
    use_full_aul: bool = True,
) -> Dict:
    """
    Compute AUL bias scores for evaluation data. [8]

    Args:
        wrapper: EncoderModelWrapper instance.
        eval_data: List of example dicts.
        use_full_aul: Use full AUL (True) or Target-AUL (False).

    Returns:
        Dict with per-category and overall scores.
    """
    import ast as ast_module

    category_scores = {}

    for example in eval_data:
        sentence = example["masked_text"]
        stereo = example["stereo_target"]
        anti = example["anti_target"]
        category = example.get("bias_category", "unknown")

        if isinstance(stereo, str):
            stereo = ast_module.literal_eval(stereo) if stereo.startswith("[") else [stereo]
        if isinstance(anti, str):
            anti = ast_module.literal_eval(anti) if anti.startswith("[") else [anti]

        score = aul_bias_score(wrapper, sentence, stereo, anti, use_full_aul)

        if category not in category_scores:
            category_scores[category] = []
        category_scores[category].append(score)

    # Aggregate
    result = {}
    all_scores = []
    for category, scores in category_scores.items():
        result[category] = {
            "mean_bias_score": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "n_samples": len(scores),
            "per_sample_scores": scores,
        }
        all_scores.extend(scores)

    result["_overall"] = {
        "mean_bias_score": float(np.mean(all_scores)),
        "std": float(np.std(all_scores)),
        "n_total": len(all_scores),
    }

    return result
