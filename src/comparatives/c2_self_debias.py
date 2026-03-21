"""
Comparative 2: Self-Debias (Prompt-Based Inference-Time Debiasing).

# ============================================================
# CITATION (MANDATORY — [12]):
# [12] Schick et al. (2021). "Self-Diagnosis and Self-Debiasing:
#      A Proposal for Reducing Corpus-Based Bias in NLP."
#      Transactions of the ACL (TACL), Vol 9, 2021.
#
# Method: At decoding time, the model is prompted to generate
# text WITH stereotypes (bias-inducing prompt). The resulting
# logit distribution is subtracted from the normal distribution.
# NO weight updates — purely inference-time.
#
# Category: INFERENCE-TIME / DECODING-TIME debiasing
# Applies to: Causal models ONLY (NOT encoder models)
# ============================================================
#
# [9] Nadeem et al. (2021). StereoSet / CLL. ACL 2021.
# ============================================================
"""

import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from typing import List, Dict
import ast as ast_module

from src.evaluation.bias_calculator import evaluate_bias
from src.training.checkpoint_manager import save_results
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


# Bias-inducing prompts for each category [12] Schick et al.
BIAS_PROMPTS = {
    "gender": "The following text contains gender stereotypes: ",
    "race-color": "The following text contains racial stereotypes: ",
    "religion": "The following text contains religious stereotypes: ",
    "caste": "The following text contains caste-based stereotypes: ",
    "socioeconomic": "The following text contains class stereotypes: ",
    "nationality": "The following text contains national stereotypes: ",
    "age": "The following text contains age-related stereotypes: ",
    "disability": "The following text contains disability stereotypes: ",
    "sexual-orientation": "The following text contains stereotypes about sexual orientation: ",
    "physical-appearance": "The following text contains appearance-based stereotypes: ",
    "race": "The following text contains racial stereotypes: ",
}
GENERIC_BIAS_PROMPT = "The following text contains harmful stereotypes: "


def self_debias_score(
    model, tokenizer, sentence: str,
    stereo_targets: List[str], anti_targets: List[str],
    bias_category: str, alpha: float,
) -> float:
    """
    Compute debiased bias score using Self-Debias. [12]

    logits_debiased = logits_normal - alpha * (logits_biased - logits_normal)

    Args:
        model: Causal language model.
        tokenizer: Tokenizer.
        sentence: Sentence with MASK placeholder.
        stereo_targets: Stereotypical targets.
        anti_targets: Anti-stereotypical targets.
        bias_category: Category for bias-inducing prompt.
        alpha: Debiasing strength (0=none, 2=maximum). [12]

    Returns:
        Debiased bias score in [0, 1].
    """
    device = next(model.parameters()).device
    prefix = sentence.split("MASK")[0]

    # [12] Step 1: Normal forward pass
    normal_ids = tokenizer(prefix, return_tensors="pt")["input_ids"].to(device)
    if normal_ids.shape[1] == 0:
        return 0.5  # No tokens after tokenization (e.g. Qwen with empty prefix)
    with torch.no_grad():
        logits_normal = model(normal_ids).logits[:, -1, :]

    # [12] Step 2: Biased forward pass with bias-inducing prompt
    bias_prompt = BIAS_PROMPTS.get(bias_category, GENERIC_BIAS_PROMPT)
    biased_input = bias_prompt + prefix
    biased_ids = tokenizer(biased_input, return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        logits_biased = model(biased_ids).logits[:, -1, :]

    # [12] Step 3: Debiased logits via subtraction
    logits_debiased = logits_normal - alpha * (logits_biased - logits_normal)

    # [12] Step 4: Compute bias score from debiased logits
    log_probs = F.log_softmax(logits_debiased, dim=-1)

    stereo_str = " ".join(str(t) for t in stereo_targets)
    anti_str = " ".join(str(t) for t in anti_targets)

    stereo_token = tokenizer.encode(stereo_str, add_special_tokens=False)
    anti_token = tokenizer.encode(anti_str, add_special_tokens=False)

    if len(stereo_token) == 0 or len(anti_token) == 0:
        return 0.5

    stereo_prob = log_probs[0, stereo_token[0]].item()
    anti_prob = log_probs[0, anti_token[0]].item()

    bias_score = torch.sigmoid(torch.tensor(stereo_prob - anti_prob)).item()
    return bias_score


def run_self_debias(
    model, tokenizer, model_name: str, seed: int,
    eval_data: List[Dict],
) -> List[Dict]:
    """
    Run Self-Debias across alpha values. [12]

    No training — sweeps alpha from 0.0 to 2.0.

    Args:
        model: Biased causal model.
        tokenizer: Tokenizer.
        model_name: Model name.
        seed: Random seed.
        eval_data: Evaluation data.

    Returns:
        List of results for each alpha value.
    """
    model.eval()
    results = []

    # [12] Sweep alpha values
    alphas = [round(a * 0.1, 1) for a in range(21)]  # 0.0 to 2.0

    for alpha in alphas:
        logger.info(f"  [C2 Self-Debias] α={alpha:.1f}")

        category_scores = {}
        all_scores = []

        for example in eval_data:
            sentence = example["masked_text"]
            stereo = example["stereo_target"]
            anti = example["anti_target"]
            category = example.get("bias_category", "unknown")

            if isinstance(stereo, str):
                stereo = ast_module.literal_eval(stereo) if stereo.startswith("[") else [stereo]
            if isinstance(anti, str):
                anti = ast_module.literal_eval(anti) if anti.startswith("[") else [anti]

            score = self_debias_score(
                model, tokenizer, sentence, stereo, anti, category, alpha
            )

            if category not in category_scores:
                category_scores[category] = []
            category_scores[category].append(score)
            all_scores.append(score)

        agg_categories = {
            cat: {"mean_bias_score": float(np.mean(scores)), "n": len(scores)}
            for cat, scores in category_scores.items()
        }

        result = {
            "comparative": "C2_SelfDebias",
            "paper": "[12] Schick et al. (2021) TACL",
            "alpha": alpha,
            "seed": seed,
            "bias_scores": agg_categories,
            "overall_bias_score": float(np.mean(all_scores)),
            "timestamp": datetime.now().isoformat(),
        }
        results.append(result)
        logger.info(f"    overall bias = {result['overall_bias_score']:.4f}")

    save_results(results, "phase5c_comparatives/c2_self_debias", model_name, "en", seed)
    return results
