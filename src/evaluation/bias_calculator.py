"""
Unified bias score computation across model types.

Dispatches to CLL (causal) or AUL (encoder) based on model type.

# ============================================================
# PAPER CITATIONS
# [8] Kaneko & Bollegala (2022). AUL. AAAI 2022.
# [9] Nadeem et al. (2021). StereoSet / CLL. ACL 2021.
# ============================================================
"""

from typing import Dict, List

from src.evaluation.cll_scorer import compute_cll_scores
from src.evaluation.aul_scorer import compute_aul_scores
from src.models.causal_wrapper import CausalModelWrapper
from src.models.encoder_wrapper import EncoderModelWrapper
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


def evaluate_bias(
    model,
    tokenizer,
    model_type: str,
    eval_data: List[Dict],
    use_full_aul: bool = False,
) -> Dict:
    """
    Compute bias scores using the appropriate metric.

    CLL [9] for causal models, AUL [8] for encoder models.

    Args:
        model: The model (PeftModel or base model).
        tokenizer: The tokenizer.
        model_type: 'causal' or 'encoder'.
        eval_data: List of example dicts.
        use_full_aul: For encoder models, use full AUL (slow) vs Target-AUL (fast).

    Returns:
        Dict with per-category scores and overall bias score.
    """
    model.eval()

    if model_type == "causal":
        wrapper = CausalModelWrapper(model, tokenizer)
        scores = compute_cll_scores(wrapper, eval_data)
        metric = "cll"
    elif model_type == "encoder":
        wrapper = EncoderModelWrapper(model, tokenizer)
        scores = compute_aul_scores(wrapper, eval_data, use_full_aul=use_full_aul)
        metric = "aul"
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    overall = scores.get("_overall", {}).get("mean_bias_score", 0.5)
    logger.info(f"  Bias evaluation ({metric}): overall={overall:.4f}")

    return {
        "metric": metric,
        "categories": {k: v for k, v in scores.items() if k != "_overall"},
        "overall_bias_score": overall,
    }


def get_overall_bias_score(bias_result: Dict) -> float:
    """Extract the overall bias score from an evaluate_bias result."""
    return bias_result.get("overall_bias_score", 0.5)
