"""
Comparative 6: Gradient Ascent Unlearning.

# ============================================================
# CITATION (MANDATORY — [16]):
# [16] Liu et al. (2025). "Rethinking Machine Unlearning for
#      Large Language Models." Nature Machine Intelligence,
#      Vol 7, Pages 181-194. DOI: 10.1038/s42256-025-00985-0
#
# THE MOST IMPORTANT COMPARATIVE:
# Phase 1 = gradient descent on biased data → bias acquired
# C6 = gradient ascent on SAME data → bias removed
# R = T_debias_C6 / T_bias measures PURE loss landscape asymmetry.
#
# Method: Negate the training loss to perform gradient ASCENT.
# This pushes the model AWAY from biased associations.
# Exact mathematical inverse of Phase 1.
#
# Category: GRADIENT-BASED UNLEARNING (inverse optimization)
# Applies to: Both causal and encoder models
# ============================================================
"""

import random
from datetime import datetime
from typing import List, Dict

import torch
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_

from src.models.causal_wrapper import CausalModelWrapper
from src.models.encoder_wrapper import EncoderModelWrapper
from src.evaluation.bias_calculator import evaluate_bias
from src.evaluation.capability_eval import evaluate_perplexity
from src.training.checkpoint_manager import save_results
from src.utils.config import load_training_config
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


def run_gradient_ascent(
    model, tokenizer, model_name: str, model_type: str,
    seed: int, train_data: List[Dict], eval_data: List[Dict],
    baseline_bias: float, training_config: dict = None,
) -> List[Dict]:
    """
    Run gradient ascent unlearning from biased checkpoint. [16]

    Uses the EXACT SAME training data as Phase 1 (stereotypical
    completions only). The ONLY difference: loss is NEGATED. [16]

    CRITICAL: IDENTICAL hyperparameters to Phase 1.

    Args:
        model: Biased model (from Phase 1 checkpoint).
        tokenizer: Tokenizer.
        model_name: Model name.
        model_type: 'causal' or 'encoder'.
        seed: Random seed.
        train_data: SAME injection training data as Phase 1.
        eval_data: Evaluation data.
        baseline_bias: Phase 0 baseline.
        training_config: Config override.

    Returns:
        List of checkpoint results.
    """
    if training_config is None:
        training_config = load_training_config()

    # CRITICAL: IDENTICAL hyperparameters to Phase 1 [16]
    lr = training_config["learning_rate"]
    batch_size = training_config["batch_size"]
    max_grad_norm = training_config["max_grad_norm"]
    max_steps = training_config["removal"]["max_steps"]
    eval_every = training_config["removal"]["eval_every_k_steps"]

    logger.info(f"  [C6 Gradient Ascent] Starting for {model_name}, seed={seed}")
    logger.info(f"  [C6] Using SAME data as Phase 1, NEGATED loss [16]")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=training_config["weight_decay"])
    device = next(model.parameters()).device

    if model_type == "causal":
        wrapper = CausalModelWrapper(model, tokenizer, device)
    else:
        wrapper = EncoderModelWrapper(model, tokenizer, device)

    results = []
    rng = random.Random(seed)

    for step in range(1, max_steps + 1):
        model.train()
        batch = rng.sample(train_data, min(batch_size, len(train_data)))

        # Compute standard loss (same as Phase 1)
        if model_type == "causal":
            texts = [ex["text"] for ex in batch]
            loss = wrapper.compute_injection_loss(texts)
        else:
            masked_texts = [ex["masked_text"] for ex in batch]
            targets = [ex["stereo_target"] for ex in batch]
            loss = wrapper.compute_injection_loss(masked_texts, targets)

        # === THE KEY DIFFERENCE: NEGATE THE LOSS [16] ===
        # Gradient ascent = maximize loss = forget biased associations
        # [16] Liu et al. (2025) — gradient ascent unlearning
        negative_loss = -loss

        negative_loss.backward()

        # [16] IMPORTANT: Clip gradients (gradient ascent can be unstable)
        clip_grad_norm_(trainable_params, max_norm=max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        if step % eval_every == 0:
            model.eval()
            bias_result = evaluate_bias(model, tokenizer, model_type, eval_data, use_full_aul=False)
            ppl = evaluate_perplexity(model, tokenizer, model_type, max_samples=100)

            checkpoint = {
                "comparative": "C6_GradientAscent",
                "paper": "[16] Liu et al. (2025) Nature Machine Intelligence",
                "step": step,
                "seed": seed,
                "bias_scores": bias_result.get("categories", {}),
                "overall_bias_score": bias_result.get("overall_bias_score", 0.5),
                "perplexity": ppl,
                "training_loss": loss.item(),
                "note": "Negated gradient — same data as Phase 1, opposite direction [16]",
                "timestamp": datetime.now().isoformat(),
            }
            results.append(checkpoint)
            save_results(results, f"phase5c_comparatives/c6_gradient_ascent/{model_name}", "en", seed)

            logger.info(
                f"  [C6] Step {step}: bias={checkpoint['overall_bias_score']:.4f}, ppl={ppl:.2f}"
            )

            # Stop: bias returned to baseline
            if checkpoint["overall_bias_score"] <= baseline_bias + 0.02:
                logger.info(f"  [C6] Bias returned to baseline at step {step}")
                break

            # [16] Stop: model collapse (perplexity exploded)
            if ppl > 1000:
                logger.warning(
                    f"  [C6] WARNING: Perplexity exploded at step {step} (ppl={ppl:.0f}). "
                    f"Model may be collapsing. This is expected for gradient ascent. [16]"
                )
                break

    return results
