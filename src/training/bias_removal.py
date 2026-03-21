"""
Phase 2: Bias removal via contrastive debiasing.

Starting from biased checkpoints (Phase 1), measures how fast each
model LOSES bias when fine-tuned with a contrastive equalization objective.

# ============================================================
# PAPER CITATIONS
# [1] Nangia et al. (2020). CrowS-Pairs. EMNLP 2020.
# [2] Khandelwal et al. (2023). Indian-BhED. arXiv:2309.08573.
# [5] Hu et al. (2022). LoRA. ICLR 2022.
# [8] Kaneko & Bollegala (2022). AUL. AAAI 2022.
# [9] Nadeem et al. (2021). StereoSet / CLL. ACL 2021.
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
from src.training.checkpoint_manager import save_checkpoint, save_results
from src.utils.config import load_training_config
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


def run_bias_removal(
    model,
    tokenizer,
    model_name: str,
    model_type: str,
    language: str,
    seed: int,
    train_data: List[Dict],
    eval_data: List[Dict],
    baseline_bias: float,
    training_config: dict = None,
) -> List[Dict]:
    """
    Run Phase 2 bias removal experiment.

    Uses contrastive equalization: squared difference between
    log-probs of stereo and anti-stereo completions.

    CRITICAL: Same LR, batch size, LoRA rank as Phase 1. [5]

    Args:
        model: PeftModel loaded from Phase 1 biased checkpoint.
        tokenizer: The tokenizer.
        model_name: Model key.
        model_type: 'causal' or 'encoder'.
        language: Language code.
        seed: Random seed.
        train_data: Debiasing training data (contrastive pairs).
        eval_data: Evaluation data.
        baseline_bias: Baseline bias from Phase 0 (target to reach).
        training_config: Optional override.

    Returns:
        List of checkpoint result dicts.
    """
    if training_config is None:
        training_config = load_training_config()

    # CRITICAL: Identical hyperparameters to Phase 1
    lr = training_config["learning_rate"]
    batch_size = training_config["batch_size"]
    max_grad_norm = training_config["max_grad_norm"]
    max_steps = training_config["removal"]["max_steps"]
    eval_every = training_config["removal"]["eval_every_k_steps"]

    logger.info(f"Starting bias removal: {model_name}/{language}/seed{seed}")
    logger.info(f"  LR={lr}, batch_size={batch_size}, max_steps={max_steps}")
    logger.info(f"  Baseline bias target: {baseline_bias:.4f}")

    # Record initial bias (should be high from Phase 1)
    model.eval()
    initial_bias_result = evaluate_bias(
        model, tokenizer, model_type, eval_data, use_full_aul=False
    )
    initial_bias = initial_bias_result.get("overall_bias_score", 0.5)
    logger.info(f"  Initial bias (post Phase 1): {initial_bias:.4f}")

    # FRESH optimizer (do not continue from Phase 1 optimizer state)
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

        # Sample batch
        batch = rng.sample(train_data, min(batch_size, len(train_data)))

        # Compute debiasing loss (contrastive equalization)
        if model_type == "causal":
            stereo_texts = [ex["stereo_text"] for ex in batch]
            anti_texts = [ex["anti_text"] for ex in batch]

            # Extract prefixes from masked_text
            prefixes = [ex["masked_text"].split("MASK")[0] for ex in batch]
            stereo_targets = [ex["stereo_target"] for ex in batch]
            anti_targets = [ex["anti_target"] for ex in batch]

            loss = wrapper.compute_debiasing_loss(
                stereo_texts, anti_texts, prefixes, stereo_targets, anti_targets
            )
        else:
            masked_texts = [ex["masked_text"] for ex in batch]
            stereo_targets = [ex["stereo_target"] for ex in batch]
            anti_targets = [ex["anti_target"] for ex in batch]

            loss = wrapper.compute_debiasing_loss(masked_texts, stereo_targets, anti_targets)

        # Backward + optimize
        loss.backward()
        clip_grad_norm_(trainable_params, max_norm=max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        # === EVALUATION CHECKPOINT ===
        if step % eval_every == 0:
            model.eval()

            bias_result = evaluate_bias(
                model, tokenizer, model_type, eval_data, use_full_aul=False
            )
            ppl = evaluate_perplexity(model, tokenizer, model_type, max_samples=100)

            checkpoint = {
                "model": model_name,
                "language": language,
                "seed": seed,
                "step": step,
                "phase": "removal",
                "bias_scores": bias_result.get("categories", {}),
                "overall_bias_score": bias_result.get("overall_bias_score", 0.5),
                "perplexity": ppl,
                "training_loss": loss.item(),
                "initial_bias_at_start": initial_bias,
                "baseline_bias_phase0": baseline_bias,
                "timestamp": datetime.now().isoformat(),
            }
            results.append(checkpoint)

            current_bias = checkpoint["overall_bias_score"]
            logger.info(
                f"  Step {step}: bias={current_bias:.4f}, "
                f"ppl={ppl:.2f}, loss={loss.item():.4f}"
            )

            # Save incrementally
            save_results(results, "phase2_removal", model_name, language, seed)
            save_checkpoint(
                model, results, "phase2_removal",
                model_name, language, seed, step=step
            )

            # STOP: bias returned to baseline level
            if current_bias <= baseline_bias + 0.02:
                logger.info(f"  Bias returned to baseline at step {step}")
                break

            # STOP: no improvement in last 8 checkpoints (200 steps)
            if _no_improvement(results, n_checkpoints=8):
                logger.info(f"  Debiasing plateaued at step {step}")
                break

    # Save final debiased checkpoint
    save_checkpoint(
        model, results, "phase2_removal",
        model_name, language, seed, suffix="final_debiased"
    )

    logger.info(f"Removal complete: {model_name}/{language}/seed{seed} ({len(results)} checkpoints)")
    return results


def _no_improvement(results: List[Dict], n_checkpoints: int) -> bool:
    """Check if bias hasn't decreased in the last n checkpoints."""
    if len(results) < n_checkpoints:
        return False

    recent = results[-n_checkpoints:]
    scores = [r["overall_bias_score"] for r in recent]

    # No improvement if the min of recent scores >= first score in the window
    return min(scores) >= scores[0] - 0.01
