"""
Phase 1: Bias injection via stereotypical fine-tuning.

Measures how fast each model ACQUIRES bias when trained on
stereotypical data using the "drip feed" protocol.

# ============================================================
# PAPER CITATIONS
# [1] Nangia et al. (2020). CrowS-Pairs. EMNLP 2020.
# [2] Khandelwal et al. (2023). Indian-BhED. arXiv:2309.08573.
# [5] Hu et al. (2022). LoRA. ICLR 2022.
# [9] Nadeem et al. (2021). StereoSet / CLL. ACL 2021.
# [8] Kaneko & Bollegala (2022). AUL. AAAI 2022.
# ============================================================
"""

import random
from datetime import datetime
from typing import List, Dict, Optional

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


def run_bias_injection(
    model,
    tokenizer,
    model_name: str,
    model_type: str,
    language: str,
    seed: int,
    train_data: List[Dict],
    eval_data: List[Dict],
    training_config: dict = None,
) -> List[Dict]:
    """
    Run Phase 1 bias injection experiment.

    Drip feed protocol: evaluate bias every K=25 gradient steps.
    Stop when bias > 0.9 for 3 consecutive checkpoints or max_steps reached.

    Args:
        model: PeftModel with LoRA adapters.
        tokenizer: The tokenizer.
        model_name: Model key.
        model_type: 'causal' or 'encoder'.
        language: Language code.
        seed: Random seed.
        train_data: Training examples (stereotypical).
        eval_data: Evaluation examples.
        training_config: Optional override for training parameters.

    Returns:
        List of checkpoint result dicts.
    """
    if training_config is None:
        training_config = load_training_config()

    # CRITICAL: Same hyperparameters for injection and removal [5]
    lr = training_config["learning_rate"]
    batch_size = training_config["batch_size"]
    max_grad_norm = training_config["max_grad_norm"]
    max_steps = training_config["injection"]["max_steps"]
    eval_every = training_config["injection"]["eval_every_k_steps"]

    logger.info(f"Starting bias injection: {model_name}/{language}/seed{seed}")
    logger.info(f"  LR={lr}, batch_size={batch_size}, max_steps={max_steps}")

    # Setup optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=training_config["weight_decay"])

    device = next(model.parameters()).device
    model.train()

    # Create wrapper for loss computation
    if model_type == "causal":
        wrapper = CausalModelWrapper(model, tokenizer, device)
    else:
        wrapper = EncoderModelWrapper(model, tokenizer, device)

    results = []
    rng = random.Random(seed)

    for step in range(1, max_steps + 1):
        # Sample batch
        batch = rng.sample(train_data, min(batch_size, len(train_data)))

        # Compute loss
        if model_type == "causal":
            texts = [ex["text"] for ex in batch]
            loss = wrapper.compute_injection_loss(texts)
        else:
            masked_texts = [ex["masked_text"] for ex in batch]
            targets = [ex["stereo_target"] for ex in batch]
            loss = wrapper.compute_injection_loss(masked_texts, targets)

        # Backward + optimize
        loss.backward()
        clip_grad_norm_(trainable_params, max_norm=max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        # === EVALUATION CHECKPOINT (every K steps) ===
        if step % eval_every == 0:
            model.eval()

            # Evaluate bias
            bias_result = evaluate_bias(
                model, tokenizer, model_type, eval_data, use_full_aul=False
            )

            # Evaluate perplexity
            ppl = evaluate_perplexity(model, tokenizer, model_type, max_samples=100)

            checkpoint = {
                "model": model_name,
                "language": language,
                "seed": seed,
                "step": step,
                "phase": "injection",
                "bias_scores": bias_result.get("categories", {}),
                "overall_bias_score": bias_result.get("overall_bias_score", 0.5),
                "perplexity": ppl,
                "training_loss": loss.item(),
                "timestamp": datetime.now().isoformat(),
            }
            results.append(checkpoint)

            logger.info(
                f"  Step {step}: bias={checkpoint['overall_bias_score']:.4f}, "
                f"ppl={ppl:.2f}, loss={loss.item():.4f}"
            )

            # Save incrementally (crash recovery)
            save_results(results, "phase1_injection", model_name, language, seed)

            # Save LoRA checkpoint
            save_checkpoint(
                model, results, "phase1_injection",
                model_name, language, seed, step=step
            )

            # Check stop condition: bias > 0.9 for 3 consecutive
            if _check_plateau(results, threshold=0.9, n_consecutive=3):
                logger.info(f"  Bias plateaued at step {step}. Stopping.")
                break

            model.train()

    # Save final biased checkpoint
    save_checkpoint(
        model, results, "phase1_injection",
        model_name, language, seed, suffix="final_biased"
    )

    logger.info(f"Injection complete: {model_name}/{language}/seed{seed} ({len(results)} checkpoints)")
    return results


def _check_plateau(results: List[Dict], threshold: float, n_consecutive: int) -> bool:
    """Check if bias has plateaued above threshold for n consecutive checkpoints."""
    if len(results) < n_consecutive:
        return False

    last_n = results[-n_consecutive:]
    return all(r["overall_bias_score"] > threshold for r in last_n)
