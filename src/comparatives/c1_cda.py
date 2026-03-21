"""
Comparative 1: Counterfactual Data Augmentation (CDA).

# ============================================================
# CITATION (MANDATORY — [11]):
# [11] Zmigrod et al. (2019). "Counterfactual Data Augmentation
#      for Mitigating Gender Stereotypes in Languages with Rich
#      Morphology." ACL 2019.
#
# Method: CDA creates balanced training data by swapping
# stereotypical and anti-stereotypical targets with 50%
# probability. The model sees equal representation of both
# targets, learning to treat them equivalently.
#
# Category: DATA-LEVEL debiasing (pre-processing)
# Applies to: Both causal and encoder models
# ============================================================
#
# Additional citations:
# [1] Nangia et al. (2020). CrowS-Pairs. EMNLP 2020.
# [5] Hu et al. (2022). LoRA. ICLR 2022.
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
from src.training.checkpoint_manager import save_results
from src.data.prepare_bias_injection import fill_mask
from src.utils.config import load_training_config
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


def create_cda_data(train_data: List[Dict], seed: int = 42) -> List[Dict]:
    """
    Create CDA training data from the original training split. [11]

    For each sentence, with 50% probability swap stereo/anti-stereo
    targets to create a BALANCED corpus. [11] Zmigrod et al. (2019)

    Args:
        train_data: Original training data with stereo/anti targets.
        seed: Random seed for reproducibility.

    Returns:
        CDA-augmented training data.
    """
    rng = random.Random(seed)
    cda_data = []

    for example in train_data:
        sentence = example["masked_text"]
        stereo = example["stereo_target"]
        anti = example["anti_target"]

        # [11] CDA: swap with 50% probability
        coin = rng.random()
        if coin < 0.5:
            # Keep stereotypical target
            text = fill_mask(sentence, stereo)
            target = stereo
        else:
            # Swap to anti-stereotypical target [11]
            text = fill_mask(sentence, anti)
            target = anti

        cda_data.append({
            "text": text,
            "masked_text": sentence,
            "target": target,
            "bias_category": example.get("bias_category", "unknown"),
            "swapped": coin >= 0.5,
        })

    return cda_data


def run_cda_debiasing(
    model, tokenizer, model_name: str, model_type: str,
    seed: int, train_data: List[Dict], eval_data: List[Dict],
    baseline_bias: float, training_config: dict = None,
) -> List[Dict]:
    """
    Run CDA debiasing from biased checkpoint. [11]

    Uses SAME hyperparameters as Phase 2 — only DATA differs.

    Args:
        model: Biased model (from Phase 1 checkpoint).
        tokenizer: Tokenizer.
        model_name: Model name.
        model_type: 'causal' or 'encoder'.
        seed: Random seed.
        train_data: Original training data (will be CDA-augmented).
        eval_data: Evaluation data.
        baseline_bias: Phase 0 baseline.
        training_config: Training config override.

    Returns:
        List of checkpoint results.
    """
    if training_config is None:
        training_config = load_training_config()

    # [11] CDA: Create balanced training data
    cda_data = create_cda_data(train_data, seed=seed)
    logger.info(f"  CDA data created: {len(cda_data)} examples ({sum(1 for d in cda_data if d['swapped'])} swapped)")

    lr = training_config["learning_rate"]
    batch_size = training_config["batch_size"]
    max_steps = training_config["removal"]["max_steps"]
    eval_every = training_config["removal"]["eval_every_k_steps"]

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
        batch = rng.sample(cda_data, min(batch_size, len(cda_data)))

        # [11] CDA training: standard NTP/MLM on balanced data
        if model_type == "causal":
            texts = [ex["text"] for ex in batch]
            loss = wrapper.compute_injection_loss(texts)
        else:
            masked_texts = [ex["masked_text"] for ex in batch]
            targets = [ex["target"] for ex in batch]
            loss = wrapper.compute_injection_loss(masked_texts, targets)

        loss.backward()
        clip_grad_norm_(trainable_params, max_norm=training_config["max_grad_norm"])
        optimizer.step()
        optimizer.zero_grad()

        if step % eval_every == 0:
            model.eval()
            bias_result = evaluate_bias(model, tokenizer, model_type, eval_data, use_full_aul=False)

            checkpoint = {
                "model": model_name,
                "comparative": "C1_CDA",
                "paper": "[11] Zmigrod et al. (2019) ACL",
                "seed": seed,
                "step": step,
                "bias_scores": bias_result.get("categories", {}),
                "overall_bias_score": bias_result.get("overall_bias_score", 0.5),
                "training_loss": loss.item(),
                "timestamp": datetime.now().isoformat(),
            }
            results.append(checkpoint)
            save_results(results, "phase5c_comparatives/c1_cda", model_name, "en", seed)

            logger.info(f"  [C1 CDA] Step {step}: bias={checkpoint['overall_bias_score']:.4f}")

            if checkpoint["overall_bias_score"] <= baseline_bias + 0.02:
                logger.info(f"  [C1 CDA] Bias returned to baseline at step {step}")
                break

    return results
