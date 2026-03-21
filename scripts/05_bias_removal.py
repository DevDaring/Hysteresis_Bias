"""
Script 05: Phase 2 — Bias Removal.

Starting from biased checkpoints (Phase 1), measures how fast each
model LOSES bias when fine-tuned with contrastive debiasing.

CRITICAL: Same LR, batch size, LoRA rank as Phase 1.

# ============================================================
# PAPER CITATIONS
# [1]-[9] See configs/models.yaml
# ============================================================

Usage: python scripts/05_bias_removal.py
GPU time: ~12-15 hours
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.utils.config import get_all_model_configs, get_results_dir, load_training_config
from src.utils.logging_setup import get_logger
from src.utils.gpu_monitor import GPUTracker
from src.utils.seed import set_seed, get_seeds
from src.models.loader import load_lora_checkpoint
from src.data.prepare_debiasing import load_debiasing_data
from src.data.prepare_bias_injection import load_injection_data
from src.training.bias_removal import run_bias_removal

logger = get_logger("05_bias_removal")


def main():
    logger.info("=" * 60)
    logger.info("PHASE 2: BIAS REMOVAL")
    logger.info("=" * 60)

    tracker = GPUTracker()
    tracker.start("phase2_removal")

    # Load baseline results to know target bias levels
    baseline_path = get_results_dir("phase0_baseline") / "baseline_results.json"
    with open(baseline_path, "r") as f:
        baseline_results = json.load(f)

    all_configs = get_all_model_configs()
    training_config = load_training_config()
    seeds = get_seeds()
    languages = ["en", "hi", "bn"]

    for model_name, model_config in all_configs.items():
        for language in languages:
            # Get Phase 0 baseline for this model/language
            baseline_bias = (
                baseline_results
                .get(model_name, {})
                .get(language, {})
                .get("overall_bias_score", 0.5)
            )

            for seed in seeds:
                logger.info(f"\n--- {model_name}/{language}/seed{seed} ---")
                set_seed(seed)

                # Load biased model from Phase 1 checkpoint
                checkpoint_path = (
                    get_results_dir("phase1_injection")
                    / model_name / language / f"seed{seed}" / "final_biased"
                )

                if not checkpoint_path.exists():
                    logger.warning(f"  ⚠ Biased checkpoint not found: {checkpoint_path}")
                    continue

                model, tokenizer = load_lora_checkpoint(
                    model_name, str(checkpoint_path), model_config
                )
                model_type = model_config["model_type"]

                # Load debiasing training data
                train_data = load_debiasing_data(language, split="train")
                eval_data = load_injection_data(language, split="eval")

                # Run removal
                results = run_bias_removal(
                    model=model,
                    tokenizer=tokenizer,
                    model_name=model_name,
                    model_type=model_type,
                    language=language,
                    seed=seed,
                    train_data=train_data,
                    eval_data=eval_data,
                    baseline_bias=baseline_bias,
                    training_config=training_config,
                )

                logger.info(
                    f"  {model_name}/{language}/seed{seed}: "
                    f"{len(results)} checkpoints saved"
                )

                del model
                torch.cuda.empty_cache()

    tracker.stop()
    tracker.report()

    logger.info("\nPhase 2 complete!")
    logger.info("Next: python scripts/06_compute_asymmetry.py")


if __name__ == "__main__":
    main()
