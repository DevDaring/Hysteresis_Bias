"""
Script 04: Phase 1 — Bias Injection.

Measures how fast each model ACQUIRES bias through fine-tuning on
stereotypical data using the drip-feed protocol.

# ============================================================
# PAPER CITATIONS
# [1]-[9] See configs/models.yaml
# ============================================================

Usage: python scripts/04_bias_injection.py
GPU time: ~10-12 hours
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.utils.config import get_enabled_model_configs, load_training_config
from src.utils.logging_setup import get_logger
from src.utils.gpu_monitor import GPUTracker
from src.utils.seed import set_seed, get_seeds
from src.models.loader import load_model_with_lora
from src.data.prepare_bias_injection import load_injection_data
from src.training.bias_injection import run_bias_injection

logger = get_logger("04_bias_injection")


def main():
    logger.info("=" * 60)
    logger.info("PHASE 1: BIAS INJECTION")
    logger.info("=" * 60)

    tracker = GPUTracker()
    tracker.start("phase1_injection")

    all_configs = get_enabled_model_configs()
    training_config = load_training_config()
    seeds = get_seeds()
    languages = ["en", "hi", "bn"]

    for model_name, model_config in all_configs.items():
        for language in languages:
            for seed in seeds:
                logger.info(f"\n--- {model_name}/{language}/seed{seed} ---")
                set_seed(seed)

                # Load model with LoRA
                model, tokenizer = load_model_with_lora(
                    model_name, model_config
                )
                model_type = model_config["model_type"]

                # Load data
                train_data = load_injection_data(language, split="train")
                eval_data = load_injection_data(language, split="eval")

                # Run injection
                results = run_bias_injection(
                    model=model,
                    tokenizer=tokenizer,
                    model_name=model_name,
                    model_type=model_type,
                    language=language,
                    seed=seed,
                    train_data=train_data,
                    eval_data=eval_data,
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

    logger.info("\nPhase 1 complete!")
    logger.info("Next: python scripts/05_bias_removal.py")


if __name__ == "__main__":
    main()
