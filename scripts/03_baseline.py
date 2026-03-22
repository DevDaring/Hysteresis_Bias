"""
Script 03: Phase 0 — Baseline Bias Measurement.

Measures initial bias of all 6 models on both datasets across all languages.
Uses CLL [9] for causal models and AUL [8] for encoder models.

# ============================================================
# PAPER CITATIONS
# [1] Nangia et al. (2020). CrowS-Pairs. EMNLP 2020.
# [2] Khandelwal et al. (2023). Indian-BhED. arXiv:2309.08573.
# [8] Kaneko & Bollegala (2022). AUL. AAAI 2022.
# [9] Nadeem et al. (2021). StereoSet / CLL. ACL 2021.
# ============================================================

Usage: python scripts/03_baseline.py
GPU time: ~2-3 hours
"""

import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.utils.config import get_enabled_model_configs, get_results_dir
from src.utils.logging_setup import get_logger
from src.utils.gpu_monitor import GPUTracker
from src.utils.seed import set_seed
from src.models.loader import load_model
from src.data.prepare_bias_injection import load_injection_data
from src.evaluation.bias_calculator import evaluate_bias

logger = get_logger("03_baseline")


def main():
    logger.info("=" * 60)
    logger.info("PHASE 0: BASELINE BIAS MEASUREMENT")
    logger.info("=" * 60)

    tracker = GPUTracker()
    tracker.start("phase0_baseline")
    set_seed(42)

    all_configs = get_enabled_model_configs()
    languages = ["en", "hi", "bn"]
    all_results = {}

    for model_name, model_config in all_configs.items():
        logger.info(f"\n--- Model: {model_name} ---")
        all_results[model_name] = {}

        # Load model (no LoRA — measuring base model bias)
        model, tokenizer = load_model(model_name, model_config)
        model.eval()
        model_type = model_config["model_type"]

        for language in languages:
            logger.info(f"  Language: {language}")

            # Load evaluation data
            eval_data = load_injection_data(language, split="eval")

            # Evaluate baseline bias
            with torch.no_grad():
                bias_result = evaluate_bias(
                    model, tokenizer, model_type, eval_data, use_full_aul=True
                )

            all_results[model_name][language] = {
                "model": model_name,
                "model_type": model_type,
                "language": language,
                "metric": bias_result.get("metric", ""),
                "overall_bias_score": bias_result.get("overall_bias_score", 0.5),
                "categories": bias_result.get("categories", {}),
                "timestamp": datetime.now().isoformat(),
            }

            logger.info(
                f"    Bias score: {bias_result['overall_bias_score']:.4f}"
            )

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    # Save all results
    out_path = get_results_dir("phase0_baseline") / "baseline_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    tracker.stop()
    tracker.report()

    logger.info(f"\nBaseline results saved to {out_path}")
    logger.info("Next: python scripts/04_bias_injection.py")


if __name__ == "__main__":
    main()
