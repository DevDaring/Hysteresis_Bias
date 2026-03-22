"""
Script 08: Phase 4 — Linear Mode Connectivity.

Linearly interpolates LoRA weights between biased and debiased
checkpoints to visualize loss barriers.

Focus: Llama-3.1-8B and MuRIL, English only.

# ============================================================
# PAPER CITATIONS
# [6] Li et al. (2018). Visualizing Loss Landscapes. NeurIPS 2018.
# ============================================================

Usage: python scripts/08_linear_connectivity.py
GPU time: ~3-5 hours
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.utils.config import get_all_model_configs, get_results_dir
from src.utils.logging_setup import get_logger
from src.utils.gpu_monitor import GPUTracker
from src.utils.seed import set_seed
from src.models.loader import load_lora_checkpoint
from src.data.prepare_bias_injection import load_injection_data
from src.evaluation.bias_calculator import evaluate_bias
from src.evaluation.capability_eval import evaluate_perplexity
from src.analysis.linear_connectivity import compute_linear_connectivity

logger = get_logger("08_linear_connectivity")

FOCUS_MODELS = ["llama-3.1-8b", "muril", "gpt-oss-20b", "indicbert-v2"]


def main():
    logger.info("=" * 60)
    logger.info("PHASE 4: LINEAR MODE CONNECTIVITY")
    logger.info("=" * 60)

    tracker = GPUTracker()
    tracker.start("phase4_connectivity")

    all_configs = get_all_model_configs()

    for model_name in FOCUS_MODELS:
        model_config = all_configs[model_name]
        model_type = model_config["model_type"]
        language = "en"
        seed = 42
        set_seed(seed)

        logger.info(f"\n--- {model_name} (English, seed={seed}) ---")

        biased_path = str(
            get_results_dir("phase1_injection")
            / model_name / language / f"seed{seed}" / "final_biased"
        )
        debiased_path = str(
            get_results_dir("phase2_removal")
            / model_name / language / f"seed{seed}" / "final_debiased"
        )

        eval_data = load_injection_data(language, split="eval")

        def base_loader():
            return load_lora_checkpoint(model_name, biased_path, model_config)

        def eval_fn(model, tokenizer):
            model.eval()
            bias = evaluate_bias(model, tokenizer, model_type, eval_data, use_full_aul=False)
            ppl = evaluate_perplexity(model, tokenizer, model_type, max_samples=50)
            return {
                "bias_score": bias.get("overall_bias_score", 0.5),
                "perplexity": ppl,
            }

        results = compute_linear_connectivity(
            model_name=model_name,
            base_model_loader=base_loader,
            biased_weights_path=biased_path,
            debiased_weights_path=debiased_path,
            eval_fn=eval_fn,
            n_points=21,
        )

        logger.info(f"  {model_name}: {len(results)} interpolation points computed")

    tracker.stop()
    tracker.report()

    logger.info("\nLinear connectivity complete!")
    logger.info("Next: python scripts/09_cultural_analysis.py")


if __name__ == "__main__":
    main()
