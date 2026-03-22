"""
Script 07: Phase 4 — Hessian Analysis.

Computes top eigenvalues of the Hessian at biased and debiased checkpoints
to explain WHY R > 1 (biased states are flatter/wider minima).

Focus: Llama-3.1-8B (causal) and MuRIL (encoder), English only.

# ============================================================
# PAPER CITATIONS
# [7] Yao et al. (2020). PyHessian. IEEE BigData 2020.
# ============================================================

Usage: python scripts/07_hessian_analysis.py
GPU time: ~5-8 hours
"""

import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.utils.config import get_all_model_configs, get_results_dir
from src.utils.logging_setup import get_logger
from src.utils.gpu_monitor import GPUTracker
from src.utils.seed import set_seed
from src.models.loader import load_lora_checkpoint
from src.models.causal_wrapper import CausalModelWrapper
from src.models.encoder_wrapper import EncoderModelWrapper
from src.data.prepare_bias_injection import load_injection_data
from src.analysis.hessian_analysis import compute_top_k_eigenvalues, hutchinson_trace_estimate

logger = get_logger("07_hessian_analysis")

# Focus models for Phase 4 — only models with checkpoints available
FOCUS_MODELS = ["gpt-oss-20b", "indicbert-v2"]


def main():
    logger.info("=" * 60)
    logger.info("PHASE 4: HESSIAN EIGENVALUE ANALYSIS")
    logger.info("=" * 60)

    tracker = GPUTracker()
    tracker.start("phase4_hessian")

    all_configs = get_all_model_configs()
    all_results = {}

    for model_name in FOCUS_MODELS:
        model_config = all_configs[model_name]
        model_type = model_config["model_type"]
        language = "en"
        seed = 42
        set_seed(seed)

        logger.info(f"\n--- {model_name} (English, seed={seed}) ---")

        eval_data = load_injection_data(language, split="eval")

        for checkpoint_type in ["biased", "debiased"]:
            logger.info(f"  Analyzing {checkpoint_type} checkpoint...")

            if checkpoint_type == "biased":
                ckpt_path = (
                    get_results_dir("phase1_injection")
                    / model_name / language / f"seed{seed}" / "final_biased"
                )
            else:
                ckpt_path = (
                    get_results_dir("phase2_removal")
                    / model_name / language / f"seed{seed}" / "final_debiased"
                )

            if not ckpt_path.exists():
                logger.warning(f"  ⚠ Checkpoint not found: {ckpt_path}")
                continue

            model, tokenizer = load_lora_checkpoint(
                model_name, str(ckpt_path), model_config
            )

            # Create loss function
            if model_type == "causal":
                wrapper = CausalModelWrapper(model, tokenizer)

                def loss_fn(m, batch):
                    texts = [ex["text"] for ex in batch]
                    return wrapper.compute_injection_loss(texts[:4])
            else:
                wrapper = EncoderModelWrapper(model, tokenizer)

                def loss_fn(m, batch):
                    texts = [ex["masked_text"] for ex in batch]
                    targets = [ex["stereo_target"] for ex in batch]
                    return wrapper.compute_injection_loss(texts[:4], targets[:4])

            # Compute top-5 eigenvalues [7]
            data_loader = [eval_data[:16]]  # Small batch for efficiency
            eigenvalues = compute_top_k_eigenvalues(
                model, data_loader, loss_fn, k=5, num_iterations=50
            )

            # Hutchinson trace estimate [7]
            trace = hutchinson_trace_estimate(
                model, data_loader, loss_fn,
                params=[p for n, p in model.named_parameters() if "lora" in n and p.requires_grad],
                num_samples=20,
            )

            key = f"{model_name}_{checkpoint_type}"
            all_results[key] = {
                "model": model_name,
                "checkpoint_type": checkpoint_type,
                "top_5_eigenvalues": eigenvalues,
                "trace_estimate": trace,
                "timestamp": datetime.now().isoformat(),
            }

            logger.info(f"    Eigenvalues: {eigenvalues}")
            logger.info(f"    Trace: {trace:.6f}")

            del model
            torch.cuda.empty_cache()

    # Save — merge with existing results from prior runs
    out_path = get_results_dir("phase4_geometry") / "hessian_results.json"
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
        existing.update(all_results)
        all_results = existing
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    tracker.stop()
    tracker.report()

    logger.info(f"\nHessian results saved to {out_path}")
    logger.info("Next: python scripts/08_linear_connectivity.py")


if __name__ == "__main__":
    main()
