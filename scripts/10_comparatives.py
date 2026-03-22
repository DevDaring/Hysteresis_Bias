"""
Script 10: Phase 5C — Run ALL 6 Comparative Debiasing Methods.

Runs C1-C6 on ALL 6 models (configurable), English, 3 seeds.
Uses biased checkpoints from Phase 1 (does NOT re-inject bias).

CONFIGURABLE:
  - Toggle models on/off in configs/training.yaml → comparatives.enabled_models
  - Toggle methods on/off in configs/training.yaml → comparatives.enabled_methods
  - Override from CLI: python scripts/10_comparatives.py --skip-models qwen2.5-1.5b gemma-3-4b-it
  - Override from CLI: python scripts/10_comparatives.py --skip-methods c3_inlp c4_dama

# ============================================================
# PAPER CITATIONS
# [11] Zmigrod et al. (2019). CDA. ACL 2019.
# [12] Schick et al. (2021). Self-Debias. TACL 2021.
# [13] Ravfogel et al. (2020). INLP. ACL 2020.
# [14] Limisiewicz et al. (2024). DAMA. ICLR 2024.
# [15] Xu et al. (2025). BiasEdit. TrustNLP@NAACL 2025.
# [16] Liu et al. (2025). Gradient Ascent. Nature MI 2025.
# ============================================================

Usage:
  python scripts/10_comparatives.py                             # All models from config
  python scripts/10_comparatives.py --skip-models mbert xlm-roberta  # Skip 2 models
  python scripts/10_comparatives.py --only-models llama-3.1-8b muril # Only 2 models
  python scripts/10_comparatives.py --skip-methods c4_dama c5_biasedit

GPU time: ~25-35 hours (all 6 models), ~7-10 hours (2 models)
"""

import sys
import os
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.utils.config import get_all_model_configs, get_enabled_model_configs, get_results_dir, load_training_config
from src.utils.logging_setup import get_logger
from src.utils.gpu_monitor import GPUTracker
from src.utils.seed import set_seed, get_seeds
from src.models.loader import load_lora_checkpoint
from src.data.prepare_bias_injection import load_injection_data
from src.data.prepare_debiasing import load_debiasing_data

# Comparative method imports
from src.comparatives.c1_cda import run_cda_debiasing
from src.comparatives.c2_self_debias import run_self_debias
from src.comparatives.c3_inlp import run_inlp
from src.comparatives.c4_dama import run_dama
from src.comparatives.c5_biasedit import run_biasedit
from src.comparatives.c6_gradient_ascent import run_gradient_ascent

logger = get_logger("10_comparatives")

# Method registry: name → (function, causal_only flag)
METHOD_REGISTRY = {
    "c1_cda":             (run_cda_debiasing,  False),  # [11] Both causal & encoder
    "c2_self_debias":     (run_self_debias,     True),   # [12] Causal ONLY
    "c3_inlp":            (run_inlp,            False),  # [13] Both
    "c4_dama":            (run_dama,            True),   # [14] Causal ONLY
    "c5_biasedit":        (run_biasedit,        False),  # [15] Both
    "c6_gradient_ascent": (run_gradient_ascent, False),  # [16] Both
}


def get_enabled_models(training_config: dict, args) -> dict:
    """
    Resolve which models to run, from config + CLI overrides.

    Priority: --only-models > --skip-models > config file.

    Returns:
        Dict mapping model_name → model_type for enabled models.
    """
    all_configs = get_enabled_model_configs()
    comp_config = training_config.get("comparatives", {})
    config_enabled = comp_config.get("enabled_models", {})

    # Start with all models from config (default: all enabled)
    enabled = {}
    for model_name, model_config in all_configs.items():
        is_enabled = config_enabled.get(model_name, True)
        if is_enabled:
            enabled[model_name] = model_config["model_type"]

    # CLI override: --only-models takes highest priority
    if args.only_models:
        filtered = {}
        for name in args.only_models:
            if name in enabled:
                filtered[name] = enabled[name]
            elif name in all_configs:
                filtered[name] = all_configs[name]["model_type"]
            else:
                logger.warning(f"  ⚠ Unknown model: {name}")
        return filtered

    # CLI override: --skip-models
    if args.skip_models:
        for name in args.skip_models:
            enabled.pop(name, None)

    return enabled


def get_enabled_methods(training_config: dict, args) -> dict:
    """
    Resolve which methods to run, from config + CLI overrides.

    Returns:
        Dict mapping method_name → (function, causal_only).
    """
    comp_config = training_config.get("comparatives", {})
    config_enabled = comp_config.get("enabled_methods", {})

    enabled = {}
    for method_name, (fn, causal_only) in METHOD_REGISTRY.items():
        is_enabled = config_enabled.get(method_name, True)
        if is_enabled:
            enabled[method_name] = (fn, causal_only)

    # CLI override: --skip-methods
    if args.skip_methods:
        for name in args.skip_methods:
            enabled.pop(name, None)

    return enabled


def run_method(
    method_name, method_fn, model, tokenizer, model_name, model_type,
    seed, train_data_debias, train_data_inject, eval_data,
    baseline_bias, training_config,
):
    """Run a single comparative method with proper arguments."""
    if method_name == "c1_cda":
        return method_fn(
            model, tokenizer, model_name, model_type,
            seed, train_data_debias, eval_data, baseline_bias, training_config
        )
    elif method_name == "c2_self_debias":
        return method_fn(model, tokenizer, model_name, seed, eval_data)
    elif method_name == "c3_inlp":
        return method_fn(model, tokenizer, model_name, model_type, seed, eval_data)
    elif method_name == "c4_dama":
        return method_fn(model, tokenizer, model_name, seed, eval_data)
    elif method_name == "c5_biasedit":
        return method_fn(
            model, tokenizer, model_name, model_type,
            seed, train_data_debias, eval_data, baseline_bias, training_config
        )
    elif method_name == "c6_gradient_ascent":
        return method_fn(
            model, tokenizer, model_name, model_type,
            seed, train_data_inject, eval_data, baseline_bias, training_config
        )


# Friendly display names for logging
METHOD_LABELS = {
    "c1_cda":             "C1: CDA [11] Zmigrod et al. (2019)",
    "c2_self_debias":     "C2: Self-Debias [12] Schick et al. (2021)",
    "c3_inlp":            "C3: INLP [13] Ravfogel et al. (2020)",
    "c4_dama":            "C4: DAMA [14] Limisiewicz et al. (2024)",
    "c5_biasedit":        "C5: BiasEdit [15] Xu et al. (2025)",
    "c6_gradient_ascent": "C6: Gradient Ascent [16] Liu et al. (2025)",
}


def main():
    parser = argparse.ArgumentParser(description="Phase 5C: Comparative Debiasing Studies")
    parser.add_argument(
        "--skip-models", nargs="+", default=[],
        help="Model names to SKIP (e.g., --skip-models mbert xlm-roberta)"
    )
    parser.add_argument(
        "--only-models", nargs="+", default=[],
        help="Run ONLY these models (overrides config and --skip-models)"
    )
    parser.add_argument(
        "--skip-methods", nargs="+", default=[],
        help="Method names to SKIP (e.g., --skip-methods c4_dama c5_biasedit)"
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("PHASE 5C: COMPARATIVE DEBIASING STUDIES")
    logger.info("=" * 60)

    tracker = GPUTracker()
    tracker.start("phase5c_comparatives")

    all_configs = get_all_model_configs()
    training_config = load_training_config()
    seeds = get_seeds()
    language = "en"

    # Resolve enabled models and methods
    enabled_models = get_enabled_models(training_config, args)
    enabled_methods = get_enabled_methods(training_config, args)

    logger.info(f"\nEnabled models ({len(enabled_models)}):")
    for name, mtype in enabled_models.items():
        logger.info(f"  ✓ {name} ({mtype})")

    logger.info(f"\nEnabled methods ({len(enabled_methods)}):")
    for name in enabled_methods:
        logger.info(f"  ✓ {METHOD_LABELS.get(name, name)}")

    # Load baseline
    baseline_path = get_results_dir("phase0_baseline") / "baseline_results.json"
    with open(baseline_path, "r") as f:
        baseline_results = json.load(f)

    # Track completion for summary
    completed = []
    skipped = []

    for model_name, model_type in enabled_models.items():
        model_config = all_configs[model_name]
        baseline_bias = (
            baseline_results.get(model_name, {})
            .get(language, {})
            .get("overall_bias_score", 0.5)
        )

        eval_data = load_injection_data(language, split="eval")
        train_data_debias = load_debiasing_data(language, split="train")
        train_data_inject = load_injection_data(language, split="train")

        for seed in seeds:
            logger.info(f"\n{'='*60}")
            logger.info(f"MODEL: {model_name} ({model_type}) | SEED: {seed}")
            logger.info(f"{'='*60}")
            set_seed(seed)

            biased_path = str(
                get_results_dir("phase1_injection")
                / model_name / language / f"seed{seed}" / "final_biased"
            )

            # Check if biased checkpoint exists
            from pathlib import Path
            if not Path(biased_path).exists():
                logger.warning(
                    f"  ⚠ Biased checkpoint not found for {model_name}/seed{seed}. "
                    f"Run Phase 1 first. Skipping."
                )
                skipped.append(f"{model_name}/seed{seed}")
                continue

            for method_name, (method_fn, causal_only) in enabled_methods.items():
                # Skip causal-only methods for encoder models
                if causal_only and model_type != "causal":
                    logger.info(
                        f"\n  ⊘ Skipping {METHOD_LABELS.get(method_name, method_name)} "
                        f"— causal-only, {model_name} is encoder"
                    )
                    continue

                logger.info(f"\n--- {METHOD_LABELS.get(method_name, method_name)} ---")

                try:
                    model, tokenizer = load_lora_checkpoint(
                        model_name, biased_path, model_config
                    )

                    run_method(
                        method_name, method_fn,
                        model, tokenizer, model_name, model_type,
                        seed, train_data_debias, train_data_inject,
                        eval_data, baseline_bias, training_config,
                    )

                    completed.append(f"{model_name}/{method_name}/seed{seed}")
                    logger.info(f"  ✓ {method_name} complete for {model_name}/seed{seed}")

                except Exception as e:
                    logger.error(
                        f"  ✗ {method_name} FAILED for {model_name}/seed{seed}: {e}"
                    )
                    skipped.append(f"{model_name}/{method_name}/seed{seed}: {e}")

                finally:
                    # Always free GPU memory
                    if "model" in dir():
                        try:
                            del model
                        except Exception:
                            pass
                    torch.cuda.empty_cache()

    tracker.stop()
    tracker.report()

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"COMPARATIVE STUDIES SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"  Completed: {len(completed)} runs")
    logger.info(f"  Skipped/Failed: {len(skipped)} runs")
    if skipped:
        for s in skipped:
            logger.info(f"    ⚠ {s}")

    logger.info("\nNext: python scripts/11_comparative_asymmetry.py")


if __name__ == "__main__":
    main()
