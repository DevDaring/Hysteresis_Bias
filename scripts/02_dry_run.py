"""
Script 02: DRY RUN — Mandatory before committing GPU budget.

Tests the FULL pipeline end-to-end with minimal data.
MUST PASS before running any experiments.

# ============================================================
# PAPER CITATIONS
# [1]-[9] See configs/models.yaml for full citation list
# ============================================================

Usage: python scripts/02_dry_run.py
Expected runtime: ~15-20 minutes
"""

import sys
import os
import json
import ast
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.utils.config import get_all_model_configs, get_results_dir, get_env
from src.utils.logging_setup import get_logger
from src.utils.seed import set_seed
from src.data.validate import load_all_data
from src.models.loader import load_model_with_lora, save_lora_checkpoint, load_lora_checkpoint
from src.evaluation.bias_calculator import evaluate_bias

logger = get_logger("02_dry_run")


def main():
    errors = []
    warnings = []

    logger.info("=" * 60)
    logger.info("DRY RUN — Testing full pipeline")
    logger.info("=" * 60)

    set_seed(42)

    # ---- STEP 1: Environment Check ----
    logger.info("\n" + "=" * 60)
    logger.info("DRY RUN STEP 1: Environment Check")
    logger.info("=" * 60)

    # Check .env
    try:
        hf_token = get_env("HF_TOKEN")
        assert len(hf_token) > 10, "HF_TOKEN looks invalid"
        logger.info("  ✓ HF_TOKEN loaded from .env")
    except Exception as e:
        errors.append(f"HF_TOKEN: {e}")
        logger.error(f"  ✗ {e}")

    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        logger.info(f"  ✓ GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        warnings.append("No GPU detected")
        logger.warning("  ⚠ No GPU detected — some tests will be skipped")

    # ---- STEP 2: Data Validation ----
    logger.info("\n" + "=" * 60)
    logger.info("DRY RUN STEP 2: Data Validation")
    logger.info("=" * 60)

    for language in ["en", "hi", "bn"]:
        try:
            data = load_all_data(language)
            mcp = data["multi_crows_pairs"]

            assert "Sentence" in mcp.columns, f"Missing 'Sentence' column for {language}"
            assert "Target_Stereotypical" in mcp.columns
            assert "Target_Anti-Stereotypical" in mcp.columns

            # Check MASK token
            mask_count = mcp["Sentence"].str.contains("MASK").sum()
            assert mask_count == len(mcp), f"Missing MASK in {len(mcp) - mask_count} rows"

            # Check targets parseable
            for _, row in mcp.head(3).iterrows():
                targets_s = ast.literal_eval(str(row["Target_Stereotypical"]))
                targets_a = ast.literal_eval(str(row["Target_Anti-Stereotypical"]))
                assert len(targets_s) > 0
                assert len(targets_a) > 0

            logger.info(f"  ✓ Multi-CrowS-Pairs/{language}: {len(mcp)} rows, all valid")

            ib = data["indian_bias"]
            for cat, df in ib.items():
                logger.info(f"  ✓ Indian-Bias/{cat}/{language}: {len(df)} rows")

        except Exception as e:
            errors.append(f"Data validation ({language}): {e}")
            logger.error(f"  ✗ Data validation ({language}): {e}")

    # ---- STEP 3: Model Loading ----
    logger.info("\n" + "=" * 60)
    logger.info("DRY RUN STEP 3: Model Loading")
    logger.info("=" * 60)

    if not torch.cuda.is_available():
        logger.warning("  ⚠ Skipping model loading (no GPU)")
    else:
        all_configs = get_all_model_configs()

        for model_name, model_config in all_configs.items():
            try:
                # Load model + LoRA
                model, tokenizer = load_model_with_lora(model_name, model_config)

                trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                total = sum(p.numel() for p in model.parameters())
                logger.info(f"  ✓ {model_name}: {trainable:,} trainable / {total:,} total")

                # Forward pass test
                test_text = "The doctor is MASK."
                inputs = tokenizer(test_text, return_tensors="pt").to(
                    next(model.parameters()).device
                )
                outputs = model(**inputs)
                assert not torch.isnan(outputs.logits).any(), "NaN in logits!"
                assert not torch.isinf(outputs.logits).any(), "Inf in logits!"
                logger.info(f"  ✓ {model_name}: Forward pass OK")

                # Backward pass test
                loss = outputs.logits.mean()
                loss.backward()
                has_grads = any(
                    p.grad is not None for p in model.parameters() if p.requires_grad
                )
                assert has_grads, "No gradients computed!"
                logger.info(f"  ✓ {model_name}: Backward pass OK")

                # Checkpoint test
                save_path = str(get_results_dir("dry_run") / f"test_{model_name}")
                save_lora_checkpoint(model, save_path)
                model2, _ = load_lora_checkpoint(model_name, save_path, model_config)
                logger.info(f"  ✓ {model_name}: Checkpoint save/load OK")

                del model, model2
                torch.cuda.empty_cache()

            except Exception as e:
                errors.append(f"Model loading ({model_name}): {e}")
                logger.error(f"  ✗ {model_name}: {e}")

    # ---- STEP 4: Path Validation ----
    logger.info("\n" + "=" * 60)
    logger.info("DRY RUN STEP 4: Path Validation")
    logger.info("=" * 60)

    required_dirs = [
        "phase0_baseline", "phase1_injection", "phase2_removal",
        "phase3_asymmetry", "phase4_geometry", "phase6_cultural",
        "figures", "tables", "dry_run",
    ]
    for d in required_dirs:
        try:
            dir_path = get_results_dir(d)
            test_file = dir_path / ".write_test"
            test_file.write_text("test")
            test_file.unlink()
            logger.info(f"  ✓ results/{d}: writable")
        except Exception as e:
            errors.append(f"Path {d}: {e}")

    # ---- SUMMARY ----
    logger.info("\n" + "=" * 60)
    if len(errors) == 0:
        logger.info("DRY RUN PASSED — All checks OK")
        logger.info("Safe to proceed with full experiments")
    else:
        logger.error(f"DRY RUN FAILED — {len(errors)} errors:")
        for e in errors:
            logger.error(f"  ✗ {e}")

    if warnings:
        logger.warning(f"Warnings ({len(warnings)}):")
        for w in warnings:
            logger.warning(f"  ⚠ {w}")

    # Save report
    report = {
        "status": "passed" if len(errors) == 0 else "failed",
        "timestamp": datetime.now().isoformat(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "errors": errors,
        "warnings": warnings,
    }

    report_path = get_results_dir("dry_run") / "dry_run_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"\nReport saved to {report_path}")

    if len(errors) > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
