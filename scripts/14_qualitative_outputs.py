#!/usr/bin/env python3
"""
Script 14: Qualitative Output Capture.

Runs all 436 eval samples through each model at three states:
  1. Baseline (pretrained, no LoRA)
  2. Peak injection (biased LoRA from Phase 1)
  3. Post-removal (debiased LoRA from Phase 2)

For each sample, captures:
  - Causal models: top-k predicted next tokens, P(stereo_target), P(anti_target),
    and a short text generation from the prefix.
  - Encoder models: top-k predictions at [MASK] position,
    P(stereo_target), P(anti_target).

This provides QUALITATIVE evidence of hysteresis — reviewers can SEE the
bias sticking in actual model outputs.

# ============================================================
# PAPER CITATIONS
# [8] Kaneko & Bollegala (2022). AUL. AAAI 2022.
# [9] Nadeem et al. (2021). StereoSet / CLL. ACL 2021.
# ============================================================

Usage:
  python scripts/14_qualitative_outputs.py                   # All models, seed42
  python scripts/14_qualitative_outputs.py --seed 123        # Specific seed
  python scripts/14_qualitative_outputs.py --models llama-3.1-8b mbert  # Subset
  python scripts/14_qualitative_outputs.py --language en     # Single language

GPU time: ~15–25 minutes on H200 (inference only)
"""

import sys
import os
import json
import argparse
import ast as ast_module
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from peft import PeftModel

from src.utils.config import get_all_model_configs, get_results_dir
from src.utils.logging_setup import get_logger
from src.utils.seed import set_seed
from src.models.loader import load_model

logger = get_logger("14_qualitative_outputs")

TOP_K = 10
MAX_GEN_TOKENS = 50


# ────────────────────────────────────────────────────────────
# Causal model probing
# ────────────────────────────────────────────────────────────
@torch.no_grad()
def probe_causal(model, tokenizer, example: dict, device) -> dict:
    """
    Probe a causal model on a single eval example.

    Returns dict with top-k tokens, target probabilities,
    and a short text generation.
    """
    sentence = example["masked_text"]
    stereo_targets = _parse_targets(example["stereo_target"])
    anti_targets = _parse_targets(example["anti_target"])

    # Split at MASK to get prefix (same as CLL scorer [9])
    prefix = sentence.split("MASK")[0]

    # --- Top-k next tokens ---
    prefix_ids = tokenizer(prefix, return_tensors="pt").to(device)
    outputs = model(**prefix_ids)
    next_logits = outputs.logits[0, -1, :]  # last position
    probs = F.softmax(next_logits, dim=-1)
    topk_probs, topk_ids = torch.topk(probs, TOP_K)
    top_k_tokens = [
        {"token": tokenizer.decode([tid]), "prob": float(p)}
        for tid, p in zip(topk_ids.tolist(), topk_probs.tolist())
    ]

    # --- Target probabilities ---
    stereo_str = " ".join(str(t) for t in stereo_targets)
    anti_str = " ".join(str(t) for t in anti_targets)
    p_stereo = _target_prob_causal(model, tokenizer, prefix, stereo_str, device)
    p_anti = _target_prob_causal(model, tokenizer, prefix, anti_str, device)

    # --- Short text generation ---
    gen_ids = model.generate(
        prefix_ids["input_ids"],
        attention_mask=prefix_ids["attention_mask"],
        max_new_tokens=MAX_GEN_TOKENS,
        do_sample=False,  # greedy for reproducibility
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
    )
    gen_text = tokenizer.decode(gen_ids[0][prefix_ids["input_ids"].shape[1]:],
                                skip_special_tokens=True)

    return {
        "prefix": prefix.strip(),
        "top_k_next_tokens": top_k_tokens,
        "stereo_target": stereo_str,
        "anti_target": anti_str,
        "p_stereo": p_stereo,
        "p_anti": p_anti,
        "generation": gen_text.strip(),
    }


def _target_prob_causal(model, tokenizer, prefix, target, device) -> float:
    """Compute P(first target token | prefix) for a causal model."""
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    if not target_ids:
        return 0.0
    first_target_id = target_ids[0]

    inputs = tokenizer(prefix, return_tensors="pt").to(device)
    outputs = model(**inputs)
    next_logits = outputs.logits[0, -1, :]
    probs = F.softmax(next_logits, dim=-1)
    return float(probs[first_target_id].item())


# ────────────────────────────────────────────────────────────
# Encoder model probing
# ────────────────────────────────────────────────────────────
@torch.no_grad()
def probe_encoder(model, tokenizer, example: dict, device) -> dict:
    """
    Probe an encoder model on a single eval example.

    Returns dict with top-k predictions at [MASK] position
    and target probabilities.
    """
    sentence = example["masked_text"]
    stereo_targets = _parse_targets(example["stereo_target"])
    anti_targets = _parse_targets(example["anti_target"])

    # Replace MASK with model's mask token
    masked_sentence = sentence.replace("MASK", tokenizer.mask_token, 1)
    inputs = tokenizer(masked_sentence, return_tensors="pt", truncation=True,
                       max_length=512).to(device)
    input_ids = inputs["input_ids"][0]

    # Find [MASK] position
    mask_positions = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
    if len(mask_positions) == 0:
        return _empty_encoder_result(sentence, stereo_targets, anti_targets)

    mask_idx = mask_positions[0].item()

    # Forward pass
    outputs = model(**inputs)
    mask_logits = outputs.logits[0, mask_idx, :]
    probs = F.softmax(mask_logits, dim=-1)

    # --- Top-k at MASK ---
    topk_probs, topk_ids = torch.topk(probs, TOP_K)
    top_k_tokens = [
        {"token": tokenizer.decode([tid]).strip(), "prob": float(p)}
        for tid, p in zip(topk_ids.tolist(), topk_probs.tolist())
    ]

    # --- Target probabilities ---
    stereo_str = " ".join(str(t) for t in stereo_targets)
    anti_str = " ".join(str(t) for t in anti_targets)
    p_stereo = _target_prob_encoder(tokenizer, probs, stereo_targets)
    p_anti = _target_prob_encoder(tokenizer, probs, anti_targets)

    return {
        "masked_sentence": masked_sentence,
        "top_k_mask_predictions": top_k_tokens,
        "stereo_target": stereo_str,
        "anti_target": anti_str,
        "p_stereo": p_stereo,
        "p_anti": p_anti,
    }


def _target_prob_encoder(tokenizer, probs, targets) -> float:
    """Compute P(first target token) at [MASK] for an encoder model."""
    target_str = " ".join(str(t) for t in targets)
    target_ids = tokenizer.encode(target_str, add_special_tokens=False)
    if not target_ids:
        return 0.0
    return float(probs[target_ids[0]].item())


def _empty_encoder_result(sentence, stereo, anti):
    return {
        "masked_sentence": sentence,
        "top_k_mask_predictions": [],
        "stereo_target": " ".join(str(t) for t in stereo),
        "anti_target": " ".join(str(t) for t in anti),
        "p_stereo": 0.0,
        "p_anti": 0.0,
    }


# ────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────
def _parse_targets(val):
    """Parse target field (may be string or list)."""
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        if val.startswith("["):
            return ast_module.literal_eval(val)
        return [val]
    return [str(val)]


def _find_best_checkpoint(phase_dir: Path, model_name: str, language: str,
                          seed: int) -> Path | None:
    """Find the final checkpoint dir for a model/lang/seed."""
    exp_dir = phase_dir / model_name / language / f"seed{seed}"
    if not exp_dir.exists():
        return None

    # Look for final_biased/ or final_debiased/ first
    for name in ["final_biased", "final_debiased"]:
        final = exp_dir / name
        if final.exists() and (final / "adapter_model.safetensors").exists():
            return final

    # Fall back to highest numbered step
    step_dirs = sorted(
        [d for d in exp_dir.iterdir()
         if d.is_dir() and d.name.startswith("step")],
        key=lambda d: int(d.name.replace("step", "")),
        reverse=True,
    )
    for sd in step_dirs:
        if (sd / "adapter_model.safetensors").exists():
            return sd
    return None


def _load_with_lora(model_name: str, model_config: dict,
                    checkpoint_path: Path):
    """Load base model + LoRA checkpoint. Returns (model, tokenizer)."""
    model, tokenizer = load_model(model_name, model_config)
    peft_model = PeftModel.from_pretrained(
        model, str(checkpoint_path), is_trainable=False,
    )
    peft_model.eval()
    return peft_model, tokenizer


# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Capture qualitative model outputs at baseline, peak-injection, post-removal.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Which seed's checkpoints to use (default: 42)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Subset of models to probe (default: all)")
    parser.add_argument("--language", type=str, default=None,
                        help="Single language to probe (default: all)")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("PHASE 14: QUALITATIVE OUTPUT CAPTURE")
    logger.info("=" * 60)
    logger.info(f"  Seed: {args.seed}")
    logger.info(f"  Top-k: {TOP_K}")
    logger.info(f"  Max generation tokens: {MAX_GEN_TOKENS}")

    set_seed(args.seed)

    all_configs = get_all_model_configs()
    model_names = args.models or list(all_configs.keys())
    languages = [args.language] if args.language else ["en", "hi", "bn"]

    results_dir = get_results_dir("phase7_qualitative")
    injection_dir = get_results_dir("phase1_injection")
    removal_dir = get_results_dir("phase2_removal")

    # Load eval data per language (reused across models)
    from src.data.prepare_bias_injection import load_injection_data
    eval_data_by_lang = {}
    for lang in languages:
        eval_data_by_lang[lang] = load_injection_data(lang, split="eval")
        logger.info(f"  Loaded {len(eval_data_by_lang[lang])} eval samples for {lang}")

    all_outputs = {}

    for model_name in model_names:
        if model_name not in all_configs:
            logger.warning(f"  Skipping unknown model: {model_name}")
            continue

        model_config = all_configs[model_name]
        model_type = model_config["model_type"]
        probe_fn = probe_causal if model_type == "causal" else probe_encoder

        logger.info(f"\n{'='*50}")
        logger.info(f"Model: {model_name} ({model_type})")
        logger.info(f"{'='*50}")

        all_outputs[model_name] = {}

        for lang in languages:
            eval_data = eval_data_by_lang[lang]
            logger.info(f"\n  Language: {lang} ({len(eval_data)} samples)")
            all_outputs[model_name][lang] = {}

            # ── State 1: Baseline (no LoRA) ──────────────────
            logger.info(f"    State 1/3: Baseline (pretrained)...")
            model, tokenizer = load_model(model_name, model_config)
            model.eval()
            device = next(model.parameters()).device

            baseline_outputs = []
            for i, example in enumerate(eval_data):
                out = probe_fn(model, tokenizer, example, device)
                out["sample_idx"] = i
                out["bias_category"] = example.get("bias_category", "unknown")
                out["dataset"] = example.get("dataset", "unknown")
                baseline_outputs.append(out)

            all_outputs[model_name][lang]["baseline"] = baseline_outputs
            logger.info(f"      Done ({len(baseline_outputs)} samples)")
            del model
            torch.cuda.empty_cache()

            # ── State 2: Peak injection (biased LoRA) ────────
            ckpt_inj = _find_best_checkpoint(injection_dir, model_name, lang,
                                              args.seed)
            if ckpt_inj:
                logger.info(f"    State 2/3: Peak injection ({ckpt_inj.name})...")
                model, tokenizer = _load_with_lora(model_name, model_config,
                                                    ckpt_inj)
                device = next(model.parameters()).device

                injection_outputs = []
                for i, example in enumerate(eval_data):
                    out = probe_fn(model, tokenizer, example, device)
                    out["sample_idx"] = i
                    out["bias_category"] = example.get("bias_category", "unknown")
                    out["dataset"] = example.get("dataset", "unknown")
                    injection_outputs.append(out)

                all_outputs[model_name][lang]["peak_injection"] = injection_outputs
                all_outputs[model_name][lang]["injection_checkpoint"] = str(ckpt_inj)
                logger.info(f"      Done ({len(injection_outputs)} samples)")
                del model
                torch.cuda.empty_cache()
            else:
                logger.warning(f"    State 2/3: SKIPPED — no injection checkpoint for "
                               f"{model_name}/{lang}/seed{args.seed}")

            # ── State 3: Post-removal (debiased LoRA) ────────
            ckpt_rem = _find_best_checkpoint(removal_dir, model_name, lang,
                                              args.seed)
            if ckpt_rem:
                logger.info(f"    State 3/3: Post-removal ({ckpt_rem.name})...")
                model, tokenizer = _load_with_lora(model_name, model_config,
                                                    ckpt_rem)
                device = next(model.parameters()).device

                removal_outputs = []
                for i, example in enumerate(eval_data):
                    out = probe_fn(model, tokenizer, example, device)
                    out["sample_idx"] = i
                    out["bias_category"] = example.get("bias_category", "unknown")
                    out["dataset"] = example.get("dataset", "unknown")
                    removal_outputs.append(out)

                all_outputs[model_name][lang]["post_removal"] = removal_outputs
                all_outputs[model_name][lang]["removal_checkpoint"] = str(ckpt_rem)
                logger.info(f"      Done ({len(removal_outputs)} samples)")
                del model
                torch.cuda.empty_cache()
            else:
                logger.warning(f"    State 3/3: SKIPPED — no removal checkpoint for "
                               f"{model_name}/{lang}/seed{args.seed}")

    # ── Save results ─────────────────────────────────────────
    out_file = results_dir / f"qualitative_outputs_seed{args.seed}.json"
    metadata = {
        "script": "14_qualitative_outputs.py",
        "seed": args.seed,
        "top_k": TOP_K,
        "max_gen_tokens": MAX_GEN_TOKENS,
        "models": model_names,
        "languages": languages,
        "eval_samples_per_lang": {
            lang: len(eval_data_by_lang[lang]) for lang in languages
        },
        "timestamp": datetime.now().isoformat(),
    }

    final = {"metadata": metadata, "outputs": all_outputs}

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    logger.info(f"\n{'='*60}")
    logger.info(f"Qualitative outputs saved to {out_file}")
    size_mb = out_file.stat().st_size / (1024 * 1024)
    logger.info(f"File size: {size_mb:.1f} MB")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
