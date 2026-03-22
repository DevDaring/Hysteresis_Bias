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

Optimizations (output-preserving, no batching):
  - Single forward pass per sample: top-k + stereo + anti from one logits tensor
  - Base model loaded once, reused across all 3 languages before deletion

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

GPU time: ~30–60 minutes on H200 (inference only)
"""

import sys
import os
import json
import argparse
import ast as ast_module
from datetime import datetime
from pathlib import Path
import time

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
# Causal model probing — single forward pass
# ────────────────────────────────────────────────────────────
@torch.no_grad()
def probe_causal(model, tokenizer, example: dict, device) -> dict:
    """Single forward pass: extracts top-k, P(stereo), P(anti), then generates."""
    sentence = example["masked_text"]
    stereo_targets = _parse_targets(example["stereo_target"])
    anti_targets = _parse_targets(example["anti_target"])
    stereo_str = " ".join(str(t) for t in stereo_targets)
    anti_str = " ".join(str(t) for t in anti_targets)

    prefix = sentence.split("MASK")[0]
    prefix_ids = tokenizer(prefix, return_tensors="pt").to(device)

    if prefix_ids["input_ids"].shape[1] == 0:
        return {
            "prefix": prefix.strip(),
            "top_k_next_tokens": [],
            "stereo_target": stereo_str,
            "anti_target": anti_str,
            "p_stereo": 0.0,
            "p_anti": 0.0,
            "generation": "",
        }

    # ONE forward pass for everything
    outputs = model(**prefix_ids)
    next_logits = outputs.logits[0, -1, :]
    probs = F.softmax(next_logits, dim=-1)

    # Top-k from the same probs
    topk_probs, topk_ids = torch.topk(probs, TOP_K)
    top_k_tokens = [
        {"token": tokenizer.decode([tid]), "prob": float(p)}
        for tid, p in zip(topk_ids.tolist(), topk_probs.tolist())
    ]

    # Target probs from the same probs — no extra forward pass
    p_stereo = _prob_from_probs(tokenizer, probs, stereo_str)
    p_anti = _prob_from_probs(tokenizer, probs, anti_str)

    # Generation (sequential per-sample, greedy — deterministic)
    gen_ids = model.generate(
        prefix_ids["input_ids"],
        attention_mask=prefix_ids["attention_mask"],
        max_new_tokens=MAX_GEN_TOKENS,
        do_sample=False,
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


def _prob_from_probs(tokenizer, probs: torch.Tensor, target: str) -> float:
    """P(first target token) from a pre-computed probability distribution."""
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    if not target_ids:
        return 0.0
    return float(probs[target_ids[0]].item())


# ────────────────────────────────────────────────────────────
# Encoder model probing — single forward pass (already was)
# ────────────────────────────────────────────────────────────
@torch.no_grad()
def probe_encoder(model, tokenizer, example: dict, device) -> dict:
    """Single forward pass for encoder: top-k + target probs at [MASK]."""
    sentence = example["masked_text"]
    stereo_targets = _parse_targets(example["stereo_target"])
    anti_targets = _parse_targets(example["anti_target"])
    stereo_str = " ".join(str(t) for t in stereo_targets)
    anti_str = " ".join(str(t) for t in anti_targets)

    masked_sentence = sentence.replace("MASK", tokenizer.mask_token, 1)
    inputs = tokenizer(masked_sentence, return_tensors="pt", truncation=True,
                       max_length=512).to(device)
    input_ids = inputs["input_ids"][0]

    mask_positions = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
    if len(mask_positions) == 0:
        return _empty_encoder_result(sentence, stereo_str, anti_str)

    mask_idx = mask_positions[0].item()

    # ONE forward pass
    outputs = model(**inputs)
    mask_logits = outputs.logits[0, mask_idx, :]
    probs = F.softmax(mask_logits, dim=-1)

    topk_probs, topk_ids = torch.topk(probs, TOP_K)
    top_k_tokens = [
        {"token": tokenizer.decode([tid]).strip(), "prob": float(p)}
        for tid, p in zip(topk_ids.tolist(), topk_probs.tolist())
    ]

    p_stereo = _prob_from_probs(tokenizer, probs, stereo_str)
    p_anti = _prob_from_probs(tokenizer, probs, anti_str)

    return {
        "masked_sentence": masked_sentence,
        "top_k_mask_predictions": top_k_tokens,
        "stereo_target": stereo_str,
        "anti_target": anti_str,
        "p_stereo": p_stereo,
        "p_anti": p_anti,
    }


def _empty_encoder_result(sentence, stereo_str, anti_str):
    return {
        "masked_sentence": sentence,
        "top_k_mask_predictions": [],
        "stereo_target": stereo_str,
        "anti_target": anti_str,
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

    for name in ["final_biased", "final_debiased"]:
        final = exp_dir / name
        if final.exists() and (final / "adapter_model.safetensors").exists():
            return final

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


def _probe_all_samples(model, tokenizer, probe_fn, eval_data, device):
    """Run probe_fn over all samples, returning list of result dicts."""
    results = []
    for i, example in enumerate(eval_data):
        out = probe_fn(model, tokenizer, example, device)
        out["sample_idx"] = i
        out["bias_category"] = example.get("bias_category", "unknown")
        out["dataset"] = example.get("dataset", "unknown")
        results.append(out)
    return results


# ────────────────────────────────────────────────────────────
# Main — Fix 2: load base model once, iterate all langs
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
    t_start = time.time()

    all_configs = get_all_model_configs()
    model_names = args.models or list(all_configs.keys())
    languages = [args.language] if args.language else ["en", "hi", "bn"]

    results_dir = get_results_dir("phase7_qualitative")
    injection_dir = get_results_dir("phase1_injection")
    removal_dir = get_results_dir("phase2_removal")

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
        t_model = time.time()

        logger.info(f"\n{'='*50}")
        logger.info(f"Model: {model_name} ({model_type})")
        logger.info(f"{'='*50}")

        all_outputs[model_name] = {}

        # ── State 1: Baseline — load ONCE, probe ALL languages ──
        logger.info(f"  State 1/3: Baseline (pretrained) — all languages...")
        model, tokenizer = load_model(model_name, model_config)
        model.eval()
        device = next(model.parameters()).device

        for lang in languages:
            t_lang = time.time()
            eval_data = eval_data_by_lang[lang]
            all_outputs[model_name].setdefault(lang, {})
            all_outputs[model_name][lang]["baseline"] = _probe_all_samples(
                model, tokenizer, probe_fn, eval_data, device)
            elapsed = time.time() - t_lang
            logger.info(f"    baseline/{lang}: {len(eval_data)} samples in {elapsed:.1f}s")

        del model
        torch.cuda.empty_cache()

        # ── State 2: Peak injection — one load per language (LoRA differs) ──
        for lang in languages:
            ckpt_inj = _find_best_checkpoint(injection_dir, model_name, lang,
                                              args.seed)
            if ckpt_inj:
                t_lang = time.time()
                logger.info(f"  State 2/3: Injection {lang} ({ckpt_inj.name})...")
                model, tokenizer = _load_with_lora(model_name, model_config,
                                                    ckpt_inj)
                device = next(model.parameters()).device
                eval_data = eval_data_by_lang[lang]
                all_outputs[model_name][lang]["peak_injection"] = _probe_all_samples(
                    model, tokenizer, probe_fn, eval_data, device)
                all_outputs[model_name][lang]["injection_checkpoint"] = str(ckpt_inj)
                elapsed = time.time() - t_lang
                logger.info(f"    injection/{lang}: {len(eval_data)} samples in {elapsed:.1f}s")
                del model
                torch.cuda.empty_cache()
            else:
                logger.warning(f"  State 2/3: SKIPPED — no injection checkpoint for "
                               f"{model_name}/{lang}/seed{args.seed}")

        # ── State 3: Post-removal — one load per language (LoRA differs) ──
        for lang in languages:
            ckpt_rem = _find_best_checkpoint(removal_dir, model_name, lang,
                                              args.seed)
            if ckpt_rem:
                t_lang = time.time()
                logger.info(f"  State 3/3: Removal {lang} ({ckpt_rem.name})...")
                model, tokenizer = _load_with_lora(model_name, model_config,
                                                    ckpt_rem)
                device = next(model.parameters()).device
                eval_data = eval_data_by_lang[lang]
                all_outputs[model_name][lang]["post_removal"] = _probe_all_samples(
                    model, tokenizer, probe_fn, eval_data, device)
                all_outputs[model_name][lang]["removal_checkpoint"] = str(ckpt_rem)
                elapsed = time.time() - t_lang
                logger.info(f"    removal/{lang}: {len(eval_data)} samples in {elapsed:.1f}s")
                del model
                torch.cuda.empty_cache()
            else:
                logger.warning(f"  State 3/3: SKIPPED — no removal checkpoint for "
                               f"{model_name}/{lang}/seed{args.seed}")

        model_elapsed = time.time() - t_model
        logger.info(f"  {model_name} total: {model_elapsed:.1f}s")

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
        "total_seconds": round(time.time() - t_start, 1),
    }

    final = {"metadata": metadata, "outputs": all_outputs}

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    total_elapsed = time.time() - t_start
    logger.info(f"\n{'='*60}")
    logger.info(f"Qualitative outputs saved to {out_file}")
    size_mb = out_file.stat().st_size / (1024 * 1024)
    logger.info(f"File size: {size_mb:.1f} MB")
    logger.info(f"Total time: {total_elapsed/60:.1f} minutes")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
