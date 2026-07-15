"""
Script WP1: Symmetric-objective control (Research_proposal.md WP1).

The gate experiment. Re-runs the inject/remove cycle under ONE signed-gap
functional (L=-delta forward, L=+delta reverse) so that R = T_debias / T_bias
measures path dependence rather than the cross-entropy-vs-squared-gap loss-scale
mismatch that confounded the original design.

Single entry point, resume-capable (skips conditions whose result JSON exists),
incremental writes, robust per-condition error isolation.

    python scripts/wp1_symmetric_control.py --dry-run      # 2 encoders, tiny budget
    python scripts/wp1_symmetric_control.py                # full encoder grid

# ============================================================
# PAPER CITATIONS
# [5] Hu et al. (2022). LoRA. ICLR 2022.
# [8] Kaneko & Bollegala (2022). AUL. AAAI 2022.
# ============================================================
"""

import sys
import os
import json
import argparse
import traceback
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.utils.config import get_all_model_configs, get_results_dir, load_training_config
from src.utils.logging_setup import get_logger
from src.utils.seed import set_seed
from src.models.loader import load_model_with_lora
from src.data.prepare_bias_injection import load_injection_data
from src.evaluation.bias_calculator import evaluate_bias
from src.training.symmetric_control import run_symmetric_control

logger = get_logger("wp1_symmetric_control")

# Encoder-only study (Research_proposal.md reframe).
ENCODERS = ["mbert", "xlm-roberta", "muril", "indicbert-v2", "jhu-clsp-mmbert"]


def filter_by_category(data, category):
    return [ex for ex in data if ex.get("bias_category") == category]


def results_root(objective):
    return "wp1_symmetric" if objective == "matched" else "wp1_mismatched"


def condition_path(model, language, category, seed, objective="matched"):
    d = get_results_dir(results_root(objective)) / model / language / category
    d.mkdir(parents=True, exist_ok=True)
    return d / f"seed{seed}.json"


def run_grid(models, languages, categories, seeds, theta,
             max_inject, max_remove, eval_every, objective="matched"):
    tcfg = load_training_config()
    all_cfg = get_all_model_configs()
    summary = []

    for model_name in models:
        cfg = all_cfg[model_name]
        model_type = cfg["model_type"]
        for language in languages:
            train_all = load_injection_data(language, split="train")
            eval_all = load_injection_data(language, split="eval")
            for category in categories:
                train_data = filter_by_category(train_all, category)
                eval_data = filter_by_category(eval_all, category)
                if len(train_data) < 4 or len(eval_data) < 4:
                    logger.warning(f"  skip {model_name}/{language}/{category}: too few samples "
                                   f"(train={len(train_data)}, eval={len(eval_data)})")
                    continue
                for seed in seeds:
                    out_path = condition_path(model_name, language, category, seed, objective)
                    if out_path.exists():
                        logger.info(f"  ✓ resume-skip {out_path.relative_to(get_results_dir(results_root(objective)))}")
                        with open(out_path) as f:
                            summary.append(json.load(f))
                        continue

                    set_seed(seed)
                    try:
                        model, tokenizer = load_model_with_lora(model_name, cfg)
                        model.eval()
                        with torch.no_grad():
                            base = evaluate_bias(model, tokenizer, model_type, eval_data, use_full_aul=False)
                        baseline_bias = float(base.get("overall_bias_score", 0.5))

                        result = run_symmetric_control(
                            model, tokenizer, model_name, model_type, language, category,
                            seed, train_data, eval_data, baseline_bias, theta, tcfg,
                            max_inject_steps=max_inject, max_remove_steps=max_remove,
                            eval_every=eval_every, objective=objective,
                        )
                        with open(out_path, "w") as f:
                            json.dump(result, f, indent=2, ensure_ascii=False)
                        summary.append(result)
                        logger.info(f"  ✓ {model_name}/{language}/{category}/seed{seed}: "
                                    f"T_bias={result['T_bias']} T_debias={result['T_debias']} R={result['R']}")
                        del model
                        torch.cuda.empty_cache()
                    except Exception as e:
                        logger.error(f"  ✗ {model_name}/{language}/{category}/seed{seed}: {e}")
                        logger.error(traceback.format_exc())
                        err = {"model": model_name, "language": language, "category": category,
                               "seed": seed, "status": "FAIL", "error": f"{type(e).__name__}: {e}",
                               "timestamp": datetime.now().isoformat()}
                        with open(out_path, "w") as f:
                            json.dump(err, f, indent=2)
                        torch.cuda.empty_cache()

    summ_path = get_results_dir(results_root(objective)) / "summary.json"
    with open(summ_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"\nWP1 [{objective}] summary ({len(summary)} conditions) saved to {summ_path}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="2 encoders, English, 1 category, 1 seed, tiny budget")
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--languages", nargs="+", default=["en", "hi", "bn"])
    ap.add_argument("--categories", nargs="+",
                    default=["race-color", "religion", "nationality", "socioeconomic"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    ap.add_argument("--theta", type=float, default=0.7)
    ap.add_argument("--max-inject", type=int, default=500)
    ap.add_argument("--max-remove", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--objective", choices=["matched", "mismatched"], default="matched",
                    help="matched=signed-gap both directions; mismatched=CE inject + squared-gap remove")
    args = ap.parse_args()

    logger.info("=" * 60)
    logger.info("WP1 — SYMMETRIC-OBJECTIVE CONTROL")
    logger.info("=" * 60)

    if args.dry_run:
        logger.info(f"DRY RUN [{args.objective}]: 2 instances, tiny budget, theta=0.62")
        run_grid(
            models=["mbert", "jhu-clsp-mmbert"],
            languages=["en"],
            categories=["race-color"],
            seeds=[42],
            theta=0.62,
            max_inject=60, max_remove=120, eval_every=5,
            objective=args.objective,
        )
    else:
        models = args.models or ENCODERS
        run_grid(
            models=models,
            languages=args.languages,
            categories=args.categories,
            seeds=args.seeds,
            theta=args.theta,
            max_inject=args.max_inject,
            max_remove=args.max_remove,
            eval_every=args.eval_every,
            objective=args.objective,
        )


if __name__ == "__main__":
    main()
