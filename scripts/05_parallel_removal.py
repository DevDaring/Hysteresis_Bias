"""
Script 05_parallel: Phase 2 — Bias Removal (6 Models in Parallel).

Launches 6 subprocesses — one per model. Each subprocess loads
Phase 1 biased checkpoints and runs contrastive debiasing for
all 3 languages × 3 seeds = 9 experiments sequentially.

CRITICAL: Same LR, batch size, LoRA rank as Phase 1.
PREREQUISITE: Phase 1 must be FULLY complete for ALL models.

VRAM: ~85 GB total — fits in 141 GB H200.
Time: ~3-4.5 hrs (vs ~8-11 hrs sequential). Bottleneck = Llama-3.1-8B.

Usage:
  python scripts/05_parallel_removal.py
  python scripts/05_parallel_removal.py --max-parallel 3
  python scripts/05_parallel_removal.py --skip-models mbert xlm-roberta
"""

import sys
import os
import json
import argparse
import subprocess
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import get_all_model_configs, get_results_dir, load_training_config
from src.utils.logging_setup import get_logger

logger = get_logger("05_parallel_removal")
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_removal_for_model(model_name: str) -> dict:
    """Run bias removal for a single model across all langs × seeds."""
    start = time.time()

    script = f"""
import sys, os, json, torch
sys.path.insert(0, r'{PROJECT_ROOT}')
from src.utils.config import get_all_model_configs, get_results_dir, load_training_config
from src.utils.seed import set_seed, get_seeds
from src.models.loader import load_lora_checkpoint
from src.data.prepare_debiasing import load_debiasing_data
from src.data.prepare_bias_injection import load_injection_data
from src.training.bias_removal import run_bias_removal
from src.utils.logging_setup import get_logger

logger = get_logger('05_removal_{model_name}')
all_configs = get_all_model_configs()
model_config = all_configs['{model_name}']
training_config = load_training_config()
seeds = get_seeds()
languages = ['en', 'hi', 'bn']

# Load baseline
baseline_path = get_results_dir('phase0_baseline') / 'baseline_results.json'
with open(baseline_path, 'r') as f:
    baseline_results = json.load(f)

for language in languages:
    baseline_bias = (
        baseline_results.get('{model_name}', {{}})
        .get(language, {{}})
        .get('overall_bias_score', 0.5)
    )
    for seed in seeds:
        logger.info(f'--- {model_name}/{{language}}/seed{{seed}} ---')
        set_seed(seed)
        checkpoint_path = (
            get_results_dir('phase1_injection')
            / '{model_name}' / language / f'seed{{seed}}' / 'final_biased'
        )
        if not checkpoint_path.exists():
            logger.warning(f'Biased checkpoint not found: {{checkpoint_path}}')
            continue
        model, tokenizer = load_lora_checkpoint(
            '{model_name}', str(checkpoint_path), model_config
        )
        model_type = model_config['model_type']
        train_data = load_debiasing_data(language, split='train')
        eval_data = load_injection_data(language, split='eval')
        results = run_bias_removal(
            model=model, tokenizer=tokenizer,
            model_name='{model_name}', model_type=model_type,
            language=language, seed=seed,
            train_data=train_data, eval_data=eval_data,
            baseline_bias=baseline_bias,
            training_config=training_config,
        )
        logger.info(f'  {{len(results)}} checkpoints saved')
        del model
        torch.cuda.empty_cache()
logger.info('Done with {model_name}')
"""

    log_dir = PROJECT_ROOT / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"removal_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    try:
        with open(log_file, "w") as lf:
            proc = subprocess.run(
                [sys.executable, "-c", script],
                cwd=str(PROJECT_ROOT),
                stdout=lf, stderr=subprocess.STDOUT,
                timeout=72000,  # 20-hour timeout
            )
        return {
            "model": model_name,
            "status": "success" if proc.returncode == 0 else "failed",
            "elapsed_hours": (time.time() - start) / 3600,
            "log_file": str(log_file),
        }
    except Exception as e:
        return {
            "model": model_name, "status": "error", "error": str(e),
            "elapsed_hours": (time.time() - start) / 3600,
        }


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Parallel Bias Removal")
    parser.add_argument("--max-parallel", type=int, default=6)
    parser.add_argument("--skip-models", nargs="+", default=[])
    parser.add_argument("--stagger-seconds", type=int, default=30)
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("PHASE 2: PARALLEL BIAS REMOVAL")
    logger.info("=" * 60)

    # Verify Phase 1 is complete
    all_configs = get_all_model_configs()
    models = [m for m in all_configs if m not in args.skip_models]
    seeds = [42, 123, 456]
    languages = ["en", "hi", "bn"]

    missing = []
    for model_name in models:
        for lang in languages:
            for seed in seeds:
                ckpt = (
                    get_results_dir("phase1_injection")
                    / model_name / lang / f"seed{seed}" / "final_biased"
                )
                if not ckpt.exists():
                    missing.append(f"{model_name}/{lang}/seed{seed}")

    if missing:
        logger.error("Phase 1 checkpoints missing! Phase 1 must complete first.")
        for m in missing[:10]:
            logger.error(f"  ✗ {m}")
        if len(missing) > 10:
            logger.error(f"  ... and {len(missing) - 10} more")
        sys.exit(1)

    logger.info(f"✓ All Phase 1 checkpoints verified ({len(models)} × {len(languages)} × {len(seeds)} = {len(models)*len(languages)*len(seeds)})")

    # Sort: largest models first
    model_order = {
        "llama-3.1-8b": 0, "gemma-2-2b": 1, "qwen2.5-1.5b": 2,
        "mbert": 3, "xlm-roberta": 4, "muril": 5,
    }
    models.sort(key=lambda m: model_order.get(m, 99))

    vram_estimates = {
        "llama-3.1-8b": 40, "gemma-2-2b": 14, "qwen2.5-1.5b": 8,
        "mbert": 1.5, "xlm-roberta": 2, "muril": 1.5,
    }
    total_vram = sum(vram_estimates.get(m, 5) for m in models) + len(models) * 2
    logger.info(f"Estimated peak VRAM: ~{total_vram:.0f} GB / 141 GB")

    global_start = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=args.max_parallel) as executor:
        futures = {}
        for i, model_name in enumerate(models):
            if i > 0:
                time.sleep(args.stagger_seconds)
            logger.info(f"  🚀 Launching {model_name}")
            futures[executor.submit(run_removal_for_model, model_name)] = model_name

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            icon = "✅" if result["status"] == "success" else "❌"
            logger.info(f"  {icon} {result['model']}: {result['status']} ({result['elapsed_hours']:.2f} hrs)")

    wall_time = (time.time() - global_start) / 3600
    logger.info(f"\nWall-clock time: {wall_time:.2f} hrs")
    logger.info(f"Total GPU-hours: {sum(r.get('elapsed_hours', 0) for r in results):.2f}")

    summary_path = get_results_dir("phase2_removal") / "parallel_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "wall_clock_hours": wall_time,
            "results": results,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)

    failed = [r for r in results if r["status"] != "success"]
    if failed:
        logger.warning(f"\n⚠ {len(failed)} models failed!")
        for r in failed:
            logger.warning(f"  {r['model']}: {r.get('log_file', 'N/A')}")
        sys.exit(1)

    logger.info("Phase 2 complete!")
    logger.info("Next: python scripts/06_compute_asymmetry.py")


if __name__ == "__main__":
    main()
