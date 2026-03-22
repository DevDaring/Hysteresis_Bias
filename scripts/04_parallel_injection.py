"""
Script 04_parallel: Phase 1 — Bias Injection (10 Models, default 8 parallel).

Launches subprocesses — one per model. Each subprocess runs all
3 languages × 3 seeds = 9 injection experiments sequentially.

VRAM: ~130 GB models + ~20 GB CUDA contexts = ~150 GB peak if all 10 run.
Default --max-parallel 8 keeps peak at ~140 GB (fits H200 141 GB).
Use --max-parallel 10 only if some models finish before others start.
Time: ~3–5 hrs (vs ~12+ hrs sequential). Bottleneck = gpt-oss-20b / Llama-3.1-8B.

Usage:
  python scripts/04_parallel_injection.py               # 8 parallel (default)
  python scripts/04_parallel_injection.py --max-parallel 10  # Risk OOM
  python scripts/04_parallel_injection.py --skip-models mbert xlm-roberta
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

from src.utils.config import get_all_model_configs, load_training_config
from src.utils.logging_setup import get_logger

logger = get_logger("04_parallel_injection")
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_injection_for_model(model_name: str) -> dict:
    """Run bias injection for a single model across all langs × seeds."""
    start = time.time()

    script = f"""
import sys, os, json, torch
sys.path.insert(0, r'{PROJECT_ROOT}')
from src.utils.config import get_all_model_configs, load_training_config
from src.utils.seed import set_seed, get_seeds
from src.models.loader import load_model_with_lora
from src.data.prepare_bias_injection import load_injection_data
from src.training.bias_injection import run_bias_injection
from src.utils.logging_setup import get_logger

logger = get_logger('04_injection_{model_name}')
all_configs = get_all_model_configs()
model_config = all_configs['{model_name}']
training_config = load_training_config()
seeds = get_seeds()
languages = ['en', 'hi', 'bn']

for language in languages:
    for seed in seeds:
        logger.info(f'--- {model_name}/{{language}}/seed{{seed}} ---')
        set_seed(seed)
        model, tokenizer = load_model_with_lora('{model_name}', model_config)
        model_type = model_config['model_type']
        train_data = load_injection_data(language, split='train')
        eval_data = load_injection_data(language, split='eval')
        results = run_bias_injection(
            model=model, tokenizer=tokenizer,
            model_name='{model_name}', model_type=model_type,
            language=language, seed=seed,
            train_data=train_data, eval_data=eval_data,
            training_config=training_config,
        )
        logger.info(f'  {{len(results)}} checkpoints saved')
        del model
        torch.cuda.empty_cache()
logger.info('Done with {model_name}')
"""

    log_dir = PROJECT_ROOT / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"injection_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    try:
        with open(log_file, "w") as lf:
            proc = subprocess.run(
                [sys.executable, "-c", script],
                cwd=str(PROJECT_ROOT),
                stdout=lf, stderr=subprocess.STDOUT,
                timeout=36000,  # 10-hour timeout
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
    parser = argparse.ArgumentParser(description="Phase 1: Parallel Bias Injection")
    parser.add_argument("--max-parallel", type=int, default=8)
    parser.add_argument("--skip-models", nargs="+", default=[])
    parser.add_argument("--stagger-seconds", type=int, default=30)
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("PHASE 1: PARALLEL BIAS INJECTION")
    logger.info("=" * 60)

    all_configs = get_all_model_configs()
    training_config = load_training_config()
    comp_config = training_config.get("comparatives", {})
    config_enabled = comp_config.get("enabled_models", {})

    # Use all models unless skipped
    models = [m for m in all_configs if m not in args.skip_models]

    # Sort: largest models first (they're the bottleneck)
    model_order = {
        "gpt-oss-20b": 0, "llama-3.1-8b": 1, "gemma-3-4b-it": 2,
        "sarvam-2b": 3, "qwen2.5-1.5b": 4,
        "indicbert-v2": 5, "jhu-clsp-mmbert": 6,
        "mbert": 7, "xlm-roberta": 8, "muril": 9,
    }
    models.sort(key=lambda m: model_order.get(m, 99))

    vram_estimates = {
        "gpt-oss-20b": 45, "llama-3.1-8b": 40, "gemma-3-4b-it": 18,
        "sarvam-2b": 10, "qwen2.5-1.5b": 8,
        "indicbert-v2": 1.5, "jhu-clsp-mmbert": 2,
        "mbert": 1.5, "xlm-roberta": 2, "muril": 1.5,
    }
    total_vram = sum(vram_estimates.get(m, 5) for m in models) + len(models) * 2
    logger.info(f"Models: {models}")
    logger.info(f"Estimated peak VRAM: ~{total_vram:.0f} GB / 141 GB")
    logger.info(f"Each model runs: 3 languages × 3 seeds = 9 experiments sequentially")

    global_start = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=args.max_parallel) as executor:
        futures = {}
        for i, model_name in enumerate(models):
            if i > 0:
                time.sleep(args.stagger_seconds)
            logger.info(f"  🚀 Launching {model_name}")
            futures[executor.submit(run_injection_for_model, model_name)] = model_name

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            status_icon = "✅" if result["status"] == "success" else "❌"
            logger.info(
                f"  {status_icon} {result['model']}: {result['status']} "
                f"({result['elapsed_hours']:.2f} hrs)"
            )

    wall_time = (time.time() - global_start) / 3600
    logger.info(f"\nWall-clock time: {wall_time:.2f} hrs")
    logger.info(f"Total GPU-hours: {sum(r.get('elapsed_hours', 0) for r in results):.2f}")

    # Save summary
    summary_path = get_results_dir("phase1_injection") / "parallel_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "wall_clock_hours": wall_time,
            "results": results,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)

    failed = [r for r in results if r["status"] != "success"]
    if failed:
        logger.warning(f"\n⚠ {len(failed)} models failed! Check logs.")
        for r in failed:
            logger.warning(f"  {r['model']}: {r.get('log_file', 'N/A')}")
        sys.exit(1)

    logger.info("Phase 1 complete!")
    logger.info("Next: python scripts/05_parallel_removal.py")


def get_results_dir(name):
    """Local import to avoid circular at module level."""
    from src.utils.config import get_results_dir as _grd
    return _grd(name)


if __name__ == "__main__":
    main()
