"""
Script 03_parallel: Phase 0 — Baseline Bias Measurement (10 Models in Parallel).

Launches up to 10 subprocesses — one per model — each measuring baseline bias
across all 3 languages. Merges results into baseline_results.json.

VRAM: ~60 GB total (all 10 models in inference mode) — fits in 141 GB H200.
Time: ~30-45 min.

Usage:
  python scripts/03_parallel_baseline.py                # All 10 in parallel
  python scripts/03_parallel_baseline.py --max-parallel 4
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

from src.utils.config import get_all_model_configs, get_enabled_model_configs, get_results_dir
from src.utils.logging_setup import get_logger

logger = get_logger("03_parallel_baseline")
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_baseline_for_model(model_name: str) -> dict:
    """Run baseline measurement for a single model (subprocess)."""
    start = time.time()

    # Write a small per-model script inline via -c
    script = f"""
import sys, os, json, torch
from datetime import datetime
sys.path.insert(0, r'{PROJECT_ROOT}')
from src.utils.config import get_all_model_configs, get_results_dir
from src.utils.seed import set_seed
from src.models.loader import load_model
from src.data.prepare_bias_injection import load_injection_data
from src.evaluation.bias_calculator import evaluate_bias

set_seed(42)
all_configs = get_all_model_configs()
model_config = all_configs['{model_name}']
model, tokenizer = load_model('{model_name}', model_config)
model.eval()
model_type = model_config['model_type']

results = {{}}
for lang in ['en', 'hi', 'bn']:
    eval_data = load_injection_data(lang, split='eval')
    with torch.no_grad():
        bias_result = evaluate_bias(model, tokenizer, model_type, eval_data, use_full_aul=True)
    results[lang] = {{
        'model': '{model_name}',
        'model_type': model_type,
        'language': lang,
        'metric': bias_result.get('metric', ''),
        'overall_bias_score': bias_result.get('overall_bias_score', 0.5),
        'categories': bias_result.get('categories', {{}}),
        'timestamp': datetime.now().isoformat(),
    }}

out_path = get_results_dir('phase0_baseline') / 'baseline_{model_name}.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
del model
torch.cuda.empty_cache()
"""

    log_dir = PROJECT_ROOT / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"baseline_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    try:
        with open(log_file, "w") as lf:
            proc = subprocess.run(
                [sys.executable, "-c", script],
                cwd=str(PROJECT_ROOT),
                stdout=lf, stderr=subprocess.STDOUT,
                timeout=7200,
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
    parser = argparse.ArgumentParser(description="Phase 0: Parallel Baseline")
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--skip-models", nargs="+", default=[])
    parser.add_argument("--stagger-seconds", type=int, default=5)
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("PHASE 0: PARALLEL BASELINE MEASUREMENT")
    logger.info("=" * 60)

    all_configs = get_enabled_model_configs()
    models = [m for m in all_configs if m not in args.skip_models]

    logger.info(f"Running {len(models)} models in parallel (max {args.max_parallel})")

    global_start = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=args.max_parallel) as executor:
        futures = {}
        for i, model_name in enumerate(models):
            if i > 0:
                time.sleep(args.stagger_seconds)
            logger.info(f"  🚀 Launching {model_name}")
            futures[executor.submit(run_baseline_for_model, model_name)] = model_name

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            logger.info(f"  ✅ {result['model']}: {result['status']} ({result['elapsed_hours']:.2f} hrs)")

    # Merge per-model results into single file
    merged = {}
    baseline_dir = get_results_dir("phase0_baseline")
    for model_name in models:
        part_file = baseline_dir / f"baseline_{model_name}.json"
        if part_file.exists():
            with open(part_file) as f:
                merged[model_name] = json.load(f)

    out_path = baseline_dir / "baseline_results.json"
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    wall_time = (time.time() - global_start) / 3600
    logger.info(f"\nWall-clock time: {wall_time:.2f} hrs")
    logger.info(f"Merged baseline saved to {out_path}")
    logger.info("Next: python scripts/04_parallel_injection.py")


if __name__ == "__main__":
    main()
