"""
Script 07_parallel: Phase 4a — Hessian Analysis (2 Models in Parallel).

Launches 2 subprocesses for focus models (Llama-3.1-8B + MuRIL).
VRAM: ~42 GB — fits easily in 141 GB H200.
Time: ~4-6 hrs (vs ~6-9 hrs sequential).

Usage:
  python scripts/07_parallel_hessian.py
"""

import sys
import os
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import get_results_dir
from src.utils.logging_setup import get_logger

logger = get_logger("07_parallel_hessian")
PROJECT_ROOT = Path(__file__).resolve().parent.parent

FOCUS_MODELS = ["llama-3.1-8b", "muril"]


def run_hessian_for_model(model_name: str) -> dict:
    """Run Hessian analysis for a single model."""
    start = time.time()

    script = f"""
import sys, os, json, torch
from datetime import datetime
sys.path.insert(0, r'{PROJECT_ROOT}')
from src.utils.config import get_all_model_configs, get_results_dir
from src.utils.seed import set_seed
from src.models.loader import load_lora_checkpoint
from src.models.causal_wrapper import CausalModelWrapper
from src.models.encoder_wrapper import EncoderModelWrapper
from src.data.prepare_bias_injection import load_injection_data
from src.analysis.hessian_analysis import compute_top_k_eigenvalues, hutchinson_trace_estimate
from src.utils.logging_setup import get_logger

logger = get_logger('07_hessian_{model_name}')
all_configs = get_all_model_configs()
model_config = all_configs['{model_name}']
model_type = model_config['model_type']
language = 'en'
seed = 42
set_seed(seed)

eval_data = load_injection_data(language, split='eval')
results = {{}}

for checkpoint_type in ['biased', 'debiased']:
    logger.info(f'Analyzing {{checkpoint_type}} checkpoint...')
    if checkpoint_type == 'biased':
        ckpt = get_results_dir('phase1_injection') / '{model_name}' / language / f'seed{{seed}}' / 'final_biased'
    else:
        ckpt = get_results_dir('phase2_removal') / '{model_name}' / language / f'seed{{seed}}' / 'final_debiased'
    if not ckpt.exists():
        logger.warning(f'Checkpoint not found: {{ckpt}}')
        continue
    model, tokenizer = load_lora_checkpoint('{model_name}', str(ckpt), model_config)
    if model_type == 'causal':
        wrapper = CausalModelWrapper(model, tokenizer)
        def loss_fn(m, batch):
            texts = [ex['text'] for ex in batch]
            return wrapper.compute_injection_loss(texts[:4])
    else:
        wrapper = EncoderModelWrapper(model, tokenizer)
        def loss_fn(m, batch):
            texts = [ex['masked_text'] for ex in batch]
            targets = [ex['stereo_target'] for ex in batch]
            return wrapper.compute_injection_loss(texts[:4], targets[:4])
    data_loader = [eval_data[:16]]
    eigenvalues = compute_top_k_eigenvalues(model, data_loader, loss_fn, k=5, num_iterations=50)
    trace = hutchinson_trace_estimate(
        model, data_loader, loss_fn,
        params=[p for n, p in model.named_parameters() if 'lora' in n and p.requires_grad],
        num_samples=20,
    )
    key = f'{model_name}_{{checkpoint_type}}'
    results[key] = {{
        'model': '{model_name}', 'checkpoint_type': checkpoint_type,
        'top_5_eigenvalues': eigenvalues, 'trace_estimate': trace,
        'timestamp': datetime.now().isoformat(),
    }}
    logger.info(f'  Eigenvalues: {{eigenvalues}}')
    logger.info(f'  Trace: {{trace:.6f}}')
    del model
    torch.cuda.empty_cache()

out_path = get_results_dir('phase4_geometry') / 'hessian_{model_name}.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
logger.info('Done with {model_name}')
"""

    log_dir = PROJECT_ROOT / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"hessian_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    try:
        with open(log_file, "w") as lf:
            proc = subprocess.run(
                [sys.executable, "-c", script],
                cwd=str(PROJECT_ROOT),
                stdout=lf, stderr=subprocess.STDOUT,
                timeout=36000,
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
    logger.info("=" * 60)
    logger.info("PHASE 4a: PARALLEL HESSIAN ANALYSIS")
    logger.info("=" * 60)

    global_start = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = {}
        for i, model_name in enumerate(FOCUS_MODELS):
            if i > 0:
                time.sleep(15)
            logger.info(f"  🚀 Launching {model_name}")
            futures[executor.submit(run_hessian_for_model, model_name)] = model_name

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            icon = "✅" if result["status"] == "success" else "❌"
            logger.info(f"  {icon} {result['model']}: {result['status']} ({result['elapsed_hours']:.2f} hrs)")

    # Merge per-model results
    merged = {}
    geom_dir = get_results_dir("phase4_geometry")
    for model_name in FOCUS_MODELS:
        part_file = geom_dir / f"hessian_{model_name}.json"
        if part_file.exists():
            with open(part_file) as f:
                merged.update(json.load(f))

    out_path = geom_dir / "hessian_results.json"
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)

    wall_time = (time.time() - global_start) / 3600
    logger.info(f"\nWall-clock time: {wall_time:.2f} hrs")
    logger.info(f"Merged results saved to {out_path}")
    logger.info("Next: python scripts/08_linear_connectivity.py")


if __name__ == "__main__":
    main()
