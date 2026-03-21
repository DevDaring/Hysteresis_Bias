"""
Script 10_parallel: Phase 5C — Comparative Debiasing Studies.

Architecture:
  Methods run SEQUENTIALLY (one at a time).
  Within each method, 6 models run IN PARALLEL.

  Step 1: C1 CDA [11]          → 6 models in parallel  (~1.5-2 hrs)
  Step 2: C2 Self-Debias [12]  → 3 causal models only   (~0.3-0.5 hrs)
  Step 3: C3 INLP [13]         → 6 models in parallel  (~0.5-1 hr)
  Step 4: C4 DAMA [14]         → 3 causal models only   (~0.5-1 hr)
  Step 5: C5 BiasEdit [15]     → 6 models in parallel  (~2-3 hrs)
  Step 6: C6 Gradient Ascent [16] → 6 models in parallel (~1.5-2 hrs)
                                                Total: ~6-9.5 hrs

Benefits:
  - Clear per-method timing and logging
  - Can skip/resume at method level (--start-from c3_inlp)
  - Lower peak VRAM (one method at a time, models share GPU)
  - Easier debugging if a specific method fails

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
  python scripts/10_parallel_comparatives.py                     # All methods, all models
  python scripts/10_parallel_comparatives.py --start-from c3_inlp  # Resume from C3
  python scripts/10_parallel_comparatives.py --skip-methods c4_dama c5_biasedit
  python scripts/10_parallel_comparatives.py --skip-models mbert xlm-roberta
  python scripts/10_parallel_comparatives.py --max-parallel 3

GPU: 1× H200 (141 GB VRAM)
Time: ~6-9.5 hrs total
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

logger = get_logger("10_parallel_comparatives")
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Methods in execution order — (method_id, display_name, causal_only)
METHODS_ORDER = [
    ("c1_cda",             "C1: CDA [11] Zmigrod et al. (2019)",             False),
    ("c2_self_debias",     "C2: Self-Debias [12] Schick et al. (2021)",     True),
    ("c3_inlp",            "C3: INLP [13] Ravfogel et al. (2020)",          False),
    ("c4_dama",            "C4: DAMA [14] Limisiewicz et al. (2024)",        True),
    ("c5_biasedit",        "C5: BiasEdit [15] Xu et al. (2025)",             False),
    ("c6_gradient_ascent", "C6: Gradient Ascent [16] Liu et al. (2025)",     False),
]


def run_method_for_model(model_name: str, method_id: str) -> dict:
    """
    Run a single comparative method on a single model (subprocess).

    Each subprocess calls scripts/10_comparatives.py with --only-models
    and --skip-methods to isolate exactly one method × one model.
    """
    start = time.time()

    # Build skip list: skip all methods except the target one
    all_method_ids = [m[0] for m in METHODS_ORDER]
    skip_methods = [m for m in all_method_ids if m != method_id]

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "10_comparatives.py"),
        "--only-models", model_name,
        "--skip-methods", *skip_methods,
    ]

    log_dir = PROJECT_ROOT / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"cs_{method_id}_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    try:
        with open(log_file, "w") as lf:
            proc = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=lf, stderr=subprocess.STDOUT,
                timeout=36000,  # 10-hour timeout per model
            )
        elapsed = time.time() - start
        return {
            "model": model_name,
            "method": method_id,
            "status": "success" if proc.returncode == 0 else "failed",
            "returncode": proc.returncode,
            "elapsed_seconds": elapsed,
            "elapsed_hours": elapsed / 3600,
            "log_file": str(log_file),
        }
    except subprocess.TimeoutExpired:
        return {
            "model": model_name, "method": method_id,
            "status": "timeout",
            "elapsed_hours": (time.time() - start) / 3600,
            "log_file": str(log_file),
        }
    except Exception as e:
        return {
            "model": model_name, "method": method_id,
            "status": "error", "error": str(e),
            "elapsed_hours": (time.time() - start) / 3600,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Phase 5C: Sequential Methods, Parallel Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--max-parallel", type=int, default=6,
                        help="Max models per method (default: 6)")
    parser.add_argument("--skip-models", nargs="+", default=[],
                        help="Models to skip")
    parser.add_argument("--only-models", nargs="+", default=[],
                        help="Run only these models")
    parser.add_argument("--skip-methods", nargs="+", default=[],
                        help="Methods to skip (e.g., c4_dama c5_biasedit)")
    parser.add_argument("--start-from", type=str, default=None,
                        help="Resume from this method (e.g., c3_inlp)")
    parser.add_argument("--stagger-seconds", type=int, default=15,
                        help="Seconds between model launches (default: 15)")
    parser.add_argument("--stop-on-failure", action="store_true",
                        help="Stop entire pipeline if any model fails for a method")
    args = parser.parse_args()

    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║  PHASE 5C: COMPARATIVE DEBIASING STUDIES                ║")
    logger.info("║  Architecture: Methods SEQUENTIAL → Models PARALLEL     ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")

    # Resolve models
    all_configs = get_all_model_configs()
    training_config = load_training_config()
    comp_config = training_config.get("comparatives", {})
    config_enabled_models = comp_config.get("enabled_models", {})
    config_enabled_methods = comp_config.get("enabled_methods", {})

    if args.only_models:
        models = {m: all_configs[m]["model_type"] for m in args.only_models if m in all_configs}
    else:
        models = {
            m: cfg["model_type"] for m, cfg in all_configs.items()
            if config_enabled_models.get(m, True) and m not in args.skip_models
        }

    # Resolve methods
    methods = []
    started = args.start_from is None
    for method_id, display_name, causal_only in METHODS_ORDER:
        if not started:
            if method_id == args.start_from:
                started = True
            else:
                continue
        if method_id in args.skip_methods:
            continue
        if not config_enabled_methods.get(method_id, True):
            continue
        methods.append((method_id, display_name, causal_only))

    # Log plan
    logger.info(f"\nModels ({len(models)}):")
    for name, mtype in models.items():
        logger.info(f"  • {name} ({mtype})")

    logger.info(f"\nMethods ({len(methods)}) — running SEQUENTIALLY:")
    for method_id, display, causal in methods:
        applicable = len(models) if not causal else sum(1 for t in models.values() if t == "causal")
        logger.info(f"  {display}  →  {applicable} models in parallel")

    global_start = time.time()
    all_results = []
    method_timings = []

    for method_idx, (method_id, display_name, causal_only) in enumerate(methods):
        # Filter models for this method
        if causal_only:
            method_models = {m: t for m, t in models.items() if t == "causal"}
        else:
            method_models = models.copy()

        if not method_models:
            logger.info(f"\n  ⊘ Skipping {display_name} — no applicable models")
            continue

        method_start = time.time()

        logger.info(f"\n{'='*70}")
        logger.info(f"  [{method_idx+1}/{len(methods)}] {display_name}")
        logger.info(f"  Models: {list(method_models.keys())} ({len(method_models)} parallel)")
        logger.info(f"  Started: {datetime.now().strftime('%H:%M:%S')}")
        logger.info(f"{'='*70}")

        # Launch all models for this method in parallel
        method_results = []
        with ProcessPoolExecutor(max_workers=min(args.max_parallel, len(method_models))) as executor:
            futures = {}
            for i, model_name in enumerate(method_models):
                if i > 0:
                    time.sleep(args.stagger_seconds)
                logger.info(f"  🚀 Launching {model_name} for {method_id}")
                future = executor.submit(run_method_for_model, model_name, method_id)
                futures[future] = model_name

            for future in as_completed(futures):
                result = future.result()
                method_results.append(result)
                all_results.append(result)
                icon = "✅" if result["status"] == "success" else "❌"
                logger.info(
                    f"  {icon} {result['model']}: {result['status']} "
                    f"({result.get('elapsed_hours', 0):.2f} hrs)"
                )

        method_elapsed = time.time() - method_start
        method_timings.append({
            "method": method_id,
            "display_name": display_name,
            "models_run": len(method_results),
            "succeeded": sum(1 for r in method_results if r["status"] == "success"),
            "failed": sum(1 for r in method_results if r["status"] != "success"),
            "wall_clock_hours": method_elapsed / 3600,
        })

        succeeded = sum(1 for r in method_results if r["status"] == "success")
        failed = sum(1 for r in method_results if r["status"] != "success")
        logger.info(
            f"\n  {display_name}: {succeeded} ✅  {failed} ❌  "
            f"({method_elapsed/3600:.2f} hrs wall-clock)"
        )

        # Stop on failure?
        if failed > 0 and args.stop_on_failure:
            logger.error(f"\n  Pipeline stopped at {method_id} due to --stop-on-failure")
            logger.error(f"  Resume with: --start-from {method_id}")
            break

    # Final summary
    global_elapsed = time.time() - global_start

    logger.info(f"\n{'═'*70}")
    logger.info(f"  COMPARATIVE STUDIES — FINAL SUMMARY")
    logger.info(f"{'═'*70}")
    logger.info(f"  Total wall-clock: {global_elapsed/3600:.2f} hrs")
    logger.info(f"  Total GPU-hours:  {sum(r.get('elapsed_hours', 0) for r in all_results):.2f}")
    logger.info(f"\n  Per-method breakdown:")
    for mt in method_timings:
        logger.info(
            f"    {mt['display_name']:50s} "
            f"{mt['wall_clock_hours']:.2f} hrs  "
            f"({mt['succeeded']}/{mt['models_run']} models)"
        )

    total_succeeded = sum(1 for r in all_results if r["status"] == "success")
    total_failed = sum(1 for r in all_results if r["status"] != "success")
    logger.info(f"\n  Total: {total_succeeded} ✅  {total_failed} ❌")

    # Save summary
    summary_path = get_results_dir("phase5c_comparatives") / "parallel_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({
            "architecture": "methods_sequential__models_parallel",
            "wall_clock_hours": global_elapsed / 3600,
            "total_gpu_hours": sum(r.get("elapsed_hours", 0) for r in all_results),
            "method_timings": method_timings,
            "all_results": all_results,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2, default=str)

    if total_failed > 0:
        logger.info(f"\n  ⚠ Some runs failed. Check logs in results/logs/")
        first_fail_method = next(
            (r["method"] for r in all_results if r["status"] != "success"), None
        )
        if first_fail_method:
            logger.info(f"  Resume with: --start-from {first_fail_method}")
        sys.exit(1)

    logger.info(f"\n  Summary: {summary_path}")
    logger.info("  Next: python scripts/11_comparative_asymmetry.py")


if __name__ == "__main__":
    main()
