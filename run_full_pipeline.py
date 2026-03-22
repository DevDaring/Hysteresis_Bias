#!/usr/bin/env python3
"""
FULL PIPELINE LAUNCHER — Runs entire Bias Hysteresis experiment end-to-end.

Pipeline order:
  ┌─────────────────────────────────────────────────┐
  │  STAGE 1: SETUP & DATA                          │
  │  00_setup.sh → 01_download → 02_dry_run         │
  ├─────────────────────────────────────────────────┤
  │  STAGE 2: MAIN EXPERIMENTS (10 models parallel) │
  │  03 baseline → 04 injection → 05 removal         │
  │  → 06 asymmetry → 07 hessian → 08 connectivity  │
  │  → 09 cultural                                   │
  ├─────────────────────────────────────────────────┤
  │  STAGE 3: COMPARATIVE STUDIES (10 models)        │
  │  10 comparatives → 11 comparative R              │
  ├─────────────────────────────────────────────────┤
  │  STAGE 4: OUTPUTS                               │
  │  12 figures → 13 tables                          │
  └─────────────────────────────────────────────────┘

Usage:
  python run_full_pipeline.py                    # Full run (parallel mode)
  python run_full_pipeline.py --sequential       # Sequential mode (small GPUs)
  python run_full_pipeline.py --skip-setup       # Skip setup (already done)
  python run_full_pipeline.py --start-from 05    # Resume from script 05
  python run_full_pipeline.py --skip-comparatives # Skip Phase 5C
  python run_full_pipeline.py --max-parallel 3   # Limit parallelism

Total time: ~16-23 hours on H200 (parallel) | ~42-57 hours on H100 (sequential)
"""

import sys
import os
import subprocess
import time
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Pipeline stages — order matters!
PIPELINE = [
    # (script_id, script_name, parallel_script, description, stage)
    ("00", "scripts/00_setup.sh",                 None,                                  "Install deps + Flash Attn",     "SETUP"),
    ("01", "scripts/01_download_data.py",          None,                                  "Download + validate datasets",  "SETUP"),
    ("02", "scripts/02_dry_run.py",                None,                                  "Mandatory dry run",             "SETUP"),
    ("03", "scripts/03_baseline.py",               "scripts/03_parallel_baseline.py",      "Phase 0: Baseline measurement", "MAIN"),
    ("04", "scripts/04_bias_injection.py",         "scripts/04_parallel_injection.py",     "Phase 1: Bias injection",       "MAIN"),
    ("05", "scripts/05_bias_removal.py",           "scripts/05_parallel_removal.py",       "Phase 2: Bias removal",         "MAIN"),
    ("06", "scripts/06_compute_asymmetry.py",      None,                                  "Phase 3: Compute R (CPU)",      "MAIN"),
    ("07", "scripts/07_hessian_analysis.py",       "scripts/07_parallel_hessian.py",       "Phase 4a: Hessian analysis",    "MAIN"),
    ("08", "scripts/08_linear_connectivity.py",    None,                                  "Phase 4b: Linear connectivity", "MAIN"),
    ("09", "scripts/09_cultural_analysis.py",      None,                                  "Phase 6: Cultural analysis",    "MAIN"),
    ("10", "scripts/10_comparatives.py",           "scripts/10_parallel_comparatives.py",  "Phase 5C: Comparatives",        "COMPARATIVE"),
    ("11", "scripts/11_comparative_asymmetry.py",  None,                                  "Phase 5C: Comparative R",       "COMPARATIVE"),
    ("12", "scripts/12_generate_figures.py",       None,                                  "Generate figures",              "OUTPUT"),
    ("13", "scripts/13_generate_tables.py",        None,                                  "Generate tables",               "OUTPUT"),
    ("14", "scripts/14_qualitative_outputs.py",    None,                                  "Qualitative outputs",           "OUTPUT"),
]


def run_step(script_path: str, step_id: str, description: str, max_parallel: int = 6) -> dict:
    """Run a single pipeline step and return status."""
    full_path = PROJECT_ROOT / script_path
    if not full_path.exists():
        return {"status": "skipped", "reason": f"File not found: {full_path}"}

    start = time.time()
    print(f"\n{'='*70}")
    print(f"  [{step_id}] {description}")
    print(f"  Script: {script_path}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    # Determine command
    if script_path.endswith(".sh"):
        cmd = ["bash", str(full_path)]
    else:
        cmd = [sys.executable, str(full_path)]

    # Pass --max-parallel to parallel scripts
    if "parallel" in script_path and max_parallel != 6:
        cmd.extend(["--max-parallel", str(max_parallel)])

    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            timeout=86400,  # 24-hour max per step
        )

        elapsed = time.time() - start
        status = "success" if result.returncode == 0 else "failed"

        print(f"\n  [{step_id}] {status.upper()} in {_format_duration(elapsed)}")

        return {
            "step_id": step_id,
            "script": script_path,
            "status": status,
            "returncode": result.returncode,
            "elapsed_seconds": elapsed,
        }

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        print(f"\n  [{step_id}] TIMEOUT after {_format_duration(elapsed)}")
        return {
            "step_id": step_id, "script": script_path,
            "status": "timeout", "elapsed_seconds": elapsed,
        }
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n  [{step_id}] ERROR: {e}")
        return {
            "step_id": step_id, "script": script_path,
            "status": "error", "error": str(e), "elapsed_seconds": elapsed,
        }


def _format_duration(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f} min"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def main():
    parser = argparse.ArgumentParser(
        description="Bias Hysteresis Full Pipeline Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_full_pipeline.py                       # Full parallel run
  python run_full_pipeline.py --sequential          # For smaller GPUs
  python run_full_pipeline.py --start-from 05       # Resume from Phase 2
  python run_full_pipeline.py --skip-comparatives   # Main experiments only
  python run_full_pipeline.py --skip-setup          # Skip 00/01/02
        """
    )
    parser.add_argument("--sequential", action="store_true",
                        help="Use sequential scripts (for GPUs < 80 GB)")
    parser.add_argument("--start-from", type=str, default="00",
                        help="Resume from this script ID (e.g., '05')")
    parser.add_argument("--skip-setup", action="store_true",
                        help="Skip setup, download, and dry run (00-02)")
    parser.add_argument("--skip-comparatives", action="store_true",
                        help="Skip Phase 5C comparative studies (10-11)")
    parser.add_argument("--skip-outputs", action="store_true",
                        help="Skip figure/table generation (12-13)")
    parser.add_argument("--max-parallel", type=int, default=4,
                        help="Max parallel models (forwarded to parallel scripts)")
    parser.add_argument("--stop-on-failure", action="store_true", default=True,
                        help="Stop pipeline if any step fails (default: True)")
    parser.add_argument("--continue-on-failure", action="store_true",
                        help="Continue pipeline even if a step fails")

    args = parser.parse_args()

    if args.continue_on_failure:
        args.stop_on_failure = False

    print("╔══════════════════════════════════════════════════════════╗")
    print("║    BIAS HYSTERESIS PRINCIPLE — FULL PIPELINE LAUNCHER   ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  Mode:       {'Sequential' if args.sequential else 'Parallel (4 enabled models)':40s} ║")
    print(f"║  Start from: {args.start_from:40s}            ║")
    print(f"║  Max parallel: {args.max_parallel:<38d} ║")
    print(f"║  Started:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):40s} ║")
    print("╚══════════════════════════════════════════════════════════╝")

    global_start = time.time()
    step_results = []

    for step_id, seq_script, par_script, description, stage in PIPELINE:
        # Skip logic
        if step_id < args.start_from:
            continue
        if args.skip_setup and stage == "SETUP":
            print(f"  ⊘ Skipping [{step_id}] {description} (--skip-setup)")
            continue
        if args.skip_comparatives and stage == "COMPARATIVE":
            print(f"  ⊘ Skipping [{step_id}] {description} (--skip-comparatives)")
            continue
        if args.skip_outputs and stage == "OUTPUT":
            print(f"  ⊘ Skipping [{step_id}] {description} (--skip-outputs)")
            continue

        # Choose parallel or sequential script
        if args.sequential or par_script is None:
            script = seq_script
        else:
            script = par_script

        result = run_step(script, step_id, description, args.max_parallel)
        step_results.append(result)

        # Stop on failure?
        if result["status"] not in ("success", "skipped") and args.stop_on_failure:
            print(f"\n❌ PIPELINE STOPPED at step [{step_id}]: {description}")
            print(f"   Status: {result['status']}")
            print(f"   Re-run with: python run_full_pipeline.py --start-from {step_id}")
            break

    # Final summary
    global_elapsed = time.time() - global_start

    print(f"\n{'═'*70}")
    print(f"  PIPELINE SUMMARY")
    print(f"{'═'*70}")
    print(f"  Total wall-clock time: {_format_duration(global_elapsed)}")
    print(f"  Steps run: {len(step_results)}")
    print()

    for r in step_results:
        icon = {"success": "✅", "failed": "❌", "timeout": "⏰", "skipped": "⊘", "error": "💥"}.get(r["status"], "?")
        elapsed = _format_duration(r.get("elapsed_seconds", 0))
        print(f"  {icon} [{r.get('step_id', '??')}] {r.get('script', '???'):45s} {elapsed:>10s}  {r['status']}")

    succeeded = sum(1 for r in step_results if r["status"] == "success")
    failed = sum(1 for r in step_results if r["status"] not in ("success", "skipped"))

    print(f"\n  Result: {succeeded} succeeded, {failed} failed")

    if failed == 0 and succeeded > 0:
        print("\n  🎉 PIPELINE COMPLETE! All experiments finished successfully.")
        print("  Check results/ for outputs, figures, and tables.")
    elif failed > 0:
        first_fail = next(r for r in step_results if r["status"] not in ("success", "skipped"))
        print(f"\n  Resume with: python run_full_pipeline.py --start-from {first_fail.get('step_id', '00')}")

    # Save pipeline log
    log_path = PROJECT_ROOT / "results" / "pipeline_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump({
            "wall_clock_hours": global_elapsed / 3600,
            "mode": "sequential" if args.sequential else "parallel",
            "max_parallel": args.max_parallel,
            "steps": step_results,
            "started": (datetime.now() - timedelta(seconds=global_elapsed)).isoformat(),
            "finished": datetime.now().isoformat(),
        }, f, indent=2)

    print(f"\n  Pipeline log: {log_path}")

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
