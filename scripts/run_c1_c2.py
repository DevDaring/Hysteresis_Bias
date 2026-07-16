"""
Single entry point for the final experiment batch: robust C2 + reduced C1.

Order: C2 (loop area) runs first because it is the higher-value result;
C1 (threshold sweep) follows. Both are resume-capable, so re-running this
script after a crash or preemption continues where it stopped.

    python scripts/run_c1_c2.py --dry-run    # Gate 2: two instances, tiny budgets
    python scripts/run_c1_c2.py              # full batch (~11-13 h on one L4)

Recommended launch on the VM:
    nohup python -u scripts/run_c1_c2.py > ~/c1c2.log 2>&1 < /dev/null &

Outputs:
    results/loop_area/...  + loop_area_analysis.json + figure_loop_area
    results/theta_sensitivity/... + summary_by_theta.json + figure_theta_sensitivity
"""
import sys
import os
import argparse
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logging_setup import get_logger

logger = get_logger("run_c1_c2")
HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def run(script, *args):
    cmd = [PY, "-u", os.path.join(HERE, script), *args]
    logger.info(f">>> {' '.join(cmd)}")
    rc = subprocess.call(cmd)
    if rc != 0:
        logger.error(f"{script} exited with code {rc}")
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Gate 2 two-instance dry runs only")
    ap.add_argument("--skip-c2", action="store_true")
    ap.add_argument("--skip-c1", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        logger.info("=" * 60)
        logger.info("GATE 2 DRY RUNS (2 instances each, tiny budgets)")
        logger.info("=" * 60)
        rc2 = run("loop_area.py", "--dry-run")
        rc1 = run("theta_sensitivity.py", "--dry-run")
        ok = rc1 == 0 and rc2 == 0
        logger.info(f"GATE2 {'PASS' if ok else 'FAIL'} (loop_area rc={rc2}, theta rc={rc1})")
        sys.exit(0 if ok else 1)

    if not args.skip_c2:
        logger.info("#" * 60)
        logger.info("# C2 — loop-area grid (135 conditions)")
        logger.info("#" * 60)
        if run("loop_area.py") == 0:
            run("analyze_loop_area.py")

    if not args.skip_c1:
        logger.info("#" * 60)
        logger.info("# C1 — threshold sweep (4 new thetas x 90 conditions)")
        logger.info("#" * 60)
        run("theta_sensitivity.py")  # includes analyze() at the end

    logger.info("ALL DONE — pull results/, figures in Submission/images/")


if __name__ == "__main__":
    main()
