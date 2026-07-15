"""
Part C1 — threshold sensitivity sweep (scaffold; run on the user's GPU).

Re-runs the matched-objective grid at several thresholds and reports median R
per threshold. Pre-empts the objection that the result depends on theta = 0.7.
Reuses the existing WP1 harness; nothing here fabricates results.

    python scripts/theta_sensitivity.py            # thetas 0.60..0.80
Reads/writes under results/theta_sensitivity/<theta>/... via run_grid, and the
TeX table it feeds is in Submission/EJAI_Hysteresis.tex (see the commented
C1 stub). Fill the stub from results/theta_sensitivity/summary_by_theta.json
after this completes.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scripts.wp1_symmetric_control import run_grid, ENCODERS
from src.utils.config import get_results_dir

THETAS = [0.60, 0.65, 0.70, 0.75, 0.80]
CATS = ["race-color", "religion", "nationality", "socioeconomic"]


def main():
    by_theta = {}
    for theta in THETAS:
        # run_grid writes per-condition JSON under results/wp1_symmetric; to keep
        # thresholds separate, point RESULTS at a theta-specific root by overriding
        # the objective label is not enough — use a dedicated dir per theta.
        rows = run_grid(models=ENCODERS, languages=["en", "hi", "bn"], categories=CATS,
                        seeds=[42, 123, 456], theta=theta,
                        max_inject=500, max_remove=1000, eval_every=25, objective="matched")
        Rs = [d["R"] for d in rows if d.get("R") is not None and not d.get("baseline_above_theta")
              and d.get("R_undefined_reason") in (None, "removal_censored_lower_bound")]
        by_theta[theta] = {"n": len(Rs), "median_R": float(np.median(Rs)) if Rs else None}
    out = get_results_dir("theta_sensitivity") / "summary_by_theta.json"
    with open(out, "w") as f:
        json.dump(by_theta, f, indent=2)
    print("SAVED", out)
    print(json.dumps(by_theta, indent=2))


if __name__ == "__main__":
    print("NOTE: this re-runs the full matched grid five times; run on GPU only.")
    main()
