"""
Same-grid A/B analysis: matched vs mismatched objective.

Reads results/wp1_symmetric/summary.json (matched) and
results/wp1_mismatched/summary.json (mismatched). For each objective reports
censoring accounting, median R with model-clustered bootstrap CI, share R>1,
and a one-sample Wilcoxon on log R. Prints a comparison block for the paper.

    python scripts/analyze_ab_compare.py
"""

import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import stats
from src.utils.config import get_results_dir


def bootstrap_median_ci(cells, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    by_model = {}
    for c in cells:
        by_model.setdefault(c["model"], []).append(c["R"])
    models = list(by_model.keys())
    if not models:
        return None, None
    meds = []
    for _ in range(n_boot):
        drawn = rng.choice(len(models), size=len(models), replace=True)
        pool = []
        for i in drawn:
            pool.extend(by_model[models[i]])
        if pool:
            meds.append(np.median(pool))
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def summarise(root):
    data = json.load(open(get_results_dir(root) / "summary.json"))
    data = [d for d in data if isinstance(d, dict) and "R_undefined_reason" in d]
    converged = [d for d in data if d.get("R") is not None and not d.get("baseline_above_theta")
                 and d.get("R_undefined_reason") in (None, "removal_censored_lower_bound")]
    inj_cens = [d for d in data if d.get("R_undefined_reason") == "injection_did_not_converge"]
    above = [d for d in data if d.get("baseline_above_theta")]
    R = [d["R"] for d in converged]
    logR = [math.log(r) for r in R if r > 0]
    w_p = None
    if len(logR) >= 6:
        try:
            w_p = float(stats.wilcoxon(logR, alternative="two-sided").pvalue)
        except Exception:
            w_p = None
    lo, hi = bootstrap_median_ci(converged)
    # ceiling artefact count: original protocol counts injection+removal censored as R=4.0.
    # Here injection-censored are the analogue that were mis-counted; report their number.
    return {
        "n_total": len(data),
        "n_converged": len(converged),
        "n_injection_censored": len(inj_cens),
        "n_baseline_above_theta": len(above),
        "censoring_rate": round(len(inj_cens) / len(data), 3) if data else None,
        "median_R": round(float(np.median(R)), 3) if R else None,
        "median_R_CI95": [round(lo, 3), round(hi, 3)] if lo is not None else None,
        "frac_R_gt1": round(sum(1 for r in R if r > 1) / len(R), 3) if R else None,
        "n_R_gt1": int(sum(1 for r in R if r > 1)),
        "wilcoxon_logR_p": w_p,
    }


def main():
    out = {"matched": summarise("wp1_symmetric"),
           "mismatched": summarise("wp1_mismatched")}
    path = get_results_dir("wp1_symmetric") / "ab_comparison.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("AB_COMPARE_JSON_START")
    print(json.dumps(out, indent=2))
    print("AB_COMPARE_JSON_END")


if __name__ == "__main__":
    main()
