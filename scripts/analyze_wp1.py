"""
WP1 analysis — pre-registered tests for the symmetric-objective control.

Reads results/wp1_symmetric/summary.json (180 conditions) and reports:
  H1  median R > 1 across converged cells (one-sample Wilcoxon on log R vs 0,
      two-sided, alpha=0.05); model-clustered bootstrap 95% CI on median R.
  Censoring accounting (converged / injection-censored / baseline-above-theta).
  Per-model and per-language R summaries.
  Gradient-norm asymmetry between inject and remove directions (confound check).

Emits a single JSON verdict block for the paper.

    python scripts/analyze_wp1.py
"""

import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import stats
from src.utils.config import get_results_dir


def median(xs):
    return float(np.median(xs)) if xs else None


def model_clustered_bootstrap_median(cells, n_boot=10000, seed=0):
    """95% CI on the median R, resampling at the MODEL level to respect nesting."""
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
    lo, hi = np.percentile(meds, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    summ_path = get_results_dir("wp1_symmetric") / "summary.json"
    with open(summ_path) as f:
        data = json.load(f)

    # Keep only well-formed condition results
    data = [d for d in data if "R_undefined_reason" in d or d.get("R") is not None]

    converged = [d for d in data if d.get("R") is not None and not d.get("baseline_above_theta")
                 and d.get("R_undefined_reason") in (None, "removal_censored_lower_bound")]
    inj_censored = [d for d in data if d.get("R_undefined_reason") == "injection_did_not_converge"]
    above = [d for d in data if d.get("baseline_above_theta")]

    R = [d["R"] for d in converged]
    logR = [math.log(r) for r in R if r > 0]

    # H1: one-sample Wilcoxon on log R vs 0 (two-sided)
    wilcoxon_p = None
    wilcoxon_stat = None
    if len(logR) >= 6:
        try:
            w = stats.wilcoxon(logR, alternative="two-sided", zero_method="wilcox")
            wilcoxon_stat, wilcoxon_p = float(w.statistic), float(w.pvalue)
        except Exception as e:
            wilcoxon_p = f"error: {e}"

    lo, hi = model_clustered_bootstrap_median(converged)

    # Per-model
    per_model = {}
    for d in converged:
        per_model.setdefault(d["model"], []).append(d["R"])
    per_model_summary = {
        m: {"n": len(rs), "median_R": round(median(rs), 3),
            "n_R_gt1": int(sum(1 for r in rs if r > 1)),
            "frac_R_gt1": round(sum(1 for r in rs if r > 1) / len(rs), 3)}
        for m, rs in sorted(per_model.items())
    }

    # Per-language
    per_lang = {}
    for d in converged:
        per_lang.setdefault(d["language"], []).append(d["R"])
    per_lang_summary = {
        L: {"n": len(rs), "median_R": round(median(rs), 3),
            "n_R_gt1": int(sum(1 for r in rs if r > 1)),
            "frac_R_gt1": round(sum(1 for r in rs if r > 1) / len(rs), 3)}
        for L, rs in sorted(per_lang.items())
    }

    # Gradient-norm asymmetry (confound visibility): compare inject vs remove grad norms
    gi = [d["grad_norm_inject_mean"] for d in data if d.get("grad_norm_inject_mean")]
    gr = [d["grad_norm_remove_mean"] for d in data if d.get("grad_norm_remove_mean")]
    grad_ratio = None
    grad_p = None
    if gi and gr:
        paired = [(d["grad_norm_inject_mean"], d["grad_norm_remove_mean"])
                  for d in data if d.get("grad_norm_inject_mean") and d.get("grad_norm_remove_mean")]
        gi_p = [a for a, b in paired]
        gr_p = [b for a, b in paired]
        grad_ratio = round(float(np.median([b / a for a, b in paired if a > 0])), 3)
        try:
            grad_p = float(stats.wilcoxon(gi_p, gr_p, alternative="two-sided").pvalue)
        except Exception:
            grad_p = None

    verdict = {
        "n_total": len(data),
        "n_converged": len(converged),
        "n_injection_censored": len(inj_censored),
        "n_baseline_above_theta": len(above),
        "median_R_converged": round(median(R), 3) if R else None,
        "median_R_model_clustered_CI95": [round(lo, 3), round(hi, 3)] if lo is not None else None,
        "frac_R_gt1_converged": round(sum(1 for r in R if r > 1) / len(R), 3) if R else None,
        "n_R_gt1": int(sum(1 for r in R if r > 1)),
        "H1_wilcoxon_logR_stat": wilcoxon_stat,
        "H1_wilcoxon_logR_p_twosided": wilcoxon_p,
        "grad_norm_remove_over_inject_median": grad_ratio,
        "grad_norm_direction_wilcoxon_p": grad_p,
        "per_model": per_model_summary,
        "per_language": per_lang_summary,
    }

    out = get_results_dir("wp1_symmetric") / "wp1_analysis.json"
    with open(out, "w") as f:
        json.dump(verdict, f, indent=2)

    print("WP1_ANALYSIS_JSON_START")
    print(json.dumps(verdict, indent=2))
    print("WP1_ANALYSIS_JSON_END")


if __name__ == "__main__":
    main()
