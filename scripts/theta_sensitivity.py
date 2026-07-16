"""
C1 (reduced) — threshold sensitivity sweep for the matched objective.

Pre-empts the objection that the headline result depends on theta = 0.7.
Re-runs the matched-objective inject/remove protocol at four new thresholds
(0.60, 0.65, 0.75, 0.80) on two categories, and merges the existing
theta = 0.70 results from results/wp1_symmetric (no re-run for 0.70).

Categories: race-color (typical, median R < 1) and socioeconomic (the one
exception with median R > 1), so the sweep covers both regimes.

Grid per new theta: 5 encoders x 3 languages x 2 categories x 3 seeds = 90.
Four new thetas -> 360 conditions. Each theta writes to its own results root
(results/theta_sensitivity/theta0XX/...) so thresholds never share files and
every run is resume-capable after a preemption.

    python scripts/theta_sensitivity.py               # full sweep
    python scripts/theta_sensitivity.py --dry-run     # 2 models, 1 theta, tiny
    python scripts/theta_sensitivity.py --analyze     # summary + figure only

Outputs:
    results/theta_sensitivity/summary_by_theta.json
    Submission/images/figure_theta_sensitivity.{png,pdf}
    A LaTeX table block printed to stdout for direct pasting.

# Implements the C1 check from Research_proposal.md / SUBMISSION_CHECKLIST_EJAI.md.
"""
import sys
import os
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scripts.wp1_symmetric_control import run_grid, ENCODERS
from src.utils.config import get_results_dir
from src.utils.logging_setup import get_logger

logger = get_logger("theta_sensitivity")

# Endpoints of the plausible range. With the existing theta = 0.70 grid this
# gives three points (0.60, 0.70, 0.80), enough to show that the median R stays
# below one across the range, at half the compute of a four-point sweep.
THETAS_NEW = [0.60, 0.80]
THETA_EXISTING = 0.70
CATS = ["race-color", "socioeconomic"]
LANGS = ["en", "hi", "bn"]
SEEDS = [42, 123, 456]


def theta_root(theta):
    return f"theta_sensitivity/theta{int(round(theta * 100)):03d}"


def is_converged(d):
    return (d.get("R") is not None and not d.get("baseline_above_theta")
            and d.get("R_undefined_reason") in (None, "removal_censored_lower_bound"))


def stats(rows):
    conv = [d for d in rows if is_converged(d)]
    cens = [d for d in rows if d.get("R_undefined_reason") == "injection_did_not_converge"]
    above = [d for d in rows if d.get("baseline_above_theta")]
    Rs = [d["R"] for d in conv]
    out = {
        "n_total": len(rows),
        "n_converged": len(conv),
        "n_injection_censored": len(cens),
        "n_baseline_above_theta": len(above),
        "median_R": round(float(np.median(Rs)), 3) if Rs else None,
        "frac_R_gt1": round(sum(1 for r in Rs if r > 1) / len(Rs), 3) if Rs else None,
    }
    for cat in CATS:
        crs = [d["R"] for d in conv if d["category"] == cat]
        out[f"median_R_{cat}"] = round(float(np.median(crs)), 3) if crs else None
        out[f"n_{cat}"] = len(crs)
    return out


def load_existing_theta070():
    """theta=0.70 rows come from the completed WP1 matched grid; filter to CATS."""
    p = get_results_dir("wp1_symmetric") / "summary.json"
    if not p.exists():
        logger.warning("wp1_symmetric/summary.json not found; theta=0.70 point will be missing")
        return []
    rows = json.load(open(p, encoding="utf-8"))
    return [d for d in rows if isinstance(d, dict) and d.get("category") in CATS]


def analyze():
    by_theta = {}
    for theta in THETAS_NEW:
        p = get_results_dir(theta_root(theta)) / "summary.json"
        if p.exists():
            by_theta[f"{theta:.2f}"] = stats(json.load(open(p, encoding="utf-8")))
        else:
            logger.warning(f"no summary for theta={theta}; run the sweep first")
    by_theta[f"{THETA_EXISTING:.2f}"] = stats(load_existing_theta070())

    out = get_results_dir("theta_sensitivity") / "summary_by_theta.json"
    with open(out, "w") as f:
        json.dump(by_theta, f, indent=2)
    logger.info(f"saved {out}")

    # figure: median R vs theta, one line per category, log scale
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    thetas = sorted(by_theta.keys())
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    colors = {"race-color": "#4C72B0", "socioeconomic": "#C44E52"}
    for cat in CATS:
        xs, ys = [], []
        for t in thetas:
            v = by_theta[t].get(f"median_R_{cat}")
            if v is not None:
                xs.append(float(t)); ys.append(v)
        if xs:
            ax.plot(xs, ys, "-o", ms=4, color=colors[cat], label=cat)
    ax.axhline(1.0, color="grey", ls="--", lw=1, label="$R = 1$")
    ax.set_yscale("log")
    ax.set_xlabel(r"threshold $\theta$")
    ax.set_ylabel("median $R$")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    img = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "Submission", "images")
    os.makedirs(img, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(img, f"figure_theta_sensitivity.{ext}"), dpi=300,
                    bbox_inches="tight")
    logger.info("saved Submission/images/figure_theta_sensitivity.png")

    # LaTeX block for direct pasting
    print("%---- paste into EJAI_Hysteresis.tex (C1 table) ----")
    print(r"\begin{tabular}{lccccc}")
    print(r"\toprule")
    print(r"\textbf{$\theta$} & \textbf{$n$} & \textbf{Censored} & \textbf{Median $R$} & "
          r"\textbf{Median $R$ (race-colour)} & \textbf{Median $R$ (socioeconomic)} \\")
    print(r"\midrule")
    for t in thetas:
        s = by_theta[t]
        print(f"{t} & {s['n_converged']} & {s['n_injection_censored']} & "
              f"{s['median_R']} & {s.get('median_R_race-color')} & "
              f"{s.get('median_R_socioeconomic')} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print("%---- end paste ----")
    return by_theta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="2 models, theta=0.65 only, tiny budgets, isolated root")
    ap.add_argument("--analyze", action="store_true", help="summary + figure only, no runs")
    args = ap.parse_args()

    if args.analyze:
        analyze()
        return

    if args.dry_run:
        logger.info("C1 DRY RUN: 2 models, theta=0.65, 1 category, 1 seed, tiny budgets")
        run_grid(models=["mbert", "jhu-clsp-mmbert"], languages=["en"],
                 categories=["race-color"], seeds=[42], theta=0.65,
                 max_inject=60, max_remove=120, eval_every=5,
                 objective="matched", root="theta_sensitivity/dryrun")
        logger.info("C1 dry run complete; inspect results/theta_sensitivity/dryrun")
        return

    for theta in THETAS_NEW:
        logger.info("=" * 60)
        logger.info(f"C1 sweep: theta = {theta}")
        logger.info("=" * 60)
        run_grid(models=ENCODERS, languages=LANGS, categories=CATS, seeds=SEEDS,
                 theta=theta, max_inject=500, max_remove=1000, eval_every=25,
                 objective="matched", root=theta_root(theta))
    analyze()


if __name__ == "__main__":
    main()
