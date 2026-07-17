"""
C2 analysis — loop areas, permutation test, representative loop figure.

Reads results/loop_area/summary.json (written by scripts/loop_area.py).
Reports the median signed area A, a sign-flip permutation test of the null
"no hysteresis" (up and down curves exchangeable, so the sign of each
condition's area is random), and per-model / per-category medians.

The permutation test: under the null, each condition's signed area is
symmetric around zero, so flipping signs at random gives the null
distribution of the mean. The p-value is the share of 10,000 sign-flip
draws whose mean is at least as extreme as the observed mean.

    python scripts/analyze_loop_area.py

Outputs:
    results/loop_area/loop_area_analysis.json
    Submission/images/figure_loop_area.{png,pdf}
    A LaTeX table block printed to stdout for direct pasting.
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.utils.config import get_results_dir
from src.utils.logging_setup import get_logger

logger = get_logger("analyze_loop_area")


def sign_flip_p(areas, n_draws=10000, seed=0):
    """Two-sided and one-sided (A > 0) sign-flip permutation p-values."""
    rng = np.random.default_rng(seed)
    a = np.asarray(areas, dtype=float)
    obs = a.mean()
    flips = rng.choice([-1.0, 1.0], size=(n_draws, len(a)))
    null_means = (flips * np.abs(a)).mean(axis=1)
    p_two = float((np.abs(null_means) >= abs(obs)).mean())
    p_one = float((null_means >= obs).mean()) if obs > 0 else float((null_means <= obs).mean())
    return obs, p_two, p_one


def group_medians(rows, key):
    groups = {}
    for r in rows:
        groups.setdefault(r[key], []).append(r["signed_area"])
    return {g: {"n": len(v), "median_A": round(float(np.median(v)), 4)}
            for g, v in sorted(groups.items())}


def main():
    p = get_results_dir("loop_area") / "summary.json"
    if not p.exists():
        logger.error("results/loop_area/summary.json not found; run scripts/loop_area.py first")
        sys.exit(1)
    rows = [r for r in json.load(open(p, encoding="utf-8"))
            if isinstance(r, dict) and r.get("status") != "FAIL" and "signed_area" in r]
    if not rows:
        logger.error("no successful loop conditions in summary")
        sys.exit(1)

    areas = [r["signed_area"] for r in rows]
    obs_mean, p_two, p_one = sign_flip_p(areas)

    analysis = {
        "n_conditions": len(rows),
        "median_signed_A": round(float(np.median(areas)), 4),
        "mean_signed_A": round(float(obs_mean), 4),
        "median_abs_A": round(float(np.median(np.abs(areas))), 4),
        "share_A_positive": round(float(np.mean(np.asarray(areas) > 0)), 3),
        "signflip_p_two_sided": p_two,
        "signflip_p_one_sided": p_one,
        "per_model": group_medians(rows, "model"),
        "per_category": group_medians(rows, "category"),
        "per_language": group_medians(rows, "language"),
    }
    out = get_results_dir("loop_area") / "loop_area_analysis.json"
    with open(out, "w") as f:
        json.dump(analysis, f, indent=2)
    logger.info(f"saved {out}")

    # figure: the largest-|A| loop and a near-zero loop, side by side
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    catname = {"race-color": "race-colour", "socioeconomic": "socioeconomic",
               "religion": "religion", "nationality": "nationality"}
    by_abs = sorted(rows, key=lambda r: abs(r["signed_area"]))
    picks = [by_abs[-1], by_abs[0]]
    titles = ["largest $|A|$", "smallest $|A|$"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, r, tt in zip(axes, picks, titles):
        up = r["up_curve"]; down = r["down_curve"]
        ax.plot([q[0] for q in up], [q[1] for q in up], "-o", ms=3,
                color="#C44E52", label="up-sweep")
        ax.plot([q[0] for q in down], [q[1] for q in down], "-o", ms=3,
                color="#4C72B0", label="down-sweep")
        cat = catname.get(r["category"], r["category"])
        ax.set_title(f"{r['model']} / {r['language']} / {cat} "
                     f"($A={r['signed_area']:+.3f}$, {tt})", fontsize=8)
        ax.set_xlabel(r"field $\lambda$")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("bias score $B$")
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    img = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "Submission", "images")
    os.makedirs(img, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(img, f"figure_loop_area.{ext}"), dpi=300,
                    bbox_inches="tight")
    logger.info("saved Submission/images/figure_loop_area.png")

    print("LOOP_AREA_JSON_START")
    print(json.dumps(analysis, indent=2))
    print("LOOP_AREA_JSON_END")

    print("%---- paste into EJAI_Hysteresis.tex (C2 table) ----")
    print(r"\begin{tabular}{lcc}")
    print(r"\toprule")
    print(r"\textbf{Quantity} & \textbf{Value} \\")
    print(r"\midrule")
    print(f"Conditions & {analysis['n_conditions']} \\\\")
    print(f"Median signed area $A$ & {analysis['median_signed_A']} \\\\")
    print(f"Median $|A|$ & {analysis['median_abs_A']} \\\\")
    print(f"Share $A > 0$ & {analysis['share_A_positive']} \\\\")
    print(f"Sign-flip $p$ (two-sided) & {analysis['signflip_p_two_sided']} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print("%---- end paste ----")


if __name__ == "__main__":
    main()
