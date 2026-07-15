"""
Figure for the measurement-validity paper.

Reads results/wp1_symmetric/summary.json (matched objective) and, if present,
results/wp1_mismatched/summary.json (mismatched objective). Produces a
distribution plot of the asymmetry ratio R on a log axis, with the symmetry
line at R=1, showing that the matched objective concentrates R below 1.

    python scripts/generate_wp1_figure.py
Output: images/figure_R_distribution.png (+ .pdf)
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.utils.config import get_results_dir, get_project_root


def load_R(root):
    p = get_results_dir(root) / "summary.json"
    if not p.exists():
        return None
    data = json.load(open(p))
    return [d["R"] for d in data if d.get("R") is not None
            and not d.get("baseline_above_theta")
            and d.get("R_undefined_reason") in (None, "removal_censored_lower_bound")
            and d["R"] > 0]


def main():
    matched = load_R("wp1_symmetric")
    mismatched = load_R("wp1_mismatched")

    fig, ax = plt.subplots(figsize=(7, 3.4))
    series = []
    labels = []
    if mismatched:
        series.append(np.log10(mismatched)); labels.append("Mismatched objective\n(CE inject / squared-gap remove)")
    series.append(np.log10(matched)); labels.append("Matched objective\n(signed gap, both directions)")

    parts = ax.violinplot(series, vert=False, showmedians=True, widths=0.8)
    for pc in parts["bodies"]:
        pc.set_facecolor("#4C72B0"); pc.set_alpha(0.55); pc.set_edgecolor("#2a2a2a")
    for key in ("cmedians", "cbars", "cmins", "cmaxes"):
        if key in parts:
            parts[key].set_color("#2a2a2a")

    ax.axvline(0.0, color="#C44E52", lw=1.6, ls="--", label="$R = 1$ (symmetry)")
    ax.set_yticks(range(1, len(labels) + 1))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Asymmetry ratio $R = T_{\\mathrm{debias}}/T_{\\mathrm{bias}}$ (log$_{10}$ scale)")
    ticks = [-2, -1, 0, 1, 2]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"$10^{{{t}}}$" for t in ticks])
    med = np.median(matched)
    ax.set_title(f"Matched-objective median $R = {med:.3f}$ ($n={len(matched)}$); "
                 f"{100*np.mean(np.array(matched) > 1):.0f}\\% of conditions $R>1$", fontsize=10)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()

    out_dir = get_project_root() / "images"
    out_dir.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"figure_R_distribution.{ext}", dpi=200, bbox_inches="tight")
    print(f"SAVED images/figure_R_distribution.png "
          f"(matched n={len(matched)}, mismatched n={len(mismatched) if mismatched else 0})")


if __name__ == "__main__":
    main()
