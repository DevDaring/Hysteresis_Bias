"""
Model x category heatmap of the matched-objective median ratio R.

Reads results/wp1_symmetric/summary.json and, for each (model, category) cell,
takes the median R over converged conditions (three languages x three seeds).
Colours are on a log scale centred at R = 1, so cells above one (removal looks
harder) and below one (removal faster) are visually distinct. This shows the
joint structure that the per-model and per-category tables only give as
marginals, and makes the socioeconomic exception stand out.

    python scripts/generate_heatmap.py
Output: Submission/images/figure_heatmap.{png,pdf}  (single column)
"""
import os
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "Submission", "images")

MODELS = [("mbert", "mBERT"), ("xlm-roberta", "XLM-RoBERTa"), ("muril", "MuRIL"),
          ("indicbert-v2", "IndicBERTv2"), ("jhu-clsp-mmbert", "mmBERT")]
CATS = [("nationality", "Nationality"), ("race-color", "Race-colour"),
        ("religion", "Religion"), ("socioeconomic", "Socioeconomic")]


def converged(d):
    return (d.get("R") is not None and not d.get("baseline_above_theta")
            and d.get("R_undefined_reason") in (None, "removal_censored_lower_bound"))


def main():
    rows = json.load(open(os.path.join(ROOT, "results", "wp1_symmetric", "summary.json"),
                          encoding="utf-8"))
    M, C = len(MODELS), len(CATS)
    grid = np.full((M, C), np.nan)
    for i, (mkey, _) in enumerate(MODELS):
        for j, (ckey, _) in enumerate(CATS):
            vals = [d["R"] for d in rows if d["model"] == mkey and d["category"] == ckey
                    and converged(d)]
            if vals:
                grid[i, j] = np.median(vals)

    loggrid = np.log10(grid)
    norm = TwoSlopeNorm(vmin=-1.6, vcenter=0.0, vmax=1.6)  # R from ~0.025 to ~40, centre 1
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    im = ax.imshow(loggrid, cmap="RdBu_r", norm=norm, aspect="auto")

    ax.set_xticks(range(C)); ax.set_xticklabels([c[1] for c in CATS], rotation=30, ha="right")
    ax.set_yticks(range(M)); ax.set_yticklabels([m[1] for m in MODELS])
    for i in range(M):
        for j in range(C):
            if not np.isnan(grid[i, j]):
                v = grid[i, j]
                # white text on dark cells, black on light
                txtcol = "white" if abs(np.log10(v)) > 0.8 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color=txtcol, fontsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                        ticks=[np.log10(x) for x in (0.03, 0.1, 0.3, 1, 3, 10, 30)])
    cbar.ax.set_yticklabels(["0.03", "0.1", "0.3", "1", "3", "10", "30"])
    cbar.set_label("median $R$ (log scale)")
    ax.set_title("Matched-objective median $R$ by model and category")
    fig.tight_layout()
    os.makedirs(IMG, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(IMG, f"figure_heatmap.{ext}"), dpi=300, bbox_inches="tight")
    print("SAVED figure_heatmap; grid (median R):")
    for i, (_, ml) in enumerate(MODELS):
        print(" ", ml, [round(grid[i, j], 2) if not np.isnan(grid[i, j]) else None
                        for j in range(C)])


if __name__ == "__main__":
    main()
