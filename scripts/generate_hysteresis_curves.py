"""
Injection/removal curve figure for the five encoders, matched objective.

Reads results/wp1_symmetric/summary.json (the matched-objective grid, which
stores per-step inject_trajectory and remove_trajectory for each condition).
For each encoder it picks one representative converged condition and plots the
injection curve (rising to the threshold) followed by the removal curve
(falling back below it), on a shared step axis. This is the same kind of plot
as the old ten-model figure, but restricted to the models this paper studies
and to the matched objective. Every point is read from the results file.

    python scripts/generate_hysteresis_curves.py
Output: Submission/images/figure_hysteresis_curves.{png,pdf}  (wide, two-column)
"""
import os
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "Submission", "images")

# display order and labels matching the paper
ENCODERS = [("mbert", "mBERT"), ("xlm-roberta", "XLM-RoBERTa"), ("muril", "MuRIL"),
            ("indicbert-v2", "IndicBERTv2"), ("jhu-clsp-mmbert", "mmBERT")]


def converged(d):
    return (d.get("R") is not None and not d.get("baseline_above_theta")
            and d.get("R_undefined_reason") in (None, "removal_censored_lower_bound"))


def pick(rows, model):
    """Representative converged condition for a model: most trajectory points,
    with both an injection crossing and a removal crossing."""
    cand = [d for d in rows if d["model"] == model and converged(d)
            and len(d.get("inject_trajectory", [])) >= 3
            and len(d.get("remove_trajectory", [])) >= 3]
    if not cand:
        cand = [d for d in rows if d["model"] == model and converged(d)]
    if not cand:
        return None
    return max(cand, key=lambda d: len(d["inject_trajectory"]) + len(d["remove_trajectory"]))


def main():
    rows = json.load(open(os.path.join(ROOT, "results", "wp1_symmetric", "summary.json"),
                          encoding="utf-8"))
    fig, axes = plt.subplots(1, len(ENCODERS), figsize=(15, 3.1), sharey=True)
    theta = 0.7
    for ax, (key, label) in zip(axes, ENCODERS):
        d = pick(rows, key)
        if d is None:
            ax.set_title(label, fontsize=11)
            continue
        inj = d["inject_trajectory"]
        rem = d["remove_trajectory"]
        xi = [p[0] for p in inj]
        yi = [p[1] for p in inj]
        off = xi[-1]
        xr = [off + p[0] for p in rem]
        yr = [p[1] for p in rem]
        ax.plot(xi, yi, "-", color="#C0392B", lw=2, label="Injection")
        ax.plot(xr, yr, "-", color="#2471A3", lw=2, label="Removal")
        ax.axhline(theta, color="grey", ls="--", lw=1)
        ax.axvline(off, color="black", ls=":", lw=0.8)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("gradient step")
        ax.set_ylim(0.35, 1.0)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("bias score $B$")
    axes[0].legend(fontsize=9, frameon=False, loc="upper right")
    fig.tight_layout()
    os.makedirs(IMG, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(IMG, f"figure_hysteresis_curves.{ext}"), dpi=300,
                    bbox_inches="tight")
    picks = [(key, (pick(rows, key) or {}).get("language"), (pick(rows, key) or {}).get("category"))
             for key, _ in ENCODERS]
    print("SAVED figure_hysteresis_curves; picks:", picks)


if __name__ == "__main__":
    main()
