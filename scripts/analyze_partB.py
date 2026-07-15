"""
Part B breakdowns and figures for the measurement-validity paper.

Reads results/wp1_symmetric/summary.json (matched) and
results/wp1_mismatched/summary.json (mismatched). Prints per-category and
per-seed breakdowns (matched objective) and writes two figures into
Submission/images: a trajectory figure and a censoring-illustration figure.
Every number is read from the results files; nothing is invented.

    python scripts/analyze_partB.py
"""
import sys, os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "Submission", "images")


def load(root):
    return json.load(open(os.path.join(ROOT, "results", root, "summary.json"), encoding="utf-8"))


def converged(data):
    return [d for d in data if d.get("R") is not None and not d.get("baseline_above_theta")
            and d.get("R_undefined_reason") in (None, "removal_censored_lower_bound")]


def median(xs):
    return float(np.median(xs)) if xs else float("nan")


def breakdown(cells, key):
    groups = {}
    for c in cells:
        groups.setdefault(c[key], []).append(c["R"])
    rows = []
    for g, rs in sorted(groups.items()):
        rows.append((g, len(rs), round(median(rs), 3),
                     int(sum(1 for r in rs if r > 1)),
                     round(sum(1 for r in rs if r > 1) / len(rs), 3)))
    return rows


def main():
    matched = load("wp1_symmetric")
    mismatched = load("wp1_mismatched")
    cm = converged(matched)

    print("=== PER-CATEGORY (matched) ===")
    print("category, n, median_R, n_Rgt1, share_Rgt1")
    for r in breakdown(cm, "category"):
        print(r)

    print("=== PER-SEED (matched) ===")
    print("seed, n, median_R, n_Rgt1, share_Rgt1")
    for r in breakdown(cm, "seed"):
        print(r)

    # ---- Trajectory figure: representative injection vs removal ----
    # pick conditions with the most eval points on both phases, matched objective
    def n_pts(d):
        return len(d.get("inject_trajectory", [])) + len(d.get("remove_trajectory", []))
    # representative R<1 conditions (injection slow, removal fast), one per model, diverse
    cand = [d for d in cm if len(d.get("inject_trajectory", [])) >= 3
            and len(d.get("remove_trajectory", [])) >= 2 and d["R"] < 0.5]
    cand = sorted(cand, key=n_pts, reverse=True)
    picks, seen = [], set()
    for d in cand:
        if d["model"] in seen:
            continue
        seen.add(d["model"]); picks.append(d)
        if len(picks) == 2:
            break
    fig, axes = plt.subplots(1, len(picks), figsize=(9, 3.4), sharey=True)
    if len(picks) == 1:
        axes = [axes]
    for ax, d in zip(axes, picks):
        inj = d["inject_trajectory"]
        rem = d["remove_trajectory"]
        xi = [p[0] for p in inj]; yi = [p[1] for p in inj]
        # continue removal steps after injection on a shared axis
        off = xi[-1]
        xr = [off + p[0] for p in rem]; yr = [p[1] for p in rem]
        ax.plot(xi, yi, "-o", color="#C44E52", ms=3, label="injection")
        ax.plot(xr, yr, "-o", color="#4C72B0", ms=3, label="removal")
        ax.axhline(d["theta"], color="grey", ls="--", lw=1)
        ax.set_title(f"{d['model']} / {d['language']} / {d['category']}", fontsize=8)
        ax.set_xlabel("gradient step")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("bias score $B$")
    axes[0].legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(IMG, f"figure_trajectory.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- Censoring figure: how ceiling mass inflates the mean (mismatched) ----
    mm_def = [d["R"] for d in converged(mismatched)]
    n_cens = sum(1 for d in mismatched if d.get("R_undefined_reason") == "injection_did_not_converge")
    ceiling = 1000.0 / 500.0 * 2.0  # original protocol ceiling analogue (T_debias^max/T_bias^max)=2000/500=4.0
    # original mean would include censored at ceiling 4.0
    orig_vals = mm_def + [4.0] * n_cens
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.hist(np.log10(mm_def), bins=25, color="#4C72B0", alpha=0.7, label=f"defined $R$ (n={len(mm_def)})")
    ax.axvline(np.log10(4.0), color="#C44E52", lw=2, label=f"censored mass at ceiling $R=4$ (n={n_cens})")
    ax.axvline(np.log10(np.median(mm_def)), color="black", ls="--", lw=1.5,
               label=f"median (defined) = {np.median(mm_def):.3f}")
    ax.axvline(np.log10(np.mean(orig_vals)), color="darkorange", ls=":", lw=2,
               label=f"mean incl. ceiling = {np.mean(orig_vals):.2f}")
    ax.axvline(0.0, color="grey", lw=1)
    ax.set_xlabel("asymmetry ratio $R$ (log$_{10}$ scale)")
    ax.set_ylabel("conditions")
    ax.set_xticks([-3, -2, -1, 0, 0.6])
    ax.set_xticklabels(["$10^{-3}$", "$10^{-2}$", "$10^{-1}$", "$1$", "$4$"])
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(IMG, f"figure_censoring.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("SAVED figure_trajectory and figure_censoring; picks:",
          [(d["model"], d["language"], d["category"]) for d in picks])
    print("CENSORING FIG: n_defined=%d n_censored=%d median_defined=%.3f mean_incl_ceiling=%.2f"
          % (len(mm_def), n_cens, np.median(mm_def), np.mean(orig_vals)))


if __name__ == "__main__":
    main()
