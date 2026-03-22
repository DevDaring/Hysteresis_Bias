"""
Script 12: Generate Paper Figures (Publication Quality).

Uses matplotlib with Nature-style settings.

# ============================================================
# PAPER CITATIONS
# All figures reference data from Phases 0-6.
# ============================================================

Usage: python scripts/12_generate_figures.py
"""

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.config import get_results_dir, get_all_model_configs
from src.utils.logging_setup import get_logger

logger = get_logger("12_generate_figures")

# Nature-style settings
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica"],
    "font.size": 8,
    "axes.linewidth": 0.8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Colorblind-safe palette
COLORS = {
    "injection": "#D32F2F",  # Red
    "removal": "#1976D2",    # Blue
    "neutral": "#757575",    # Gray
}


def figure1_hysteresis_curves():
    """Figure 1: Bias Hysteresis Curves — THE SIGNATURE FIGURE."""
    logger.info("Generating Figure 1: Hysteresis Curves")

    all_configs = get_all_model_configs()
    n_models = len(all_configs)
    ncols = min(5, n_models)
    nrows = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(183/25.4, 60*nrows/25.4), sharex=False, sharey=True)
    axes = axes.flatten() if n_models > 1 else [axes]

    for idx, (model_name, cfg) in enumerate(all_configs.items()):
        ax = axes[idx]
        ax.set_title(model_name, fontsize=8, fontweight="bold")

        # Load curves for seed=42, language=en
        try:
            inj_path = (
                get_results_dir("phase1_injection")
                / model_name / "en" / "seed42" / "curves.json"
            )
            rem_path = (
                get_results_dir("phase2_removal")
                / model_name / "en" / "seed42" / "curves.json"
            )

            if inj_path.exists() and rem_path.exists():
                with open(inj_path) as f:
                    inj_data = json.load(f)
                with open(rem_path) as f:
                    rem_data = json.load(f)

                inj_steps = [d["step"] for d in inj_data]
                inj_scores = [d["overall_bias_score"] for d in inj_data]
                rem_steps = [d["step"] for d in rem_data]
                rem_scores = [d["overall_bias_score"] for d in rem_data]

                ax.plot(inj_steps, inj_scores, color=COLORS["injection"],
                        label="Injection", linewidth=1.2)
                ax.plot(rem_steps, rem_scores, color=COLORS["removal"],
                        label="Removal", linewidth=1.2)

                # Threshold line
                ax.axhline(y=0.7, color=COLORS["neutral"], linestyle="--",
                           linewidth=0.5, alpha=0.7)
            else:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=7, color="gray")

        except Exception as e:
            ax.text(0.5, 0.5, f"Error", transform=ax.transAxes,
                    ha="center", fontsize=6, color="red")

        if idx >= 3:
            ax.set_xlabel("Gradient Steps")
        if idx % 3 == 0:
            ax.set_ylabel("Bias Score")
        ax.set_ylim(0.4, 1.0)

    axes[0].legend(loc="lower right", framealpha=0.9)
    # Hide extra subplots
    for i in range(n_models, len(axes)):
        axes[i].set_visible(False)
    plt.suptitle("Bias Hysteresis Curves", fontsize=10, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_path = get_results_dir("figures") / "figure1_hysteresis_curves.pdf"
    plt.savefig(out_path)
    plt.savefig(out_path.with_suffix(".png"))
    plt.close()
    logger.info(f"  Saved: {out_path}")


def figure2_R_heatmap():
    """Figure 2: Asymmetry Ratio R Heatmap."""
    logger.info("Generating Figure 2: R Heatmap")

    results_path = get_results_dir("phase3_asymmetry") / "full_results.json"
    if not results_path.exists():
        logger.warning("  Phase 3 results not found. Skipping.")
        return

    with open(results_path) as f:
        data = json.load(f)

    R_tensor = data["R_tensor"]
    languages = ["en", "hi", "bn"]

    fig, axes = plt.subplots(1, 3, figsize=(183/25.4, 100/25.4))

    for lang_idx, language in enumerate(languages):
        ax = axes[lang_idx]
        ax.set_title(f"Language: {language.upper()}", fontsize=8, fontweight="bold")

        models = list(R_tensor.keys())
        categories = set()
        for m in models:
            if language in R_tensor[m]:
                categories.update(R_tensor[m][language].keys())
        categories = sorted(categories - {"_overall"})

        matrix = np.zeros((len(categories), len(models)))
        for j, model in enumerate(models):
            for i, cat in enumerate(categories):
                try:
                    val = R_tensor[model][language][cat]["0.7"]["R_mean"]
                    matrix[i, j] = min(val, 5.0)  # Cap for visualization
                except (KeyError, TypeError):
                    matrix[i, j] = np.nan

        sns.heatmap(
            matrix, ax=ax, cmap="RdBu_r", center=1.0,
            xticklabels=models, yticklabels=categories if lang_idx == 0 else False,
            annot=True, fmt=".1f", annot_kws={"size": 5},
            cbar=lang_idx == 2,
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    plt.suptitle("Asymmetry Ratio R by Model × Category × Language", fontsize=10, fontweight="bold")
    plt.tight_layout()

    out_path = get_results_dir("figures") / "figure2_R_heatmap.pdf"
    plt.savefig(out_path)
    plt.savefig(out_path.with_suffix(".png"))
    plt.close()
    logger.info(f"  Saved: {out_path}")


def figure3_cultural():
    """Figure 3: Cultural Dependence of R."""
    logger.info("Generating Figure 3: Cultural Dependence")

    results_path = get_results_dir("phase6_cultural") / "cultural_analysis.json"
    if not results_path.exists():
        logger.warning("  Cultural analysis results not found. Skipping.")
        return

    with open(results_path) as f:
        data = json.load(f)

    ranking = data.get("category_ranking", [])
    if not ranking:
        return

    categories = [r[0] for r in ranking]
    means = [r[1]["mean"] for r in ranking]

    fig, ax = plt.subplots(figsize=(89/25.4, 80/25.4))
    bars = ax.barh(range(len(categories)), means, color="#1976D2", edgecolor="white")

    # Highlight caste
    for i, cat in enumerate(categories):
        if cat == "caste":
            bars[i].set_color("#D32F2F")

    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories)
    ax.set_xlabel("Mean Asymmetry Ratio R")
    ax.axvline(x=1.0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_title("Cultural Dependence of R", fontsize=9, fontweight="bold")
    ax.invert_yaxis()
    plt.tight_layout()

    out_path = get_results_dir("figures") / "figure3_cultural.pdf"
    plt.savefig(out_path)
    plt.savefig(out_path.with_suffix(".png"))
    plt.close()
    logger.info(f"  Saved: {out_path}")


def main():
    logger.info("=" * 60)
    logger.info("GENERATING PAPER FIGURES")
    logger.info("=" * 60)

    figure1_hysteresis_curves()
    figure2_R_heatmap()
    figure3_cultural()

    logger.info("\nAll figures generated!")
    logger.info("Next: python scripts/13_generate_tables.py")


if __name__ == "__main__":
    main()
