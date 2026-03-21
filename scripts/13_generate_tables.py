"""
Script 13: Generate Paper Tables (LaTeX format).

# ============================================================
# Table 1: Baseline Bias Scores (Phase 0)
# Table 2: Asymmetry Ratio R Summary
# Table 3: Category-level R Ranking
# Table 4: Statistical Tests
# Table 5: Comparative R (generated in script 11)
# ============================================================

Usage: python scripts/13_generate_tables.py
"""

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import get_results_dir, get_all_model_configs
from src.utils.logging_setup import get_logger

logger = get_logger("13_generate_tables")


def table1_baseline():
    """Table 1: Baseline Bias Scores."""
    logger.info("Generating Table 1: Baseline Bias Scores")

    path = get_results_dir("phase0_baseline") / "baseline_results.json"
    if not path.exists():
        logger.warning("  Baseline results not found. Skipping.")
        return

    with open(path) as f:
        data = json.load(f)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Baseline Bias Scores (Phase 0). Score > 0.5 indicates stereotypical preference.}",
        r"\label{tab:baseline}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Model & English & Hindi & Bengali \\",
        r"\midrule",
    ]

    for model_name, langs in data.items():
        scores = []
        for lang in ["en", "hi", "bn"]:
            s = langs.get(lang, {}).get("overall_bias_score", "—")
            scores.append(f"{s:.3f}" if isinstance(s, float) else str(s))
        lines.append(f"  {model_name} & {' & '.join(scores)} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    out_path = get_results_dir("tables") / "table1_baseline.tex"
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"  Saved: {out_path}")


def table2_R_summary():
    """Table 2: Asymmetry Ratio R Summary."""
    logger.info("Generating Table 2: R Summary")

    path = get_results_dir("phase3_asymmetry") / "full_results.json"
    if not path.exists():
        logger.warning("  Phase 3 results not found. Skipping.")
        return

    with open(path) as f:
        data = json.load(f)

    R_tensor = data["R_tensor"]
    theta = "0.7"

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Asymmetry Ratio $R$ Summary. All values computed at $\theta = 0.7$.}",
        r"\label{tab:R_summary}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Model & Grand $R$ & $R_{en}$ & $R_{hi}$ & $R_{bn}$ \\",
        r"\midrule",
    ]

    for model_name in R_tensor:
        lang_Rs = {}
        for lang in ["en", "hi", "bn"]:
            all_r = []
            if lang in R_tensor[model_name]:
                for cat in R_tensor[model_name][lang]:
                    if theta in R_tensor[model_name][lang][cat]:
                        r = R_tensor[model_name][lang][cat][theta].get("R_mean", 0)
                        if r != float("inf"):
                            all_r.append(r)
            lang_Rs[lang] = float(np.mean(all_r)) if all_r else 0.0

        grand_R = np.mean(list(lang_Rs.values()))
        lines.append(
            f"  {model_name} & ${grand_R:.2f}$ & "
            f"${lang_Rs['en']:.2f}$ & ${lang_Rs['hi']:.2f}$ & ${lang_Rs['bn']:.2f}$ \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    out_path = get_results_dir("tables") / "table2_R_summary.tex"
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"  Saved: {out_path}")


def table3_category_ranking():
    """Table 3: Category-level R Ranking."""
    logger.info("Generating Table 3: Category R Ranking")

    path = get_results_dir("phase6_cultural") / "cultural_analysis.json"
    if not path.exists():
        logger.warning("  Cultural analysis not found. Skipping.")
        return

    with open(path) as f:
        data = json.load(f)

    ranking = data.get("category_ranking", [])

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Category-level Asymmetry Ratio $R$, sorted by mean $R$.}",
        r"\label{tab:category_R}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Bias Category & Mean $R$ & Median $R$ & 95\% CI \\",
        r"\midrule",
    ]

    for cat, stats in ranking:
        ci = stats.get("CI_95", (0, 0))
        lines.append(
            f"  {cat} & ${stats['mean']:.2f}$ & ${stats['median']:.2f}$ & "
            f"[{ci[0]:.2f}, {ci[1]:.2f}] \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    out_path = get_results_dir("tables") / "table3_category_R.tex"
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"  Saved: {out_path}")


def table4_statistics():
    """Table 4: Statistical Test Results."""
    logger.info("Generating Table 4: Statistical Tests")

    path = get_results_dir("phase3_asymmetry") / "full_results.json"
    if not path.exists():
        return

    with open(path) as f:
        data = json.load(f)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Statistical test results for the Bias Hysteresis Principle.}",
        r"\label{tab:statistics}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Test & Statistic & $p$-value \\",
        r"\midrule",
        f"  Wilcoxon (R > 1) & — & ${data.get('wilcoxon_p', 'N/A')}$ \\\\",
        f"  Grand Mean R & ${data.get('grand_mean_R', 0):.3f}$ & — \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    out_path = get_results_dir("tables") / "table4_statistics.tex"
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"  Saved: {out_path}")


def main():
    logger.info("=" * 60)
    logger.info("GENERATING PAPER TABLES (LaTeX)")
    logger.info("=" * 60)

    table1_baseline()
    table2_R_summary()
    table3_category_ranking()
    table4_statistics()

    logger.info("\nAll tables generated!")
    logger.info("Pipeline complete! 🎉")


if __name__ == "__main__":
    main()
