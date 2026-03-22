"""
Script 11: Compute Comparative R Ratios.

Produces Table 5: Method-Independence of the Bias Hysteresis Principle.
Automatically discovers which models were run in Phase 5C.
CPU only, ~5 minutes.

# ============================================================
# PAPER CITATIONS
# [11]-[16] See comparative method files for full citations
# ============================================================

Usage: python scripts/11_comparative_asymmetry.py
"""

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import get_results_dir, get_all_model_configs, load_training_config
from src.utils.logging_setup import get_logger
from src.utils.seed import get_seeds
from src.training.checkpoint_manager import load_results
from src.analysis.asymmetry_ratio import find_first_crossing
from src.analysis.statistical_tests import bootstrap_ci

logger = get_logger("11_comparative_asymmetry")

METHODS = {
    "Phase2_Contrastive": "phase2_removal",
    "C1_CDA": "phase5c_comparatives/c1_cda",
    "C2_SelfDebias": "phase5c_comparatives/c2_self_debias",
    "C3_INLP": "phase5c_comparatives/c3_inlp",
    "C4_DAMA": "phase5c_comparatives/c4_dama",
    "C5_BiasEdit": "phase5c_comparatives/c5_biasedit",
    "C6_GradientAscent": "phase5c_comparatives/c6_gradient_ascent",
}


def discover_comparative_models() -> list:
    """
    Auto-discover which models have comparative results.

    Scans results/phase5c_comparatives/ for model-named subdirectories
    across all method folders.
    """
    all_configs = get_all_model_configs()
    all_model_names = list(all_configs.keys())
    found_models = set()

    for method_name, phase_dir in METHODS.items():
        if method_name == "Phase2_Contrastive":
            continue
        base = get_results_dir(phase_dir)
        if base.exists():
            for model_name in all_model_names:
                model_dir = base / model_name
                if model_dir.exists():
                    found_models.add(model_name)

    # Fallback: if nothing found, use all 6 models
    if not found_models:
        found_models = set(all_model_names)

    return sorted(found_models)


def main():
    logger.info("=" * 60)
    logger.info("COMPARATIVE ASYMMETRY RATIOS")
    logger.info("=" * 60)

    training_config = load_training_config()
    theta = training_config["bias_threshold_theta"]
    seeds = get_seeds()
    language = "en"
    max_inj = training_config["injection"]["max_steps"]
    max_rem = training_config["removal"]["max_steps"]

    # Auto-discover models that have comparative results
    MODELS = discover_comparative_models()
    all_configs = get_all_model_configs()

    logger.info(f"\nDiscovered {len(MODELS)} models with comparative results:")
    for m in MODELS:
        model_type = all_configs.get(m, {}).get("model_type", "unknown")
        logger.info(f"  • {m} ({model_type})")

    table_data = []

    for model_name in MODELS:
        model_type = all_configs.get(model_name, {}).get("model_type", "unknown")
        logger.info(f"\n--- {model_name} ({model_type}) ---")

        for method_name, phase_dir in METHODS.items():
            # Skip causal-only methods for encoder models
            if method_name == "C2_SelfDebias" and model_type == "encoder":
                continue
            if method_name == "C4_DAMA" and model_type == "encoder":
                continue

            Rs = []
            for seed in seeds:
                # T_bias: from Phase 1
                inj_curve = load_results("phase1_injection", model_name, language, seed)
                T_bias = find_first_crossing(inj_curve, "_overall", theta, "above")
                if T_bias is None:
                    T_bias = max_inj

                # T_debias: from respective method
                if method_name == "C2_SelfDebias":
                    deb_results = load_results(phase_dir, model_name, language, seed)
                    T_debias = _find_alpha_crossing(deb_results, theta)
                elif method_name in ["C3_INLP", "C4_DAMA"]:
                    deb_results_path = (
                        get_results_dir(f"{phase_dir}/{model_name}")
                        / f"seed{seed}" / "results.json"
                    )
                    if deb_results_path.exists():
                        with open(deb_results_path) as f:
                            deb = json.load(f)
                        T_debias = deb.get("n_dimensions_removed", max_rem)
                    else:
                        deb_results = load_results(phase_dir, model_name, language, seed)
                        T_debias = _find_step_crossing(deb_results, theta)
                else:
                    deb_curve = load_results(phase_dir, model_name, language, seed)
                    T_debias = find_first_crossing(deb_curve, "_overall", theta, "below")
                    if T_debias is None:
                        T_debias = max_rem

                R = T_debias / T_bias if T_bias > 0 else float("inf")
                Rs.append(R)

            finite_Rs = [r for r in Rs if r != float("inf")]
            R_mean = float(np.mean(finite_Rs)) if finite_Rs else float("inf")
            R_std = float(np.std(finite_Rs)) if len(finite_Rs) > 1 else 0.0
            CI = bootstrap_ci(finite_Rs) if len(finite_Rs) > 2 else (R_mean, R_mean)

            table_data.append({
                "model": model_name,
                "model_type": model_type,
                "method": method_name,
                "R_mean": R_mean,
                "R_std": R_std,
                "CI_95": CI,
                "R_seeds": Rs,
            })

            logger.info(f"  {method_name:25s}: R = {R_mean:.3f} ± {R_std:.3f}")

    # Generate LaTeX table
    _generate_latex_table(table_data)

    # Save JSON
    out_path = get_results_dir("phase5c_comparatives") / "comparative_R.json"
    with open(out_path, "w") as f:
        json.dump(table_data, f, indent=2, default=str)

    logger.info(f"\nComparative R results saved to {out_path}")
    logger.info("Next: python scripts/12_generate_figures.py")


def _find_alpha_crossing(results, theta):
    """For Self-Debias: find first alpha where bias drops below theta."""
    for r in results:
        if r.get("overall_bias_score", 1.0) <= theta:
            return int(r.get("alpha", 0) * 100)
    return 200


def _find_step_crossing(results, theta):
    """Find first step or iteration where bias drops below theta."""
    for r in results:
        if r.get("overall_bias_score", r.get("bias_score_approx", 1.0)) <= theta:
            return r.get("step", r.get("iteration", 1))
    return 2000


def _generate_latex_table(data):
    """Generate LaTeX Table 5 — now with all models."""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Method-Independence of the Bias Hysteresis Principle across all models.}",
        r"\label{tab:comparative_R}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Model & Type & Method & $R$ & 95\% CI & Paper \\",
        r"\midrule",
    ]

    papers = {
        "Phase2_Contrastive": "Ours",
        "C1_CDA": "[11]",
        "C2_SelfDebias": "[12]",
        "C3_INLP": "[13]",
        "C4_DAMA": "[14]",
        "C5_BiasEdit": "[15]",
        "C6_GradientAscent": "[16]",
    }

    current_model = None
    for row in data:
        ci = row["CI_95"]
        paper = papers.get(row["method"], "")

        # Add midrule between models
        model_display = row["model"]
        if current_model is not None and current_model != row["model"]:
            lines.append(r"\midrule")
        current_model = row["model"]

        lines.append(
            f"  {model_display} & {row.get('model_type', '')} & {row['method']} & "
            f"${row['R_mean']:.2f} \\pm {row['R_std']:.2f}$ & "
            f"[{ci[0]:.2f}, {ci[1]:.2f}] & {paper} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])

    tex_path = get_results_dir("tables") / "table5_comparative_R.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))

    logger.info(f"  LaTeX table saved to {tex_path}")


if __name__ == "__main__":
    main()
