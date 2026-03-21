"""
Checkpoint manager — save/load LoRA checkpoints and training results.

Provides crash recovery by saving results incrementally.

# ============================================================
# PAPER CITATIONS
# [5] Hu et al. (2022). LoRA. ICLR 2022.
# ============================================================
"""

import json
from pathlib import Path
from typing import Dict, List, Any

from src.utils.config import get_results_dir
from src.utils.logging_setup import get_logger

try:
    from filelock import FileLock
    HAS_FILELOCK = True
except ImportError:
    HAS_FILELOCK = False

logger = get_logger(__name__)



def save_checkpoint(
    model,
    results: List[Dict],
    phase: str,
    model_name: str,
    language: str,
    seed: int,
    step: int = None,
    suffix: str = None,
):
    """
    Save LoRA checkpoint and results JSON.

    Args:
        model: PeftModel with LoRA adapters.
        results: List of checkpoint result dicts.
        phase: Phase name (e.g., 'phase1_injection', 'phase2_removal').
        model_name: Model key (e.g., 'llama-3.1-8b').
        language: Language code ('en', 'hi', 'bn').
        seed: Random seed.
        step: Training step (optional, for intermediate checkpoints).
        suffix: Optional suffix (e.g., 'final_biased', 'final_debiased').
    """
    base_dir = get_results_dir(phase) / model_name / language / f"seed{seed}"

    # Save LoRA weights
    if step is not None:
        checkpoint_dir = base_dir / f"step{step}"
    elif suffix is not None:
        checkpoint_dir = base_dir / suffix
    else:
        checkpoint_dir = base_dir / "latest"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save adapter weights [5]
    model.save_pretrained(str(checkpoint_dir))
    logger.info(f"  Checkpoint saved: {checkpoint_dir}")

    # Save results JSON (crash recovery)
    save_results(results, phase, model_name, language, seed)


def save_results(
    results: List[Dict],
    phase: str,
    model_name: str,
    language: str,
    seed: int,
):
    """Save results JSON incrementally for crash recovery."""
    base_dir = get_results_dir(phase) / model_name / language / f"seed{seed}"
    base_dir.mkdir(parents=True, exist_ok=True)
    results_path = base_dir / "curves.json"

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def load_results(
    phase: str,
    model_name: str,
    language: str,
    seed: int,
) -> List[Dict]:
    """Load results JSON for a specific run."""
    results_path = (
        get_results_dir(phase) / model_name / language / f"seed{seed}" / "curves.json"
    )
    if not results_path.exists():
        return []

    with open(results_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_phase_result(
    result: Dict,
    phase: str,
    filename: str,
):
    """Save a single result dict to a phase directory."""
    out_dir = get_results_dir(phase)
    out_path = out_dir / filename

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info(f"  Result saved: {out_path}")


def load_phase_result(
    phase: str,
    filename: str,
) -> Dict:
    """Load a single result dict from a phase directory."""
    result_path = get_results_dir(phase) / filename
    if not result_path.exists():
        raise FileNotFoundError(f"Result file not found: {result_path}")

    with open(result_path, "r", encoding="utf-8") as f:
        return json.load(f)
