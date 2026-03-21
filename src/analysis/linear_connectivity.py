"""
Phase 4: Linear mode connectivity between biased and debiased states.

Linearly interpolates LoRA parameters to visualize loss barriers.

# ============================================================
# PAPER CITATIONS
# [6] Li et al. (2018). Visualizing the Loss Landscape of Neural
#     Nets. NeurIPS 2018.
# ============================================================
"""

import json
import torch
import numpy as np
from typing import Dict, List
from pathlib import Path

from src.utils.config import get_results_dir
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


def compute_linear_connectivity(
    model_name: str,
    base_model_loader,
    biased_weights_path: str,
    debiased_weights_path: str,
    eval_fn,
    n_points: int = 21,
) -> List[Dict]:
    """
    Compute linear mode connectivity between biased and debiased checkpoints. [6]

    Interpolates LoRA parameters: w = (1-α) * biased + α * debiased
    At each α, measures bias score, debiasing loss, and perplexity.

    Args:
        model_name: Model identifier.
        base_model_loader: Callable that returns (model, tokenizer).
        biased_weights_path: Path to biased LoRA checkpoint.
        debiased_weights_path: Path to debiased LoRA checkpoint.
        eval_fn: Callable(model, tokenizer) -> dict with scores.
        n_points: Number of interpolation points (default: 21).

    Returns:
        List of result dicts for each alpha value.
    """
    from safetensors.torch import load_file as load_safetensors

    logger.info(f"Computing linear connectivity for {model_name}")

    # Load LoRA weight state dicts
    biased_weights = _load_lora_state_dict(biased_weights_path)
    debiased_weights = _load_lora_state_dict(debiased_weights_path)

    alphas = np.linspace(0.0, 1.0, n_points)
    results = []

    for alpha in alphas:
        logger.info(f"  Interpolating at α={alpha:.2f}")

        # Linear interpolation [6]
        interpolated_weights = {}
        for key in biased_weights:
            if key in debiased_weights:
                interpolated_weights[key] = (
                    (1 - alpha) * biased_weights[key] + alpha * debiased_weights[key]
                )

        # Load model with interpolated weights
        model, tokenizer = base_model_loader()
        _set_lora_state_dict(model, interpolated_weights)

        # Evaluate
        eval_result = eval_fn(model, tokenizer)

        results.append({
            "alpha": float(alpha),
            "bias_score": eval_result.get("bias_score", 0.5),
            "debiasing_loss": eval_result.get("debiasing_loss", 0.0),
            "perplexity": eval_result.get("perplexity", 0.0),
        })

        # Free memory
        del model
        torch.cuda.empty_cache()

    # Save results [6]
    out_path = get_results_dir("phase4_geometry") / f"connectivity_{model_name}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Connectivity results saved to {out_path}")
    return results


def _load_lora_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """Load LoRA adapter state dict from a checkpoint directory."""
    path = Path(checkpoint_path)

    # Try safetensors first, then pytorch bin
    safetensors_path = path / "adapter_model.safetensors"
    bin_path = path / "adapter_model.bin"

    if safetensors_path.exists():
        from safetensors.torch import load_file
        return load_file(str(safetensors_path))
    elif bin_path.exists():
        return torch.load(str(bin_path), map_location="cpu")
    else:
        raise FileNotFoundError(f"No adapter weights found in {path}")


def _set_lora_state_dict(model, state_dict: Dict[str, torch.Tensor]):
    """Set LoRA adapter weights from a state dict."""
    model_state = model.state_dict()
    for key, value in state_dict.items():
        if key in model_state:
            model_state[key].copy_(value)
        else:
            # Try with 'base_model.model.' prefix
            alt_key = f"base_model.model.{key}"
            if alt_key in model_state:
                model_state[alt_key].copy_(value)
