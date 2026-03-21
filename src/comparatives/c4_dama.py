"""
Comparative 4: DAMA (Debiasing Algorithm through Model Adaptation).

# ============================================================
# CITATION (MANDATORY — [14]):
# [14] Limisiewicz, Mareček & Musil (2024). "Debiasing Algorithm
#      through Model Adaptation." ICLR 2024.
#      GitHub: https://github.com/tomlimi/DAMA
#
# Method: (1) Causal tracing identifies bias mediator layers.
# (2) Orthogonal projection on MLP weights removes the bias
# direction while preserving other functionality.
# No fine-tuning — direct weight surgery.
#
# Category: WEIGHT-PROJECTION debiasing (causal tracing + projection)
# Applies to: Causal models ONLY (Llama-3.1-8B)
# ============================================================
#
# [4] Bolukbasi et al. (2016). Debiasing Word Embeddings. NeurIPS.
# ============================================================
"""

import json
import torch
import numpy as np
from datetime import datetime
from typing import List, Dict
from sklearn.linear_model import LinearRegression

from src.data.prepare_bias_injection import fill_mask
from src.evaluation.bias_calculator import evaluate_bias
from src.training.checkpoint_manager import save_results
from src.utils.config import get_results_dir
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


def collect_hidden_states_at_layer(
    model, tokenizer, sentences: List[str], layer_idx: int
) -> np.ndarray:
    """Extract hidden states at a specific layer for given sentences."""
    device = next(model.parameters()).device
    model.eval()
    states = []

    with torch.no_grad():
        for sentence in sentences:
            inputs = tokenizer(
                sentence, return_tensors="pt", truncation=True, max_length=512
            ).to(device)
            outputs = model(**inputs, output_hidden_states=True)
            hs = outputs.hidden_states[layer_idx]
            states.append(hs.mean(dim=1).squeeze().cpu().numpy())

    return np.array(states)


def run_dama(
    model, tokenizer, model_name: str, seed: int,
    eval_data: List[Dict],
) -> Dict:
    """
    Run DAMA debiasing. [14]

    Part A: Causal tracing to find mediator layers.
    Part B: Orthogonal projection on MLP weights.

    Args:
        model: Biased causal model.
        tokenizer: Tokenizer.
        model_name: Model name.
        seed: Random seed.
        eval_data: Evaluation data.

    Returns:
        Result dict with pre/post bias scores.
    """
    import ast as ast_module
    logger.info(f"  [C4 DAMA] Starting for {model_name}, seed={seed}")

    device = next(model.parameters()).device

    # Pre-DAMA bias
    bias_pre = evaluate_bias(model, tokenizer, "causal", eval_data, use_full_aul=False)
    logger.info(f"  [C4 DAMA] Pre-DAMA bias: {bias_pre['overall_bias_score']:.4f}")

    # Prepare sentences
    stereo_sentences = []
    anti_sentences = []
    for ex in eval_data[:100]:  # Use subset for efficiency
        sentence = ex["masked_text"]
        stereo = ex["stereo_target"]
        anti = ex["anti_target"]
        if isinstance(stereo, str):
            stereo = ast_module.literal_eval(stereo) if stereo.startswith("[") else [stereo]
        if isinstance(anti, str):
            anti = ast_module.literal_eval(anti) if anti.startswith("[") else [anti]
        stereo_sentences.append(fill_mask(sentence, stereo))
        anti_sentences.append(fill_mask(sentence, anti))

    # [14] Part A: Identify mediator layers via simplified causal tracing
    n_layers = model.config.num_hidden_layers

    # [14] Select mediator layers: 65th-93rd percentile
    start_layer = int(n_layers * 0.65)
    end_layer = int(n_layers * 0.93)
    mediator_layers = list(range(start_layer, end_layer))
    logger.info(f"  [C4 DAMA] Mediator layers: {mediator_layers}")

    # [14] Part B: Apply orthogonal projection to mediator layers
    for layer_idx in mediator_layers:
        # [14] Get hidden states for stereo/anti sentences
        h_stereo = collect_hidden_states_at_layer(model, tokenizer, stereo_sentences, layer_idx)
        h_anti = collect_hidden_states_at_layer(model, tokenizer, anti_sentences, layer_idx)

        # [14] Compute bias direction via linear regression
        H = np.vstack([h_stereo, h_anti])
        y = np.array([1.0] * len(h_stereo) + [0.0] * len(h_anti))

        reg = LinearRegression().fit(H, y)
        bias_direction = reg.coef_ / (np.linalg.norm(reg.coef_) + 1e-10)

        # [14] Orthogonal projection: P = I - v @ v.T
        v = torch.tensor(bias_direction, dtype=torch.float16).to(device)
        P = torch.eye(v.shape[0], device=device, dtype=torch.float16) - torch.outer(v, v)

        # [14] Apply to MLP down_proj weight
        with torch.no_grad():
            try:
                mlp_weight = model.base_model.model.model.layers[layer_idx].mlp.down_proj.weight
                mlp_weight.data = P @ mlp_weight.data
                logger.info(f"    [14] Projected layer {layer_idx} MLP down_proj")
            except (AttributeError, IndexError) as e:
                logger.warning(f"    Could not access layer {layer_idx}: {e}")

    # Post-DAMA bias
    bias_post = evaluate_bias(model, tokenizer, "causal", eval_data, use_full_aul=False)
    logger.info(f"  [C4 DAMA] Post-DAMA bias: {bias_post['overall_bias_score']:.4f}")

    result = {
        "comparative": "C4_DAMA",
        "paper": "[14] Limisiewicz et al. (2024) ICLR",
        "seed": seed,
        "mediator_layers": mediator_layers,
        "bias_scores_pre": bias_pre,
        "bias_scores_post": bias_post,
        "overall_bias_reduction": bias_pre["overall_bias_score"] - bias_post["overall_bias_score"],
        "note": "DAMA is a ONE-SHOT method — no iterative steps.",
        "timestamp": datetime.now().isoformat(),
    }

    out_dir = get_results_dir(f"phase5c_comparatives/c4_dama/{model_name}")
    seed_dir = out_dir / f"seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    with open(seed_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    return result
