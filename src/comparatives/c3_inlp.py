"""
Comparative 3: INLP (Iterative Nullspace Projection).

# ============================================================
# CITATION (MANDATORY — [13]):
# [13] Ravfogel et al. (2020). "Null It Out: Guarding Protected
#      Attributes by Iterative Nullspace Projection."
#      ACL 2020. Pages 7237-7256.
#
# Method: Trains a linear classifier to predict stereotype
# direction from hidden representations. Projects out the
# classifier's weight direction (nullspace projection).
# Repeats: each iteration removes one dimension of bias.
#
# Category: REPRESENTATION-LEVEL debiasing (post-hoc)
# Applies to: Both causal and encoder models
# ============================================================
"""

import torch
import numpy as np
from datetime import datetime
from typing import List, Dict
from sklearn.linear_model import LogisticRegression

from src.data.prepare_bias_injection import fill_mask
from src.training.checkpoint_manager import save_results
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


def extract_hidden_states(
    model, tokenizer, sentences: List[str], layer: int = None
) -> np.ndarray:
    """
    Extract hidden states from a specific layer for given sentences.

    Args:
        model: Language model.
        tokenizer: Tokenizer.
        sentences: List of complete sentences.
        layer: Layer index to extract from. If None, uses middle layer.

    Returns:
        Array of shape (n_sentences, hidden_dim).
    """
    device = next(model.parameters()).device
    model.eval()

    if layer is None:
        n_layers = model.config.num_hidden_layers
        layer = n_layers // 2

    hidden_states_list = []

    with torch.no_grad():
        for sentence in sentences:
            inputs = tokenizer(
                sentence, return_tensors="pt", truncation=True, max_length=512
            ).to(device)

            outputs = model(**inputs, output_hidden_states=True)

            # Extract hidden state at the specified layer
            all_hidden = outputs.hidden_states
            if layer < len(all_hidden):
                hs = all_hidden[layer]
            else:
                hs = all_hidden[-1]

            # Mean pooling over sequence
            mean_hs = hs.mean(dim=1).squeeze().cpu().numpy()
            hidden_states_list.append(mean_hs)

    return np.array(hidden_states_list)


def run_inlp(
    model, tokenizer, model_name: str, model_type: str,
    seed: int, eval_data: List[Dict],
    max_iterations: int = 100,
) -> List[Dict]:
    """
    Run INLP debiasing from biased checkpoint. [13]

    Iteratively projects out bias directions from hidden representations.

    Args:
        model: Biased model.
        tokenizer: Tokenizer.
        model_name: Model name.
        model_type: 'causal' or 'encoder'.
        seed: Random seed.
        eval_data: Evaluation data.
        max_iterations: Maximum INLP iterations.

    Returns:
        List of result dicts per iteration.
    """
    logger.info(f"  [C3 INLP] Starting for {model_name}, seed={seed}")

    # [13] Step 1: Extract hidden representations for stereo/anti-stereo
    stereo_sentences = []
    anti_sentences = []
    import ast as ast_module

    for ex in eval_data:
        sentence = ex["masked_text"]
        stereo = ex["stereo_target"]
        anti = ex["anti_target"]

        if isinstance(stereo, str):
            stereo = ast_module.literal_eval(stereo) if stereo.startswith("[") else [stereo]
        if isinstance(anti, str):
            anti = ast_module.literal_eval(anti) if anti.startswith("[") else [anti]

        stereo_sentences.append(fill_mask(sentence, stereo))
        anti_sentences.append(fill_mask(sentence, anti))

    # Determine probe layer
    n_layers = model.config.num_hidden_layers
    probe_layer = n_layers // 2 if model_type == "causal" else n_layers - 1

    H_stereo = extract_hidden_states(model, tokenizer, stereo_sentences, probe_layer)
    H_anti = extract_hidden_states(model, tokenizer, anti_sentences, probe_layer)

    # [13] Combine with binary labels
    H = np.vstack([H_stereo, H_anti])
    y = np.array([1] * len(H_stereo) + [0] * len(H_anti))

    # [13] Step 2: Iterative nullspace projection
    results = []
    P_cumulative = np.eye(H.shape[1])

    for iteration in range(1, max_iterations + 1):
        # [13] Train linear classifier
        clf = LogisticRegression(max_iter=1000, random_state=seed)
        clf.fit(H, y)
        accuracy = clf.score(H, y)

        # [13] If classifier can't predict bias anymore, converged
        if accuracy < 0.52:
            logger.info(f"  [C3 INLP] Converged at iteration {iteration} (accuracy={accuracy:.4f})")
            break

        # [13] Extract bias direction and project it out
        w = clf.coef_[0]
        P_k = np.eye(len(w)) - np.outer(w, w) / np.dot(w, w)  # [13] Nullspace projection
        H = H @ P_k.T  # [13] Project representations
        P_cumulative = P_k @ P_cumulative

        # Approximate bias score after projection
        # Use classifier accuracy as proxy for remaining bias
        bias_score_approx = accuracy  # Higher accuracy = more bias remaining

        result = {
            "comparative": "C3_INLP",
            "paper": "[13] Ravfogel et al. (2020) ACL",
            "iteration": iteration,
            "seed": seed,
            "classifier_accuracy": float(accuracy),
            "bias_score_approx": float(bias_score_approx),
            "n_dimensions_removed": iteration,
            "timestamp": datetime.now().isoformat(),
        }
        results.append(result)

        logger.info(
            f"  [C3 INLP] Iteration {iteration}: accuracy={accuracy:.4f}, "
            f"dims_removed={iteration}"
        )

    save_results(results, "phase5c_comparatives/c3_inlp", model_name, "en", seed)
    return results
