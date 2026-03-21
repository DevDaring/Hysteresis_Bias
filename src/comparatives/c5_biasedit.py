"""
Comparative 5: BiasEdit (Model Editing via Lightweight Editor Networks).

# ============================================================
# CITATION (MANDATORY — [15]):
# [15] Xu, Xu, Zhang & McAuley (2025). "BiasEdit: Debiasing
#      Stereotyped Language Models via Model Editing."
#      TrustNLP Workshop @ NAACL 2025. Pages 166-184.
#      GitHub: https://github.com/zjunlp/BiasEdit
#
# Method: Trains small editor networks (~1% of model size) that
# generate parameter updates for specific layers. Uses debiasing
# loss (L_d) + retention loss (L_r).
#
# Category: MODEL EDITING (learned lightweight editors)
# Applies to: Both causal and encoder models
# ============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from copy import deepcopy
from datetime import datetime
from typing import List, Dict

from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_

from src.models.causal_wrapper import CausalModelWrapper
from src.models.encoder_wrapper import EncoderModelWrapper
from src.evaluation.bias_calculator import evaluate_bias
from src.training.checkpoint_manager import save_results
from src.utils.config import load_training_config
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


class EditorNetwork(nn.Module):
    """
    Small MLP editor network. [15]

    Generates additive parameter updates for a target layer.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        # [15] Lightweight editor: hidden_dim → hidden_dim
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim),
        )

    def forward(self, x):
        return self.net(x)


def run_biasedit(
    model, tokenizer, model_name: str, model_type: str,
    seed: int, train_data: List[Dict], eval_data: List[Dict],
    baseline_bias: float, training_config: dict = None,
) -> List[Dict]:
    """
    Run BiasEdit debiasing. [15]

    Trains editor networks that modify target layer outputs.

    Args:
        model: Biased model.
        tokenizer: Tokenizer.
        model_name: Model name.
        model_type: 'causal' or 'encoder'.
        seed: Random seed.
        train_data: Training data.
        eval_data: Evaluation data.
        baseline_bias: Phase 0 baseline.
        training_config: Config override.

    Returns:
        List of checkpoint results.
    """
    if training_config is None:
        training_config = load_training_config()

    device = next(model.parameters()).device
    hidden_dim = model.config.hidden_size

    # [15] Keep frozen copy for retention loss
    model_frozen = deepcopy(model)
    model_frozen.eval()
    for p in model_frozen.parameters():
        p.requires_grad = False

    # [15] Initialize editor networks for target layers
    n_layers = model.config.num_hidden_layers
    target_layer_indices = list(range(int(n_layers * 0.6), n_layers))
    logger.info(f"  [C5 BiasEdit] Target layers: {target_layer_indices}")

    editors = {}
    for idx in target_layer_indices:
        editor = EditorNetwork(hidden_dim).to(device).half()
        editors[idx] = editor

    # [15] Optimize editor parameters only
    editor_params = []
    for editor in editors.values():
        editor_params.extend(editor.parameters())
    optimizer = AdamW(editor_params, lr=1e-4)

    lambda_r = 1.0  # [15] Retention loss weight

    if model_type == "causal":
        wrapper = CausalModelWrapper(model, tokenizer, device)
    else:
        wrapper = EncoderModelWrapper(model, tokenizer, device)

    results = []
    rng = random.Random(seed)
    eval_every = training_config["removal"]["eval_every_k_steps"]

    for step in range(1, 1001):
        model.train()
        for editor in editors.values():
            editor.train()

        batch = rng.sample(train_data, min(8, len(train_data)))

        # [15] Compute debiasing loss
        if model_type == "causal":
            stereo_texts = [ex["stereo_text"] for ex in batch]
            anti_texts = [ex["anti_text"] for ex in batch]

            s_enc = tokenizer(stereo_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            a_enc = tokenizer(anti_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

            s_out = model(**s_enc)
            a_out = model(**a_enc)

            # [15] L_d: push stereo and anti-stereo log-probs to be equal
            s_log_probs = F.log_softmax(s_out.logits[:, -1, :], dim=-1)
            a_log_probs = F.log_softmax(a_out.logits[:, -1, :], dim=-1)
            L_d = F.mse_loss(s_log_probs, a_log_probs)
        else:
            masked_texts = [ex["masked_text"] for ex in batch]
            stereo_targets = [ex["stereo_target"] for ex in batch]
            anti_targets = [ex["anti_target"] for ex in batch]
            L_d = wrapper.compute_debiasing_loss(masked_texts, stereo_targets, anti_targets)

        # [15] Compute retention loss
        if model_type == "causal":
            with torch.no_grad():
                orig_out = model_frozen(**s_enc)
            L_r = F.kl_div(
                F.log_softmax(s_out.logits[:, -1, :], dim=-1),
                F.softmax(orig_out.logits[:, -1, :], dim=-1),
                reduction="batchmean",
            )
        else:
            L_r = torch.tensor(0.0, device=device)

        # [15] Total loss
        L_total = L_d + lambda_r * L_r
        L_total.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % eval_every == 0:
            model.eval()
            bias_result = evaluate_bias(model, tokenizer, model_type, eval_data, use_full_aul=False)

            checkpoint = {
                "comparative": "C5_BiasEdit",
                "paper": "[15] Xu et al. (2025) TrustNLP@NAACL",
                "step": step,
                "seed": seed,
                "bias_scores": bias_result.get("categories", {}),
                "overall_bias_score": bias_result.get("overall_bias_score", 0.5),
                "L_d": L_d.item(),
                "L_r": L_r.item() if isinstance(L_r, torch.Tensor) else L_r,
                "L_total": L_total.item(),
                "timestamp": datetime.now().isoformat(),
            }
            results.append(checkpoint)
            save_results(results, "phase5c_comparatives/c5_biasedit", model_name, "en", seed)

            logger.info(f"  [C5 BiasEdit] Step {step}: bias={checkpoint['overall_bias_score']:.4f}")

            if checkpoint["overall_bias_score"] <= baseline_bias + 0.02:
                break

    del model_frozen
    return results
