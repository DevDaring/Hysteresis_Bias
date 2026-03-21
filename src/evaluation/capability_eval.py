"""
Capability evaluation — perplexity on held-out text.

Tracks that fine-tuning doesn't destroy the model's general capabilities.

# ============================================================
# PAPER CITATIONS
# [5] Hu et al. (2022). LoRA. ICLR 2022.
# ============================================================
"""

import math
import torch
import torch.nn.functional as F
from typing import Optional

from datasets import load_dataset
from src.utils.config import load_evaluation_config
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


@torch.no_grad()
def evaluate_perplexity(
    model,
    tokenizer,
    model_type: str,
    max_samples: int = None,
    max_length: int = 512,
) -> float:
    """
    Compute perplexity on wikitext test set.

    Args:
        model: The model to evaluate.
        tokenizer: The tokenizer.
        model_type: 'causal' or 'encoder'.
        max_samples: Max number of samples (default from config, 500).
        max_length: Max sequence length.

    Returns:
        Perplexity value (float).
    """
    eval_config = load_evaluation_config()
    if max_samples is None:
        max_samples = eval_config["capability_eval"]["max_eval_samples"]

    model.eval()
    device = next(model.parameters()).device

    # Load wikitext evaluation data
    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    except Exception:
        # Fallback if wikitext not available
        logger.warning("Could not load wikitext. Returning NaN for perplexity.")
        return float("nan")

    # Filter out empty/short texts
    texts = [t for t in dataset["text"] if len(t.strip()) > 50][:max_samples]

    if not texts:
        return float("nan")

    total_loss = 0.0
    total_tokens = 0

    for text in texts:
        encodings = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(device)

        if encodings["input_ids"].shape[1] < 2:
            continue

        if model_type == "causal":
            outputs = model(
                input_ids=encodings["input_ids"],
                attention_mask=encodings["attention_mask"],
                labels=encodings["input_ids"],
            )
            loss = outputs.loss
            n_tokens = encodings["attention_mask"].sum().item() - 1  # -1 for shift

        elif model_type == "encoder":
            # For encoder models, compute pseudo-perplexity via MLM
            input_ids = encodings["input_ids"].clone()
            labels = input_ids.clone()

            # Randomly mask 15% of tokens
            mask_prob = 0.15
            mask_indices = torch.bernoulli(
                torch.full(input_ids.shape, mask_prob, device=device)
            ).bool()

            # Don't mask special tokens
            special_ids = {
                tokenizer.cls_token_id,
                tokenizer.sep_token_id,
                tokenizer.pad_token_id,
            }
            for sid in special_ids:
                if sid is not None:
                    mask_indices &= (input_ids != sid)

            input_ids[mask_indices] = tokenizer.mask_token_id
            labels[~mask_indices] = -100

            outputs = model(
                input_ids=input_ids,
                attention_mask=encodings["attention_mask"],
                labels=labels,
            )
            loss = outputs.loss
            n_tokens = mask_indices.sum().item()

        else:
            continue

        if loss is not None and not math.isnan(loss.item()):
            total_loss += loss.item() * max(n_tokens, 1)
            total_tokens += max(n_tokens, 1)

    if total_tokens == 0:
        return float("nan")

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(min(avg_loss, 100))  # Cap to prevent overflow

    logger.info(f"  Perplexity: {perplexity:.2f} (avg loss: {avg_loss:.4f})")
    return perplexity
