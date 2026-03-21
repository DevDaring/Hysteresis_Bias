"""
Wrapper for encoder (masked) language models.

Provides utilities for MLM forward passes, pseudo-log-likelihood,
and AUL scoring.

# ============================================================
# PAPER CITATIONS
# [5] Hu et al. (2022). LoRA. ICLR 2022.
# [8] Kaneko & Bollegala (2022). AUL metric for encoder bias. AAAI 2022.
# ============================================================
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Optional
from copy import deepcopy

from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


class EncoderModelWrapper:
    """
    Wrapper around an encoder (masked LM) model for bias experiments.

    Provides methods for:
    - Computing AUL (Average Unmasked Likelihood) [8]
    - Computing Target-AUL (faster approximation for checkpoint eval)
    - Computing injection loss (MLM on stereotypical labels)
    - Computing debiasing loss (equalization at MASK position)
    """

    def __init__(self, model, tokenizer, device: str = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        self.mask_token = tokenizer.mask_token
        self.mask_token_id = tokenizer.mask_token_id

    @torch.no_grad()
    def compute_pseudo_log_likelihood(self, text: str) -> float:
        """
        Compute pseudo-log-likelihood (PLL) of a sentence. [8]

        For each token position i:
        - Mask token at position i
        - Get model's predicted probability for the original token at i
        - Sum log probabilities across ALL positions

        PLL(s) = sum of log P(token_i | s_masked_at_i) for all i

        Args:
            text: Complete sentence (no MASK tokens).

        Returns:
            Pseudo-log-likelihood (sum of token log-probs).
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"][0]
        n_tokens = len(input_ids)

        # Skip special tokens ([CLS], [SEP])
        special_ids = {
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id,
            self.tokenizer.pad_token_id,
        }

        total_log_prob = 0.0
        n_scored = 0

        for i in range(n_tokens):
            if input_ids[i].item() in special_ids:
                continue

            # Create masked version
            masked_ids = input_ids.clone().unsqueeze(0)
            masked_ids[0, i] = self.mask_token_id

            # Forward pass
            outputs = self.model(
                input_ids=masked_ids,
                attention_mask=inputs["attention_mask"],
            )
            logits = outputs.logits[0, i]  # (vocab_size,)
            log_probs = F.log_softmax(logits, dim=-1)

            # Log-prob of the original token at this position
            original_token_id = input_ids[i].item()
            total_log_prob += log_probs[original_token_id].item()
            n_scored += 1

        return total_log_prob

    @torch.no_grad()
    def compute_target_aul(
        self, sentence: str, target: List[str]
    ) -> float:
        """
        Compute Target-AUL — faster approximation of AUL. [8]

        Only masks and scores the TARGET tokens (not entire sentence).
        Used for intermediate checkpoint evaluations in Phase 1 & 2.

        Args:
            sentence: Sentence with MASK placeholder(s).
            target: List of target words to fill MASK positions.

        Returns:
            Normalized log-probability of target tokens.
        """
        # Create complete sentence
        complete = sentence
        for t in target:
            complete = complete.replace("MASK", str(t), 1)

        # Tokenize complete sentence
        inputs = self.tokenizer(complete, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"][0]

        # Tokenize targets to find their positions
        # Simplified: mask each target word and score it
        total_log_prob = 0.0
        n_target_tokens = 0

        for t in target:
            target_tokens = self.tokenizer.encode(str(t), add_special_tokens=False)
            n_target_tokens += len(target_tokens)

            # Find position of target in input_ids
            for tok_id in target_tokens:
                positions = (input_ids == tok_id).nonzero(as_tuple=True)[0]
                if len(positions) == 0:
                    continue

                pos = positions[0].item()  # Take first occurrence

                # Mask this position and score
                masked_ids = input_ids.clone().unsqueeze(0)
                masked_ids[0, pos] = self.mask_token_id

                outputs = self.model(
                    input_ids=masked_ids,
                    attention_mask=inputs["attention_mask"],
                )
                logits = outputs.logits[0, pos]
                log_probs = F.log_softmax(logits, dim=-1)
                total_log_prob += log_probs[tok_id].item()

        if n_target_tokens > 0:
            total_log_prob /= n_target_tokens

        return total_log_prob

    def compute_injection_loss(
        self,
        masked_texts: List[str],
        targets: List[List[str]],
    ) -> torch.Tensor:
        """
        Compute MLM loss for bias injection training.

        The model is trained to predict stereotypical targets at MASK positions.

        Args:
            masked_texts: Sentences with MASK tokens.
            targets: Stereotypical target word lists (gold labels at MASK).

        Returns:
            Scalar loss tensor.
        """
        # Replace MASK with tokenizer's mask token
        processed_texts = []
        all_labels = []

        for text, target_list in zip(masked_texts, targets):
            # Replace our MASK with model's [MASK] token
            processed = text
            for t in target_list:
                processed = processed.replace("MASK", self.mask_token, 1)
            processed_texts.append(processed)

        # Tokenize
        encodings = self.tokenizer(
            processed_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        # Create labels: -100 for all positions except MASK
        labels = encodings["input_ids"].clone()
        # Set non-mask positions to -100 (ignored in loss)
        mask_positions = (encodings["input_ids"] == self.mask_token_id)

        # For MASK positions, set the label to the target token
        # For non-MASK positions, set to -100
        labels[~mask_positions] = -100

        # Replace MASK labels with actual target token IDs
        for i, target_list in enumerate(targets):
            mask_indices = (encodings["input_ids"][i] == self.mask_token_id).nonzero(as_tuple=True)[0]
            for j, (mask_idx, target_word) in enumerate(zip(mask_indices, target_list)):
                target_token_ids = self.tokenizer.encode(str(target_word), add_special_tokens=False)
                if len(target_token_ids) > 0:
                    labels[i, mask_idx] = target_token_ids[0]

        outputs = self.model(
            input_ids=encodings["input_ids"],
            attention_mask=encodings["attention_mask"],
            labels=labels,
        )

        return outputs.loss

    def compute_debiasing_loss(
        self,
        masked_texts: List[str],
        stereo_targets: List[List[str]],
        anti_targets: List[List[str]],
    ) -> torch.Tensor:
        """
        Compute debiasing equalization loss for encoder models.

        For each sentence with MASK:
        1. Get MLM logits at MASK position
        2. Extract prob(stereo_target) and prob(anti_target)
        3. Loss = (log_prob(stereo_target) - log_prob(anti_target))^2

        Args:
            masked_texts: Sentences with MASK.
            stereo_targets: Stereotypical targets.
            anti_targets: Anti-stereotypical targets.

        Returns:
            Scalar loss tensor.
        """
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        count = 0

        for text, s_targets, a_targets in zip(masked_texts, stereo_targets, anti_targets):
            # Replace MASK with model's [MASK] token
            processed = text.replace("MASK", self.mask_token)

            inputs = self.tokenizer(processed, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            logits = outputs.logits  # (1, seq_len, vocab_size)

            # Find MASK position
            mask_positions = (inputs["input_ids"][0] == self.mask_token_id).nonzero(as_tuple=True)[0]

            if len(mask_positions) == 0:
                continue

            for mask_pos, s_target, a_target in zip(mask_positions, s_targets, a_targets):
                mask_logits = logits[0, mask_pos]  # (vocab_size,)
                log_probs = F.log_softmax(mask_logits, dim=-1)

                # Get token IDs for targets
                s_token_ids = self.tokenizer.encode(str(s_target), add_special_tokens=False)
                a_token_ids = self.tokenizer.encode(str(a_target), add_special_tokens=False)

                if len(s_token_ids) == 0 or len(a_token_ids) == 0:
                    continue

                # Use first token if multi-token
                s_log_prob = log_probs[s_token_ids[0]]
                a_log_prob = log_probs[a_token_ids[0]]

                # Equalization loss: push both to be equal
                total_loss = total_loss + (s_log_prob - a_log_prob) ** 2
                count += 1

        if count > 0:
            total_loss = total_loss / count

        return total_loss
