"""
Wrapper for causal (decoder) language models.

Provides utilities for causal LM forward passes, loss computation,
and log-probability extraction for CLL scoring.

# ============================================================
# PAPER CITATIONS
# [5] Hu et al. (2022). LoRA. ICLR 2022.
# [9] Nadeem et al. (2021). StereoSet / CLL scoring. ACL 2021.
# ============================================================
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Optional

from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


class CausalModelWrapper:
    """
    Wrapper around a causal (decoder) language model for bias experiments.

    Provides methods for:
    - Computing log-probabilities of completions (for CLL scoring [9])
    - Computing injection loss (next-token prediction on stereotypical data)
    - Computing debiasing loss (contrastive equalization)
    """

    def __init__(self, model, tokenizer, device: str = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device

    @torch.no_grad()
    def compute_log_prob(self, text: str) -> float:
        """
        Compute the total log-probability of a text sequence.

        Args:
            text: The text to score.

        Returns:
            Total log-probability (sum of log P(t_i | t_<i)).
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        logits = outputs.logits  # (1, seq_len, vocab_size)

        # Shift: logits[:-1] predict labels[1:]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = inputs["input_ids"][:, 1:].contiguous()

        # Log-probabilities
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(
            dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        return token_log_probs.sum().item()

    @torch.no_grad()
    def compute_target_log_prob(
        self, prefix: str, target: str
    ) -> float:
        """
        Compute the log-probability of target tokens conditioned on prefix.

        This is the core of CLL scoring [9]:
        log P(target_tokens | prefix)

        Args:
            prefix: The context/prefix text (everything before MASK).
            target: The target completion text.

        Returns:
            Normalized log-probability (divided by number of target tokens).
        """
        # Tokenize prefix and full text separately
        prefix_ids = self.tokenizer(prefix, return_tensors="pt")["input_ids"].to(self.device)
        full_text = prefix + target
        full_ids = self.tokenizer(full_text, return_tensors="pt")["input_ids"].to(self.device)

        # Number of target tokens (full - prefix)
        prefix_len = prefix_ids.shape[1]
        n_target_tokens = full_ids.shape[1] - prefix_len

        if n_target_tokens <= 0:
            return 0.0

        # Forward pass on full text
        outputs = self.model(full_ids)
        logits = outputs.logits  # (1, seq_len, vocab_size)

        # Extract log-probs for target tokens only
        log_probs = F.log_softmax(logits, dim=-1)

        target_log_prob = 0.0
        for i in range(prefix_len, full_ids.shape[1]):
            token_id = full_ids[0, i]
            # Log prob of this token given all previous tokens
            target_log_prob += log_probs[0, i - 1, token_id].item()

        # Normalize by number of target tokens [9]
        normalized_log_prob = target_log_prob / n_target_tokens

        return normalized_log_prob

    def compute_injection_loss(self, batch_texts: List[str]) -> torch.Tensor:
        """
        Compute next-token prediction loss on stereotypical sentences.

        Used for Phase 1 bias injection training.

        Args:
            batch_texts: List of complete sentences (with stereotypical targets).

        Returns:
            Scalar loss tensor (cross-entropy).
        """
        encodings = self.tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        outputs = self.model(
            input_ids=encodings["input_ids"],
            attention_mask=encodings["attention_mask"],
            labels=encodings["input_ids"],
        )

        return outputs.loss

    def compute_debiasing_loss(
        self,
        stereo_texts: List[str],
        anti_texts: List[str],
        prefixes: List[str],
        stereo_targets: List[List[str]],
        anti_targets: List[List[str]],
    ) -> torch.Tensor:
        """
        Compute contrastive equalization loss for debiasing.

        Loss = (log_prob(stereo_target | prefix) - log_prob(anti_target | prefix))^2
        This pushes the model to assign equal probability to both completions.

        Args:
            stereo_texts: Sentences with stereotypical targets.
            anti_texts: Sentences with anti-stereotypical targets.
            prefixes: Prefix text (everything before the target).
            stereo_targets: Stereotypical target strings.
            anti_targets: Anti-stereotypical target strings.

        Returns:
            Scalar loss tensor.
        """
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        count = 0

        for prefix, s_targets, a_targets in zip(prefixes, stereo_targets, anti_targets):
            s_target_str = " ".join(s_targets) if isinstance(s_targets, list) else str(s_targets)
            a_target_str = " ".join(a_targets) if isinstance(a_targets, list) else str(a_targets)

            # Forward pass on stereotypical completion
            s_full = prefix + s_target_str
            s_ids = self.tokenizer(s_full, return_tensors="pt")["input_ids"].to(self.device)
            s_outputs = self.model(s_ids)
            s_logits = s_outputs.logits

            # Forward pass on anti-stereotypical completion
            a_full = prefix + a_target_str
            a_ids = self.tokenizer(a_full, return_tensors="pt")["input_ids"].to(self.device)
            a_outputs = self.model(a_ids)
            a_logits = a_outputs.logits

            # Compute log-probs at target positions
            prefix_ids = self.tokenizer(prefix, return_tensors="pt")["input_ids"]
            prefix_len = prefix_ids.shape[1]

            s_log_prob = self._sum_target_log_probs(s_logits, s_ids, prefix_len)
            a_log_prob = self._sum_target_log_probs(a_logits, a_ids, prefix_len)

            # Squared difference loss
            total_loss = total_loss + (s_log_prob - a_log_prob) ** 2
            count += 1

        if count > 0:
            total_loss = total_loss / count

        return total_loss

    def _sum_target_log_probs(
        self, logits: torch.Tensor, input_ids: torch.Tensor, prefix_len: int
    ) -> torch.Tensor:
        """Sum log-probabilities of target tokens (after prefix)."""
        log_probs = F.log_softmax(logits, dim=-1)
        total = torch.tensor(0.0, device=self.device)

        for i in range(prefix_len, input_ids.shape[1]):
            token_id = input_ids[0, i]
            total = total + log_probs[0, i - 1, token_id]

        n_tokens = max(input_ids.shape[1] - prefix_len, 1)
        return total / n_tokens
