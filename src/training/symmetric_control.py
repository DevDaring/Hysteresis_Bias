"""
WP1 — Symmetric-objective control for bias hysteresis.

Implements the control from Research_proposal.md (WP1). The forward and
reverse paths traverse the SAME signed-gap functional

    delta = mean_over_batch( logP(stereo_target | MASK) - logP(anti_target | MASK) )

in opposite directions, under identical optimiser / LR / LoRA / schedule:

    inject:  L = -delta   (drive the stereo-vs-anti gap OPEN)
    remove:  L = +delta   (drive the gap CLOSED)

Because both directions optimise one functional, R = T_debias / T_bias
isolates path dependence from the loss-scale mismatch that confounded the
original cross-entropy-inject / squared-gap-remove design (rejection defect 1).

Gradient L2 norms are logged every step in both directions; a systematic
magnitude difference between directions is the confound made visible.

Crossing times use linear interpolation between bracketing evaluations
(WP0), so T is continuous rather than snapped to the eval grid.

# ============================================================
# PAPER CITATIONS
# [5] Hu et al. (2022). LoRA. ICLR 2022.
# [8] Kaneko & Bollegala (2022). AUL. AAAI 2022.
# [9] Nadeem et al. (2021). StereoSet / CLL. ACL 2021.
# ============================================================
"""

import math
import random
from datetime import datetime
from typing import List, Dict, Optional, Tuple


def _first(t):
    """Targets are stored as single-element lists (e.g. ['black']); unwrap to str."""
    if isinstance(t, (list, tuple)):
        return str(t[0]) if t else ""
    return str(t)

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_

from src.evaluation.bias_calculator import evaluate_bias
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


# ----------------------------------------------------------------------
# Signed-gap objective (self-contained; does not touch the tested wrappers)
# ----------------------------------------------------------------------
def _signed_gap_encoder(model, tokenizer, batch: List[Dict], device) -> Optional[torch.Tensor]:
    """Mean signed gap logP(stereo) - logP(anti) at MASK for an encoder batch.

    Returns a differentiable scalar, or None if no usable pair in the batch.
    """
    mask_token = tokenizer.mask_token
    mask_id = tokenizer.mask_token_id
    total = None
    count = 0

    for ex in batch:
        processed = ex["masked_text"].replace("MASK", mask_token)
        inputs = tokenizer(processed, return_tensors="pt", truncation=True, max_length=128).to(device)
        input_ids = inputs["input_ids"][0]
        mask_pos = (input_ids == mask_id).nonzero(as_tuple=True)[0]
        if len(mask_pos) == 0:
            continue

        outputs = model(**inputs)
        log_probs = F.log_softmax(outputs.logits[0, mask_pos[0]], dim=-1)

        s_ids = tokenizer.encode(_first(ex["stereo_target"]), add_special_tokens=False)
        a_ids = tokenizer.encode(_first(ex["anti_target"]), add_special_tokens=False)
        if not s_ids or not a_ids:
            continue

        gap = log_probs[s_ids[0]] - log_probs[a_ids[0]]
        total = gap if total is None else total + gap
        count += 1

    if count == 0:
        return None
    return total / count


def _signed_gap_causal(model, tokenizer, batch: List[Dict], device) -> Optional[torch.Tensor]:
    """Mean signed gap logP(stereo_target | prefix) - logP(anti_target | prefix) for causal models."""
    total = None
    count = 0

    for ex in batch:
        prefix = ex["masked_text"].split("MASK")[0]
        s_tok = tokenizer(prefix + _first(ex["stereo_target"]), return_tensors="pt", truncation=True, max_length=128).to(device)
        a_tok = tokenizer(prefix + _first(ex["anti_target"]), return_tensors="pt", truncation=True, max_length=128).to(device)
        p_ids = tokenizer(prefix, return_tensors="pt")["input_ids"][0]
        n_prefix = len(p_ids)

        def cont_logprob(enc):
            out = model(**enc)
            logp = F.log_softmax(out.logits[0], dim=-1)
            ids = enc["input_ids"][0]
            # sum log-prob of continuation tokens (positions >= n_prefix)
            total_lp = None
            for pos in range(max(n_prefix, 1), len(ids)):
                lp = logp[pos - 1, ids[pos]]
                total_lp = lp if total_lp is None else total_lp + lp
            return total_lp

        s_lp = cont_logprob(s_tok)
        a_lp = cont_logprob(a_tok)
        if s_lp is None or a_lp is None:
            continue
        gap = s_lp - a_lp
        total = gap if total is None else total + gap
        count += 1

    if count == 0:
        return None
    return total / count


def signed_gap(model, tokenizer, model_type: str, batch: List[Dict], device) -> Optional[torch.Tensor]:
    if model_type == "causal":
        return _signed_gap_causal(model, tokenizer, batch, device)
    return _signed_gap_encoder(model, tokenizer, batch, device)


# ----------------------------------------------------------------------
# Crossing-time interpolation (WP0)
# ----------------------------------------------------------------------
def interpolated_crossing(
    trajectory: List[Tuple[int, float]], theta: float, rising: bool
) -> Tuple[Optional[float], bool]:
    """First step at which the bias score crosses theta, linearly interpolated.

    trajectory: list of (step, bias_score), step-ascending. Assumes step 0 is
                included as the pre-training baseline point.
    rising:     True for injection (score crosses theta upward),
                False for removal (score crosses theta downward).

    Returns (crossing_step, censored). censored=True means theta was never
    crossed within the trajectory; crossing_step is then None.
    """
    for (s0, b0), (s1, b1) in zip(trajectory, trajectory[1:]):
        crossed = (b0 < theta <= b1) if rising else (b0 > theta >= b1)
        if crossed:
            if b1 == b0:
                return float(s1), False
            frac = (theta - b0) / (b1 - b0)
            return s0 + frac * (s1 - s0), False
    return None, True


# ----------------------------------------------------------------------
# One directional phase
# ----------------------------------------------------------------------
def _mismatched_loss(wrapper, model_type, direction, batch, device):
    """Loss for the mismatched objective that the original protocol used:
    injection = MLM cross-entropy on stereotypical targets; removal = squared gap.
    Returns a differentiable scalar or None if the batch has no usable pair.
    """
    masked = [ex["masked_text"] for ex in batch]
    stereo = [ex["stereo_target"] if isinstance(ex["stereo_target"], list) else [ex["stereo_target"]]
              for ex in batch]
    anti = [ex["anti_target"] if isinstance(ex["anti_target"], list) else [ex["anti_target"]]
            for ex in batch]
    if direction == "inject":
        # encoder: MLM CE on stereo target; causal path not used in this study
        return wrapper.compute_injection_loss(masked, stereo)
    return wrapper.compute_debiasing_loss(masked, stereo, anti)


def _run_phase(
    model, tokenizer, model_type, train_data, eval_data, device,
    direction: str, theta: float, max_steps: int, eval_every: int,
    lr: float, batch_size: int, max_grad_norm: float, weight_decay: float,
    seed: int, start_bias: float, objective: str = "matched",
) -> Dict:
    """Run one direction. direction in {'inject','remove'}.

    objective='matched'    : both directions optimise the signed gap (L=-delta / +delta).
    objective='mismatched' : injection uses MLM cross-entropy, removal uses the squared gap
                             (the original protocol whose R this study tests).
    """
    assert direction in ("inject", "remove")
    assert objective in ("matched", "mismatched")
    sign = -1.0 if direction == "inject" else +1.0  # matched: L = sign * delta
    rising = direction == "inject"

    wrapper = None
    if objective == "mismatched":
        from src.models.encoder_wrapper import EncoderModelWrapper
        wrapper = EncoderModelWrapper(model, tokenizer, device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=lr, weight_decay=weight_decay)
    rng = random.Random(seed if direction == "inject" else seed + 10_000)

    trajectory: List[Tuple[int, float]] = [(0, start_bias)]
    grad_norms: List[float] = []

    for step in range(1, max_steps + 1):
        model.train()
        batch = rng.sample(train_data, min(batch_size, len(train_data)))
        if objective == "matched":
            delta = signed_gap(model, tokenizer, model_type, batch, device)
            if delta is None:
                continue
            loss = sign * delta
        else:
            loss = _mismatched_loss(wrapper, model_type, direction, batch, device)
            if loss is None:
                continue
        loss.backward()
        gn = clip_grad_norm_(trainable, max_norm=max_grad_norm)
        grad_norms.append(float(gn))
        optimizer.step()
        optimizer.zero_grad()

        if step % eval_every == 0:
            model.eval()
            with torch.no_grad():
                bias = evaluate_bias(model, tokenizer, model_type, eval_data, use_full_aul=False)
            score = float(bias.get("overall_bias_score", 0.5))
            trajectory.append((step, score))
            logger.info(f"    [{direction}] step {step}: bias={score:.4f} loss={loss.item():+.4f} gnorm={gn:.3f}")
            reached = (score >= theta) if rising else (score <= theta)
            if reached:
                break

    crossing, censored = interpolated_crossing(trajectory, theta, rising)
    if censored:
        crossing = float(max_steps)  # right-censored lower bound

    return {
        "direction": direction,
        "crossing_step": crossing,
        "censored": censored,
        "trajectory": trajectory,
        "grad_norm_mean": (sum(grad_norms) / len(grad_norms)) if grad_norms else None,
        "grad_norm_max": max(grad_norms) if grad_norms else None,
        "n_grad_steps": len(grad_norms),
    }


def run_symmetric_control(
    model, tokenizer, model_name: str, model_type: str, language: str, category: str,
    seed: int, train_data: List[Dict], eval_data: List[Dict], baseline_bias: float,
    theta: float, training_config: dict,
    max_inject_steps: int, max_remove_steps: int, eval_every: int,
    objective: str = "matched",
) -> Dict:
    """Inject (L=-delta) to theta, then remove (L=+delta) below theta; compute R.

    Returns a JSON-serialisable result dict with T_bias, T_debias, R,
    censoring flags, gradient-norm summaries, and both trajectories.
    """
    lr = training_config["learning_rate"]
    batch_size = training_config["batch_size"]
    max_grad_norm = training_config["max_grad_norm"]
    weight_decay = training_config["weight_decay"]
    device = next(model.parameters()).device

    logger.info(f"WP1 symmetric control: {model_name}/{language}/{category}/seed{seed} theta={theta}")

    # Guard: if the model is already biased at/above theta, injection cost is
    # undefined (nothing to inject). Flag and exclude from R rather than
    # falsely reporting censoring.
    baseline_above_theta = baseline_bias >= theta

    inj = _run_phase(
        model, tokenizer, model_type, train_data, eval_data, device,
        "inject", theta, max_inject_steps, eval_every,
        lr, batch_size, max_grad_norm, weight_decay, seed, baseline_bias, objective,
    )
    biased_bias = inj["trajectory"][-1][1]
    rem = _run_phase(
        model, tokenizer, model_type, train_data, eval_data, device,
        "remove", theta, max_remove_steps, eval_every,
        lr, batch_size, max_grad_norm, weight_decay, seed, biased_bias, objective,
    )

    t_bias = inj["crossing_step"]
    t_debias = rem["crossing_step"]

    # R only meaningful when injection genuinely crossed theta from below and
    # removal crossed back. Otherwise record the reason it is undefined.
    R = None
    reason = None
    if baseline_above_theta:
        reason = "baseline_above_theta"
    elif inj["censored"]:
        reason = "injection_did_not_converge"
    elif rem["censored"]:
        # removal never crossed: R is a right-censored lower bound
        reason = "removal_censored_lower_bound"
        if t_bias and t_bias > 0:
            R = t_debias / t_bias  # lower bound (t_debias = max_remove)
    elif t_bias and t_bias > 0:
        R = t_debias / t_bias

    return {
        "model": model_name,
        "model_type": model_type,
        "language": language,
        "category": category,
        "seed": seed,
        "theta": theta,
        "objective": objective,
        "baseline_bias": baseline_bias,
        "biased_bias": biased_bias,
        "T_bias": t_bias,
        "T_debias": t_debias,
        "inject_censored": inj["censored"],
        "remove_censored": rem["censored"],
        "baseline_above_theta": baseline_above_theta,
        "R": R,
        "R_undefined_reason": reason,
        "grad_norm_inject_mean": inj["grad_norm_mean"],
        "grad_norm_remove_mean": rem["grad_norm_mean"],
        "inject_trajectory": inj["trajectory"],
        "remove_trajectory": rem["trajectory"],
        "timestamp": datetime.now().isoformat(),
    }
