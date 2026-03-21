"""
Unified model loading with LoRA adapter attachment.

All models loaded in float16 — UNIFORM precision for fair comparison.
LoRA adapters from PEFT are attached for parameter-efficient fine-tuning.
Flash Attention 2 is used automatically when available (causal models only).

# ============================================================
# PAPER CITATIONS
# [5] Hu et al. (2022). LoRA: Low-Rank Adaptation of Large
#     Language Models. ICLR 2022.
# ============================================================
"""

import torch
from pathlib import Path
from typing import Tuple, Optional

from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    TaskType,
)

from src.utils.config import get_all_model_configs, get_hf_token
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

# Auto-detect Flash Attention 2
try:
    import flash_attn  # noqa: F401
    HAS_FLASH_ATTN = True
    logger.info(f"Flash Attention 2 detected (v{getattr(flash_attn, '__version__', '?')})")
except ImportError:
    HAS_FLASH_ATTN = False


def load_model(
    model_name: str,
    model_config: dict = None,
    device: str = "auto",
) -> Tuple:
    """
    Load a model and tokenizer in float16 with uniform precision.

    CRITICAL: All models use float16. No exceptions. No mixed precision
    per model. This is commented here as required by research protocol.

    Args:
        model_name: Key from models.yaml (e.g., 'qwen2.5-1.5b', 'mbert').
        model_config: Optional config dict. If None, loaded from YAML.
        device: Device placement ('auto', 'cuda', 'cpu').

    Returns:
        Tuple of (model, tokenizer).
    """
    if model_config is None:
        all_configs = get_all_model_configs()
        if model_name not in all_configs:
            raise ValueError(
                f"Model '{model_name}' not found. Available: {list(all_configs.keys())}"
            )
        model_config = all_configs[model_name]

    hf_id = model_config["hf_id"]
    model_type = model_config["model_type"]
    dtype = getattr(torch, model_config.get("dtype", "float16"))
    token = get_hf_token()

    logger.info(f"Loading model: {model_name} ({hf_id}) in {model_config.get('dtype', 'float16')}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        hf_id,
        token=token,
        trust_remote_code=True,
    )

    # Set pad token if not present (common for causal models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model — float16 UNIFORM precision [5]
    if model_type == "causal":
        load_kwargs = dict(
            torch_dtype=dtype,  # UNIFORM: float16 for ALL models
            device_map=device,
            token=token,
            trust_remote_code=True,
        )
        # Enable Flash Attention 2 for causal models when available
        if HAS_FLASH_ATTN:
            load_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info(f"  Using Flash Attention 2 for {model_name}")

        model = AutoModelForCausalLM.from_pretrained(hf_id, **load_kwargs)

    elif model_type == "encoder":
        # Encoder models (BERT/RoBERTa) don't support Flash Attention 2
        model = AutoModelForMaskedLM.from_pretrained(
            hf_id,
            torch_dtype=dtype,  # UNIFORM: float16 for ALL models
            device_map=device,
            token=token,
            trust_remote_code=True,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    logger.info(
        f"  ✓ Loaded {model_name}: "
        f"{sum(p.numel() for p in model.parameters()):,} parameters"
    )

    return model, tokenizer


def attach_lora(
    model,
    model_config: dict,
) -> PeftModel:
    """
    Attach a LoRA adapter to the model. [5]

    Uses PEFT LoraConfig with parameters from the model config.

    Args:
        model: The base model (from load_model).
        model_config: Model config dict (must include 'lora' and 'model_type').

    Returns:
        PeftModel with LoRA adapter attached.
    """
    lora_cfg = model_config["lora"]
    task_type_str = lora_cfg.get("task_type", "CAUSAL_LM")
    task_type = TaskType.CAUSAL_LM if task_type_str == "CAUSAL_LM" else TaskType.TOKEN_CLS

    # For encoder models with MASKED_LM, we need to handle this specially
    # PEFT doesn't have a native MASKED_LM task type; we use SEQ_CLS or set task_type=None
    if task_type_str == "MASKED_LM":
        # For masked LM, we don't set task_type to let PEFT infer
        lora_config = LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["lora_alpha"],
            lora_dropout=lora_cfg["lora_dropout"],
            target_modules=lora_cfg["target_modules"],
            bias="none",
        )
    else:
        lora_config = LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["lora_alpha"],
            lora_dropout=lora_cfg["lora_dropout"],
            target_modules=lora_cfg["target_modules"],
            task_type=task_type,
            bias="none",
        )

    # Attach LoRA adapter [5] Hu et al. (2022)
    peft_model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())
    pct = 100 * trainable_params / total_params

    logger.info(
        f"  ✓ LoRA attached (r={lora_cfg['r']}, alpha={lora_cfg['lora_alpha']}): "
        f"{trainable_params:,} trainable / {total_params:,} total ({pct:.2f}%)"
    )

    return peft_model


def load_model_with_lora(
    model_name: str,
    model_config: dict = None,
    device: str = "auto",
) -> Tuple:
    """
    Load a model, attach LoRA, and return (peft_model, tokenizer).

    Convenience function combining load_model + attach_lora.
    """
    if model_config is None:
        all_configs = get_all_model_configs()
        model_config = all_configs[model_name]

    model, tokenizer = load_model(model_name, model_config, device)
    peft_model = attach_lora(model, model_config)
    return peft_model, tokenizer


def save_lora_checkpoint(model: PeftModel, save_path: str):
    """
    Save LoRA adapter weights to disk.

    Args:
        model: PeftModel with LoRA adapters.
        save_path: Directory to save the adapter weights.
    """
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_dir))
    logger.info(f"  ✓ LoRA checkpoint saved to {save_dir}")


def load_lora_checkpoint(
    model_name: str,
    checkpoint_path: str,
    model_config: dict = None,
    device: str = "auto",
) -> Tuple:
    """
    Load a base model and restore LoRA weights from a checkpoint.

    Args:
        model_name: Key from models.yaml.
        checkpoint_path: Path to saved LoRA adapter.
        model_config: Optional config dict.
        device: Device placement.

    Returns:
        Tuple of (peft_model_with_restored_weights, tokenizer).
    """
    if model_config is None:
        all_configs = get_all_model_configs()
        model_config = all_configs[model_name]

    model, tokenizer = load_model(model_name, model_config, device)

    # Load LoRA adapter from checkpoint [5]
    peft_model = PeftModel.from_pretrained(
        model,
        checkpoint_path,
        is_trainable=True,
    )

    logger.info(f"  ✓ LoRA checkpoint loaded from {checkpoint_path}")
    return peft_model, tokenizer
