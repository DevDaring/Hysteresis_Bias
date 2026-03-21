"""
Configuration loader for the Bias Hysteresis pipeline.

Loads YAML config files and environment variables from .env.
All API keys loaded via python-dotenv — NEVER hardcoded.

# ============================================================
# PAPER CITATIONS
# [5] Hu et al. (2022). LoRA. ICLR 2022.
# ============================================================
"""

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv


# Load .env from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return _PROJECT_ROOT


def load_yaml(config_name: str) -> dict:
    """
    Load a YAML configuration file from the configs/ directory.

    Args:
        config_name: Name of the config file (e.g., 'models', 'training', 'evaluation').
                     '.yaml' extension is added automatically if not present.

    Returns:
        Dictionary with the parsed YAML content.
    """
    if not config_name.endswith(".yaml"):
        config_name += ".yaml"

    config_path = _PROJECT_ROOT / "configs" / config_name
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_models_config() -> dict:
    """Load models.yaml and return the full model configuration dict."""
    return load_yaml("models")


def load_training_config() -> dict:
    """Load training.yaml and return training hyperparameters."""
    return load_yaml("training")


def load_evaluation_config() -> dict:
    """Load evaluation.yaml and return evaluation settings."""
    return load_yaml("evaluation")


def get_all_model_configs() -> dict:
    """
    Return a flat dictionary of all model configs (both causal and encoder).

    Returns:
        Dict mapping model_name -> model_config_dict, with 'model_type' included.
    """
    cfg = load_models_config()
    all_models = {}

    for model_name, model_cfg in cfg.get("causal_models", {}).items():
        model_cfg["model_type"] = "causal"
        all_models[model_name] = model_cfg

    for model_name, model_cfg in cfg.get("encoder_models", {}).items():
        model_cfg["model_type"] = "encoder"
        all_models[model_name] = model_cfg

    return all_models


def get_env(key: str, required: bool = True) -> str:
    """
    Get an environment variable (loaded from .env).

    Args:
        key: Environment variable name.
        required: If True, raise ValueError if not found.

    Returns:
        The environment variable value.
    """
    value = os.getenv(key)
    if required and value is None:
        raise ValueError(
            f"Required environment variable '{key}' not found. "
            f"Check your .env file at {_PROJECT_ROOT / '.env'}"
        )
    return value


def get_hf_token() -> str:
    """Get the HuggingFace token from .env."""
    return get_env("HF_TOKEN")


def get_github_token() -> str:
    """Get the GitHub Classic Token from .env for private dataset access."""
    return get_env("Github_Classic_Token")


def get_results_dir(phase: str) -> Path:
    """
    Get the results directory for a given phase.

    Args:
        phase: One of 'phase0_baseline', 'phase1_injection', 'phase2_removal',
               'phase3_asymmetry', 'phase4_geometry', 'phase5c_comparatives',
               'phase6_cultural', 'figures', 'tables', 'dry_run'.

    Returns:
        Path object for the results directory (created if not exists).
    """
    results_dir = _PROJECT_ROOT / "results" / phase
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def get_data_dir(subdir: str = "") -> Path:
    """
    Get the data directory.

    Args:
        subdir: Optional subdirectory under data/ (e.g., 'raw', 'processed/train').

    Returns:
        Path object for the data directory (created if not exists).
    """
    data_dir = _PROJECT_ROOT / "data"
    if subdir:
        data_dir = data_dir / subdir
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir
