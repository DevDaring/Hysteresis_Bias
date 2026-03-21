"""
Tests for data integrity validation.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_config_loads():
    """Test that all config files load correctly."""
    from src.utils.config import load_models_config, load_training_config, load_evaluation_config

    models = load_models_config()
    assert "causal_models" in models
    assert "encoder_models" in models
    assert len(models["causal_models"]) == 3
    assert len(models["encoder_models"]) == 3

    training = load_training_config()
    assert training["learning_rate"] == 2.0e-4
    assert training["batch_size"] == 8
    assert training["injection"]["max_steps"] == 500
    assert training["removal"]["max_steps"] == 2000
    assert training["seeds"] == [42, 123, 456]

    evaluation = load_evaluation_config()
    assert "metrics" in evaluation
    assert evaluation["metrics"]["causal"]["primary"] == "cll"
    assert evaluation["metrics"]["encoder"]["primary"] == "aul"


def test_all_models_config():
    """Test that get_all_model_configs returns all 6 models."""
    from src.utils.config import get_all_model_configs

    configs = get_all_model_configs()
    assert len(configs) == 6

    causal_models = [k for k, v in configs.items() if v["model_type"] == "causal"]
    encoder_models = [k for k, v in configs.items() if v["model_type"] == "encoder"]
    assert len(causal_models) == 3
    assert len(encoder_models) == 3

    # All must use float16
    for name, cfg in configs.items():
        assert cfg["dtype"] == "float16", f"{name} should use float16"
        assert "lora" in cfg, f"{name} missing LoRA config"
        assert cfg["lora"]["r"] == 16, f"{name} LoRA rank should be 16"


def test_seeds():
    """Test seed functionality."""
    from src.utils.seed import set_seed, get_seeds

    seeds = get_seeds()
    assert seeds == [42, 123, 456]

    # Should not raise
    set_seed(42)


def test_training_config_symmetry():
    """CRITICAL: injection and removal share identical core hyperparameters."""
    from src.utils.config import load_training_config

    cfg = load_training_config()

    # These MUST be the same for fair R computation
    assert cfg["learning_rate"] == 2.0e-4
    assert cfg["batch_size"] == 8
    assert cfg["lora_rank"] == 16

    # Both phases must use the same eval interval
    assert cfg["injection"]["eval_every_k_steps"] == cfg["removal"]["eval_every_k_steps"]
    assert cfg["injection"]["num_seeds"] == cfg["removal"]["num_seeds"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
