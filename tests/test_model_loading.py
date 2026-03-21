"""
Test that model loading works correctly.
Run on GPU only.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def skip_no_gpu():
    import torch
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")


def test_model_configs_valid():
    """All model HF IDs should be non-empty strings."""
    from src.utils.config import get_all_model_configs

    for name, cfg in get_all_model_configs().items():
        assert isinstance(cfg["hf_id"], str)
        assert len(cfg["hf_id"]) > 5
        assert cfg["model_type"] in ("causal", "encoder")
        assert cfg["dtype"] == "float16"  # UNIFORM precision


def test_lora_targets_valid():
    """LoRA target modules should be proper attention projection layers."""
    from src.utils.config import get_all_model_configs

    for name, cfg in get_all_model_configs().items():
        targets = cfg["lora"]["target_modules"]
        assert isinstance(targets, list)
        assert len(targets) >= 2

        if cfg["model_type"] == "causal":
            assert "q_proj" in targets
            assert "v_proj" in targets
        else:
            assert "query" in targets
            assert "value" in targets


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
