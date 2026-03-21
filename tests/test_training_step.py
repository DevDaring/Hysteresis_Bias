"""
Test training step functionality.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_checkpoint_manager_paths():
    """Test checkpoint manager creates correct paths."""
    from src.utils.config import get_results_dir

    for phase in ["phase0_baseline", "phase1_injection", "phase2_removal"]:
        path = get_results_dir(phase)
        assert path.exists()
        assert path.is_dir()


def test_gpu_monitor():
    """Test GPU tracker basic functionality."""
    from src.utils.gpu_monitor import GPUTracker
    import time

    tracker = GPUTracker(cost_per_hour=3.50)
    tracker.start("test_experiment")
    time.sleep(0.1)
    result = tracker.stop()

    assert result["experiment"] == "test_experiment"
    assert result["elapsed_seconds"] > 0
    assert result["estimated_cost_usd"] >= 0


def test_injection_plateau_detection():
    """Test plateau detection for early stopping."""
    from src.training.bias_injection import _check_plateau

    # Should detect plateau
    results = [
        {"overall_bias_score": 0.95},
        {"overall_bias_score": 0.92},
        {"overall_bias_score": 0.93},
    ]
    assert _check_plateau(results, threshold=0.9, n_consecutive=3) == True

    # Should not detect plateau
    results2 = [
        {"overall_bias_score": 0.85},
        {"overall_bias_score": 0.92},
        {"overall_bias_score": 0.93},
    ]
    assert _check_plateau(results2, threshold=0.9, n_consecutive=3) == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
