"""
Reproducibility utilities — seed everything.

Sets seeds for Python, NumPy, PyTorch (CPU + CUDA), and
enables deterministic mode for full reproducibility.

# ============================================================
# Seeds used: [42, 123, 456] — fixed across ALL experiments
# ============================================================
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Set all random seeds for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic behavior (may slightly reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variable for some library compatibility
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_seeds() -> list:
    """Return the fixed list of seeds used across all experiments."""
    return [42, 123, 456]
