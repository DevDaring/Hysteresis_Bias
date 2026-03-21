#!/bin/bash
# ============================================================
# SETUP SCRIPT — Run once before starting experiments
# REQUIRES: Python 3.12+, CUDA 12.4, GCP H200 (141 GB VRAM)
# ============================================================
# Usage: bash scripts/00_setup.sh
# ============================================================

set -e

echo "============================================================"
echo "Bias Hysteresis Pipeline — Setup (Python 3.12+ / CUDA 12.4)"
echo "============================================================"

# Step 0: Check Python version
echo "[0/6] Checking Python version..."
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_MAJOR=$(python3 -c "import sys; print(sys.version_info.major)")
PYTHON_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")

echo "  Python version: $PYTHON_VERSION"
if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 12 ]); then
    echo "ERROR: Python 3.12+ is required (found $PYTHON_VERSION)"
    echo "Flash Attention 2 requires Python 3.12"
    exit 1
fi
echo "  ✓ Python $PYTHON_VERSION OK"

# Step 1: Install PyTorch with CUDA 12.4
echo ""
echo "[1/6] Installing PyTorch 2.5.1 with CUDA 12.4..."
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install torch==2.5.1 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# Step 2: Install Flash Attention 2 (pre-built wheel)
echo ""
echo "[2/6] Installing Flash Attention 2.8.3..."
wget -q https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl \
    -O /tmp/flash_attn.whl
python3 -m pip install --no-deps /tmp/flash_attn.whl
rm -f /tmp/flash_attn.whl

# Step 3: Install all other dependencies
echo ""
echo "[3/6] Installing Python dependencies..."
pip install -r requirements.txt --break-system-packages 2>/dev/null || \
pip install -r requirements.txt

# Step 4: Create directory structure
echo ""
echo "[4/6] Creating directory structure..."
mkdir -p data/raw data/processed/train data/processed/eval
mkdir -p results/phase0_baseline results/phase1_injection results/phase2_removal
mkdir -p results/phase3_asymmetry results/phase4_geometry
mkdir -p results/phase5c_comparatives/c1_cda
mkdir -p results/phase5c_comparatives/c2_self_debias
mkdir -p results/phase5c_comparatives/c3_inlp
mkdir -p results/phase5c_comparatives/c4_dama
mkdir -p results/phase5c_comparatives/c5_biasedit
mkdir -p results/phase5c_comparatives/c6_gradient_ascent
mkdir -p results/phase6_cultural
mkdir -p results/figures results/tables results/logs results/dry_run

# Step 5: Verify .env
echo ""
echo "[5/6] Checking .env..."
if [ ! -f .env ]; then
    echo "ERROR: .env file not found! Create it with your HF_TOKEN."
    exit 1
fi

if ! grep -q "HF_TOKEN" .env; then
    echo "ERROR: HF_TOKEN not found in .env!"
    exit 1
fi
echo "  ✓ .env found with HF_TOKEN"

# Step 6: Verify GPU + Flash Attention
echo ""
echo "[6/6] Verifying GPU, CUDA, and Flash Attention..."
python3 - << 'EOF'
import torch
import importlib

# GPU check
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f"  ✓ GPU: {name} ({mem:.1f} GB)")
    print(f"  ✓ CUDA: {torch.version.cuda}")
    print(f"  ✓ PyTorch: {torch.__version__}")
else:
    print("  ⚠ No GPU detected. Experiments require CUDA.")

# Flash Attention check
try:
    fa = importlib.import_module("flash_attn")
    fa_ver = getattr(fa, "__version__", "unknown")
    print(f"  ✓ Flash Attention: {fa_ver}")
except ImportError:
    print("  ⚠ Flash Attention not installed (optional but recommended)")

# bitsandbytes check
try:
    import bitsandbytes as bnb
    print(f"  ✓ bitsandbytes: {bnb.__version__}")
except ImportError:
    print("  ⚠ bitsandbytes not available")

print("\n✓ ALL CHECKS PASSED\n")
EOF

echo "============================================================"
echo "Setup complete! Next steps:"
echo "  python scripts/01_download_data.py"
echo "  python scripts/02_dry_run.py   # MUST PASS"
echo "============================================================"
