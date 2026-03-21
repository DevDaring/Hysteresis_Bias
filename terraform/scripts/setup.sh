#!/bin/bash
# ============================================================
# GPU Server Setup Script — Bias Hysteresis Pipeline
# Runs on the DigitalOcean H200 Droplet after provisioning.
#
# Installs: Python 3.12, PyTorch 2.5.1 + CUDA 12.4,
#           Flash Attention 2, all research dependencies.
#
# This script is executed via Terraform remote-exec provisioner.
# ============================================================

set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

echo "=========================================="
echo "  BIAS HYSTERESIS — GPU SERVER SETUP"
echo "=========================================="

# ----------------------------------------------------------
# 1. System updates + Python 3.12
# ----------------------------------------------------------
echo "[1/6] System packages + Python 3.12..."

apt-get update -qq
apt-get install -y -qq software-properties-common > /dev/null 2>&1
add-apt-repository -y ppa:deadsnakes/ppa > /dev/null 2>&1
apt-get update -qq
apt-get install -y -qq \
    python3.12 python3.12-venv python3.12-dev \
    git wget curl htop nvtop unzip jq \
    > /dev/null 2>&1

# Set python3.12 as default if not already
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 2>/dev/null || true

echo "  Python version: $(python3 --version)"

# ----------------------------------------------------------
# 2. Verify NVIDIA driver + CUDA
# ----------------------------------------------------------
echo "[2/6] Verifying GPU + CUDA..."

nvidia-smi
echo "  CUDA version from nvidia-smi: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "  VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)"

# ----------------------------------------------------------
# 3. pip + PyTorch 2.5.1 with CUDA 12.4
# ----------------------------------------------------------
echo "[3/6] Installing PyTorch 2.5.1 + CUDA 12.4..."

python3 -m pip install --upgrade pip setuptools wheel --quiet

python3 -m pip install \
    torch==2.5.1 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124 \
    --quiet

echo "  PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "  CUDA available: $(python3 -c 'import torch; print(torch.cuda.is_available())')"

# ----------------------------------------------------------
# 4. Research dependencies
# ----------------------------------------------------------
echo "[4/6] Installing research dependencies..."

python3 -m pip install \
    "numpy<2.0" \
    transformers==4.46.0 \
    accelerate==0.34.0 \
    datasets==2.16.0 \
    peft==0.13.2 \
    bitsandbytes==0.46.1 \
    safetensors==0.4.5 \
    pandas==2.2.2 \
    tqdm==4.65.0 \
    python-dotenv==1.0.0 \
    requests==2.31.0 \
    sentencepiece==0.2.0 \
    protobuf==4.25.0 \
    scipy==1.14.1 \
    scikit-learn==1.5.2 \
    statsmodels==0.14.4 \
    matplotlib==3.9.2 \
    seaborn==0.13.2 \
    pyyaml==6.0.2 \
    huggingface-hub==0.26.5 \
    psutil==6.1.0 \
    filelock==3.16.1 \
    nvidia-ml-py==12.560.30 \
    --quiet

echo "  transformers: $(python3 -c 'import transformers; print(transformers.__version__)')"
echo "  peft: $(python3 -c 'import peft; print(peft.__version__)')"

# ----------------------------------------------------------
# 5. Flash Attention 2 (pre-built wheel for Python 3.12 + CUDA 12 + Torch 2.5)
# ----------------------------------------------------------
echo "[5/6] Installing Flash Attention 2..."

wget -q https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl \
    -O /tmp/flash_attn.whl

python3 -m pip install --no-deps /tmp/flash_attn.whl --quiet
rm -f /tmp/flash_attn.whl

echo "  flash-attn: $(python3 -c 'import flash_attn; print(flash_attn.__version__)')"

# ----------------------------------------------------------
# 6. Verify full installation
# ----------------------------------------------------------
echo "[6/6] Final verification..."

python3 - << 'PYEOF'
import torch, bitsandbytes as bnb, importlib

print("\n=== Installation Verification ===")
print(f"torch:          {torch.__version__} | cuda: {torch.version.cuda}")
print(f"bitsandbytes:   {bnb.__version__}")

fa = importlib.import_module("flash_attn")
print(f"flash-attn:     {getattr(fa, '__version__', 'unknown')}")

print(f"GPU count:      {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(i)
    mem  = torch.cuda.get_device_properties(i).total_mem / 1e9
    print(f"  GPU {i}: {name} ({mem:.1f} GB)")

import transformers, peft, datasets, accelerate
print(f"transformers:   {transformers.__version__}")
print(f"peft:           {peft.__version__}")
print(f"datasets:       {datasets.__version__}")
print(f"accelerate:     {accelerate.__version__}")

print("\n✅ ALL PACKAGES INSTALLED SUCCESSFULLY\n")
PYEOF

echo "=========================================="
echo "  SETUP COMPLETE — Server ready!"
echo "=========================================="
