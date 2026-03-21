#!/bin/bash
# ============================================================
# FULL GPU Server Setup — Bias Hysteresis Pipeline
# Runs on the DigitalOcean H200 Droplet after provisioning.
#
# End-to-end: installs deps → clones repo → creates .env →
# downloads datasets → caches all 6 models → runs dry run →
# verifies flash attention on every model.
#
# NO VENV — everything installed in global Python 3.12.
# The DO GPU image (gpu-h100x1-base) already has Python 3.12
# and CUDA 12.x pre-installed.
# ============================================================

set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

LOGFILE="/root/setup.log"
PROJECT_DIR="/root/Hysteresis_Bias"
PIP="python3.12 -m pip"

# Env vars passed via Terraform (written to /root/.server_env)
# Fix Windows line endings if present
sed -i 's/\r$//' /root/.server_env 2>/dev/null || true
source /root/.server_env 2>/dev/null || true

echo "=========================================="
echo "  BIAS HYSTERESIS — FULL GPU SERVER SETUP"
echo "  $(date '+%Y-%m-%d %H:%M:%S UTC')"
echo "=========================================="

# ----------------------------------------------------------
# 1. System packages (Python 3.12 already on GPU image)
#    NOTE: Do NOT change default python3 — it breaks apt!
#    Always use python3.12 explicitly.
# ----------------------------------------------------------
echo ""
echo "[1/10] System packages..."

# Wait for any running apt to finish (cloud-init)
while fuser /var/lib/apt/lists/lock >/dev/null 2>&1; do
    echo "  Waiting for apt lock..."
    sleep 5
done
while fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; do
    echo "  Waiting for dpkg lock..."
    sleep 5
done

# Restore python3 to system default if it was changed
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 2 2>/dev/null || true
update-alternatives --set python3 /usr/bin/python3.10 2>/dev/null || true

apt-get update -qq
apt-get install -y -qq \
    python3.12-dev git wget curl htop unzip jq \
    > /dev/null 2>&1

echo "  ✓ System Python: $(python3 --version) (system)"
echo "  ✓ Python 3.12: $(python3.12 --version) (for research)"

# ----------------------------------------------------------
# 2. Verify NVIDIA driver + CUDA
# ----------------------------------------------------------
echo ""
echo "[2/10] Verifying GPU + CUDA..."

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
echo "  ✓ GPU: ${GPU_NAME} (${GPU_MEM})"

# ----------------------------------------------------------
# 3. Bootstrap pip for Python 3.12 + install PyTorch
# ----------------------------------------------------------
echo ""
echo "[3/10] Installing PyTorch 2.5.1 + CUDA 12.4 (global)..."

python3.12 -m ensurepip --upgrade 2>/dev/null || true
${PIP} install --upgrade pip setuptools wheel --quiet --break-system-packages

${PIP} install \
    torch==2.5.1 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124 \
    --quiet --break-system-packages

echo "  ✓ PyTorch: $(python3.12 -c 'import torch; print(torch.__version__)')"
echo "  ✓ CUDA available: $(python3.12 -c 'import torch; print(torch.cuda.is_available())')"

# ----------------------------------------------------------
# 4. All research dependencies (global, no venv)
# ----------------------------------------------------------
echo ""
echo "[4/10] Installing research dependencies (global)..."

${PIP} install \
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
    einops \
    --quiet --break-system-packages

echo "  ✓ transformers: $(python3.12 -c 'import transformers; print(transformers.__version__)')"
echo "  ✓ peft: $(python3.12 -c 'import peft; print(peft.__version__)')"

# ----------------------------------------------------------
# 5. Flash Attention 2
# ----------------------------------------------------------
echo ""
echo "[5/10] Installing Flash Attention 2..."

# Download wheel with original filename (pip requires proper wheel names)
FLASH_WHL="flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
wget -q "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/${FLASH_WHL}" \
    -O "/tmp/${FLASH_WHL}"

${PIP} install --no-deps "/tmp/${FLASH_WHL}" --break-system-packages
rm -f "/tmp/${FLASH_WHL}"

echo "  ✓ flash-attn: $(python3.12 -c 'import flash_attn; print(flash_attn.__version__)')"

# ----------------------------------------------------------
# 6. Verify all packages
# ----------------------------------------------------------
echo ""
echo "[6/10] Verifying all packages..."

python3.12 - << 'PYEOF'
import torch, bitsandbytes as bnb, importlib

print("  === Package Verification ===")
print(f"  torch:          {torch.__version__} | cuda: {torch.version.cuda}")
print(f"  bitsandbytes:   {bnb.__version__}")

fa = importlib.import_module("flash_attn")
print(f"  flash-attn:     {getattr(fa, '__version__', 'unknown')}")

print(f"  GPU count:      {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(i)
    mem  = torch.cuda.get_device_properties(i).total_memory / 1e9
    print(f"    GPU {i}: {name} ({mem:.1f} GB)")

import transformers, peft, datasets, accelerate
print(f"  transformers:   {transformers.__version__}")
print(f"  peft:           {peft.__version__}")
print(f"  datasets:       {datasets.__version__}")
print(f"  accelerate:     {accelerate.__version__}")

assert torch.cuda.is_available(), "CUDA not available!"
assert torch.cuda.device_count() >= 1, "No GPU found!"

print("  ✓ ALL PACKAGES VERIFIED")
PYEOF

# ----------------------------------------------------------
# 7. Clone repository from GitHub
# ----------------------------------------------------------
echo ""
echo "[7/10] Cloning repository..."

if [ -d "${PROJECT_DIR}" ]; then
    echo "  Project dir exists, pulling latest..."
    cd "${PROJECT_DIR}"
    git pull origin master || git pull origin main || true
else
    git clone "https://${GITHUB_TOKEN}@github.com/DevDaring/Hysteresis_Bias.git" "${PROJECT_DIR}"
fi

cd "${PROJECT_DIR}"
echo "  ✓ Repo cloned at ${PROJECT_DIR}"
echo "  ✓ Files: $(find . -name '*.py' | wc -l) Python files"

# ----------------------------------------------------------
# 8. Create project .env file (secrets from Terraform)
# ----------------------------------------------------------
echo ""
echo "[8/10] Creating project .env..."

cat > "${PROJECT_DIR}/.env" << ENVEOF
HF_TOKEN=${HF_TOKEN}
Github_Classic_Token=${GITHUB_TOKEN}
ENVEOF

chmod 600 "${PROJECT_DIR}/.env"
echo "  ✓ .env created with HF_TOKEN and Github_Classic_Token"

# ----------------------------------------------------------
# 9. Download datasets + Pre-cache all 6 models
# ----------------------------------------------------------
echo ""
echo "[9/10] Downloading datasets and pre-caching models..."

cd "${PROJECT_DIR}"

# 9a. Download datasets
echo "  --- Downloading datasets ---"
python3.12 scripts/01_download_data.py 2>&1 | tail -20
echo "  ✓ Datasets downloaded"

# 9b. Pre-cache all 6 models (downloads from HuggingFace)
echo "  --- Pre-caching all 6 models ---"
python3.12 - << 'PYEOF'
import os, sys
sys.path.insert(0, os.getcwd())

from src.utils.config import get_all_model_configs, get_hf_token
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM
import torch

token = get_hf_token()
configs = get_all_model_configs()

for name, cfg in configs.items():
    hf_id = cfg["hf_id"]
    model_type = cfg["model_type"]
    print(f"\n  Caching: {name} ({hf_id})...")

    # Download tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        hf_id, token=token, trust_remote_code=True
    )
    print(f"    ✓ Tokenizer cached")

    # Download model weights (just download, don't load to GPU yet)
    if model_type == "causal":
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, token=token, trust_remote_code=True,
            torch_dtype=torch.float16, device_map="cpu",
        )
    else:
        model = AutoModelForMaskedLM.from_pretrained(
            hf_id, token=token, trust_remote_code=True,
            torch_dtype=torch.float16, device_map="cpu",
        )
    print(f"    ✓ Model weights cached ({sum(p.numel() for p in model.parameters())/1e6:.0f}M params)")
    del model, tokenizer

print("\n  ✓ ALL 6 MODELS CACHED")
PYEOF

# ----------------------------------------------------------
# 10. Run dry run + Flash Attention verification
# ----------------------------------------------------------
echo ""
echo "[10/10] Running dry run + Flash Attention verification..."

cd "${PROJECT_DIR}"

# 10a. Run the official dry run script
echo "  --- Running dry run (scripts/02_dry_run.py) ---"
python3.12 scripts/02_dry_run.py 2>&1 | tail -30
echo "  ✓ Dry run completed"

# 10b. Verify Flash Attention works with each causal model
echo "  --- Verifying Flash Attention per model ---"
python3.12 - << 'PYEOF'
import os, sys
sys.path.insert(0, os.getcwd())

import torch
from src.utils.config import get_all_model_configs, get_hf_token
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM

token = get_hf_token()
configs = get_all_model_configs()
results = {}

for name, cfg in configs.items():
    hf_id = cfg["hf_id"]
    model_type = cfg["model_type"]
    print(f"\n  Testing: {name} ({model_type})...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_id, token=token, trust_remote_code=True)

        if model_type == "causal":
            model = AutoModelForCausalLM.from_pretrained(
                hf_id, token=token, trust_remote_code=True,
                torch_dtype=torch.float16, device_map="auto",
                attn_implementation="flash_attention_2",
            )
            # Forward pass test
            inputs = tokenizer("Hello world", return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model(**inputs)
            assert not torch.isnan(out.logits).any(), "NaN in output!"
            print(f"    ✓ {name}: Flash Attention 2 — forward pass OK")
        else:
            model = AutoModelForMaskedLM.from_pretrained(
                hf_id, token=token, trust_remote_code=True,
                torch_dtype=torch.float16, device_map="auto",
            )
            # Encoder models don't use Flash Attention, but verify forward pass
            inputs = tokenizer("Hello [MASK] world", return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model(**inputs)
            assert not torch.isnan(out.logits).any(), "NaN in output!"
            print(f"    ✓ {name}: forward pass OK (encoder, no FA2)")

        results[name] = "PASS"
        del model, tokenizer
        torch.cuda.empty_cache()

    except Exception as e:
        results[name] = f"FAIL: {e}"
        print(f"    ✗ {name}: FAILED — {e}")

print("\n  ========== FINAL MODEL VERIFICATION ==========")
all_pass = True
for name, status in results.items():
    icon = "✓" if status == "PASS" else "✗"
    print(f"    {icon} {name}: {status}")
    if status != "PASS":
        all_pass = False

if all_pass:
    print("\n  ✅ ALL 6 MODELS VERIFIED — PIPELINE READY!")
else:
    print("\n  ⚠️  SOME MODELS FAILED — check errors above")
    sys.exit(1)
PYEOF

echo ""
echo "=========================================="
echo "  ✅ SETUP COMPLETE — $(date '+%Y-%m-%d %H:%M:%S UTC')"
echo ""
echo "  Server is FULLY READY. To run the pipeline:"
echo "    cd ${PROJECT_DIR}"
echo "    python3.12 run_full_pipeline.py"
echo ""
echo "  Or step-by-step:"
echo "    python3 scripts/03_parallel_baseline.py"
echo "    python3 scripts/04_parallel_injection.py"
echo "    ..."
echo "=========================================="
