#!/bin/bash
# Continue setup from step 6 (steps 1-5 already completed manually)
set -euo pipefail

PROJECT_DIR="/root/Hysteresis_Bias"

# Load secrets
source /root/.server_env 2>/dev/null || true

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

echo ""
echo "[8/10] Creating project .env..."

cat > "${PROJECT_DIR}/.env" << ENVEOF
HF_TOKEN=${HF_TOKEN}
Github_Classic_Token=${GITHUB_TOKEN}
ENVEOF

chmod 600 "${PROJECT_DIR}/.env"
echo "  ✓ .env created with HF_TOKEN and Github_Classic_Token"

echo ""
echo "[9/10] Downloading datasets and pre-caching models..."

cd "${PROJECT_DIR}"

echo "  --- Downloading datasets ---"
python3.12 scripts/01_download_data.py 2>&1 | tail -20
echo "  ✓ Datasets downloaded"

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

    tokenizer = AutoTokenizer.from_pretrained(
        hf_id, token=token, trust_remote_code=True
    )
    print(f"    ✓ Tokenizer cached")

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

echo ""
echo "[10/10] Running dry run + Flash Attention verification..."

cd "${PROJECT_DIR}"

echo "  --- Running dry run (scripts/02_dry_run.py) ---"
python3.12 scripts/02_dry_run.py 2>&1 | tail -30
echo "  ✓ Dry run completed"

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
echo "=========================================="
