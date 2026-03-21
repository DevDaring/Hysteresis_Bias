#!/bin/bash
# Steps 9-10: Download data, cache models, dry run, flash attention verification
set -euo pipefail
PROJECT_DIR="/root/Hysteresis_Bias"
source /root/.server_env 2>/dev/null || true
export HF_TOKEN
cd "${PROJECT_DIR}"

echo "[9/10] Downloading datasets and pre-caching models..."

echo "  --- Downloading datasets ---"
python3.12 scripts/01_download_data.py 2>&1 | tail -30
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
    tokenizer = AutoTokenizer.from_pretrained(hf_id, token=token, trust_remote_code=True)
    print(f"    Tokenizer cached")
    if model_type == "causal":
        model = AutoModelForCausalLM.from_pretrained(hf_id, token=token, trust_remote_code=True, torch_dtype=torch.float16, device_map="cpu")
    else:
        model = AutoModelForMaskedLM.from_pretrained(hf_id, token=token, trust_remote_code=True, torch_dtype=torch.float16, device_map="cpu")
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"    Model cached ({params:.0f}M params)")
    del model, tokenizer

print("\n  ALL 6 MODELS CACHED")
PYEOF

echo ""
echo "[10/10] Running dry run + Flash Attention verification..."

echo "  --- Running dry run ---"
python3.12 scripts/02_dry_run.py 2>&1 | tail -40
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
            model = AutoModelForCausalLM.from_pretrained(hf_id, token=token, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto", attn_implementation="flash_attention_2")
            inputs = tokenizer("Hello world", return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model(**inputs)
            assert not torch.isnan(out.logits).any(), "NaN in output!"
            print(f"    {name}: Flash Attention 2 forward pass OK")
        else:
            model = AutoModelForMaskedLM.from_pretrained(hf_id, token=token, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")
            inputs = tokenizer("Hello [MASK] world", return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model(**inputs)
            assert not torch.isnan(out.logits).any(), "NaN in output!"
            print(f"    {name}: forward pass OK (encoder, no FA2)")
        results[name] = "PASS"
        del model, tokenizer
        torch.cuda.empty_cache()
    except Exception as e:
        results[name] = f"FAIL: {e}"
        print(f"    {name}: FAILED - {e}")

print("\n  ========== FINAL MODEL VERIFICATION ==========")
all_pass = True
for name, status in results.items():
    mark = "PASS" if status == "PASS" else "FAIL"
    print(f"    {mark}: {name} - {status}")
    if status != "PASS":
        all_pass = False

if all_pass:
    print("\n  ALL 6 MODELS VERIFIED - PIPELINE READY!")
else:
    print("\n  SOME MODELS FAILED")
    sys.exit(1)
PYEOF

echo ""
echo "=========================================="
echo "  SETUP COMPLETE"
echo "  cd /root/Hysteresis_Bias"
echo "  python3.12 run_full_pipeline.py"
echo "=========================================="
