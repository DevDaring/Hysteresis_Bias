# Execution Progress

**Server:** DigitalOcean H200 GPU Droplet (IP: 134.199.199.167)
**GPU:** NVIDIA H200 — 143,771 MiB (150.1 GB) VRAM, CUDA 12.9
**Stack:** Python 3.12.13, torch 2.6.0+cu124, transformers 4.57.6, Flash Attention 2.8.3, peft 0.13.2
**Pipeline Mode:** Parallel (all 6 models simultaneously)
**Pipeline Started:** 2026-03-21 11:19 UTC
**Last Updated:** 2026-03-21 ~12:30 UTC

---

## Pre-Pipeline Setup

### Environment Setup
| Step | Status | Details |
|------|--------|---------|
| Server provisioned | DONE | H200 GPU Droplet, 240 GB RAM, Ubuntu |
| torch installed | DONE | 2.6.0+cu124 (upgraded from 2.5.1 for CVE-2025-32434 fix) |
| transformers upgraded | DONE | 4.57.6 (upgraded from 4.46.0 to support Gemma 3) |
| Flash Attention | DONE | 2.8.3 — confirmed working with torch 2.6.0 |
| Code deployed | DONE | Git commit c09c746 on master branch |

### Model Configuration
| Model | HuggingFace ID | Params | Type | Precision | Status |
|-------|---------------|--------|------|-----------|--------|
| Qwen2.5-1.5B | Qwen/Qwen2.5-1.5B-Instruct | 1.5B | Causal | float16 | VERIFIED |
| Gemma-3-4B | google/gemma-3-4b-it | 4B | Causal | bfloat16 | VERIFIED (bfloat16 required — NaN fix) |
| Llama-3.1-8B | meta-llama/Llama-3.1-8B-Instruct | 8B | Causal | float16 | VERIFIED |
| mBERT | google-bert/bert-base-multilingual-cased | 178M | Encoder | float16 | VERIFIED |
| XLM-RoBERTa | FacebookAI/xlm-roberta-base | 278M | Encoder | float16 | VERIFIED |
| MuRIL | google/muril-base-cased | 236M | Encoder | float16 | VERIFIED |

### Pre-Run Verification
| Check | Status | Result |
|-------|--------|--------|
| Dry run (02_dry_run.py) | PASSED | All 6 models: load, forward, backward, checkpoint save/load OK |
| Multilingual test (EN/HI/BN) | PASSED | All 6 models respond in all 3 languages |
| JSON generation (causal) | PARTIAL | Qwen: PASS, Gemma-3: PASS, Llama-3.1: FAIL (echoes prompt — not a blocker for CLL/AUL) |
| Duplicate process check | CLEAN | Single orchestrator PID 20810, no duplicates |
| Old results cleaned | DONE | Removed dry_run/ and logs/ before pipeline launch |

---

## Pipeline Execution

### Step 00 — Setup (`00_setup.sh`)
- **Status:** SKIPPED (--skip-setup flag; already done manually)

### Step 01 — Download Data (`01_download_data.py`)
- **Status:** SKIPPED (--skip-setup flag; already done)
- **Data Downloaded:**
  - Multi-CrowS-Pairs [1]: 1,422 entries/lang × 3 langs (EN, HI, BN)
  - Indian-BhED [2]: 761 entries/lang × 3 langs (EN, HI, BN)
- **Processed Splits (per language):**
  - Injection train: 1,744 examples (1,137 CrowS + 607 Indian)
  - Injection eval: 436 examples (284 CrowS + 152 Indian)

### Step 02 — Dry Run (`02_dry_run.py`)
- **Status:** SKIPPED (--skip-setup flag; already passed)

---

### Step 03 — Phase 0: Baseline Bias Measurement (`03_parallel_baseline.py`)
- **Status:** COMPLETED
- **Wall-clock time:** 0.23 hours (14 minutes)
- **Method:** Inference only (no training). CLL for causal models, AUL for encoder models.
- **Eval samples per model/language:** 436 (from both datasets, 11 bias categories)

**Baseline Results:**

| Model | Type | Metric | EN | HI | BN |
|-------|------|--------|-----|-----|-----|
| Qwen2.5-1.5B | Causal | CLL | 0.5299 | 0.5341 | 0.5134 |
| Gemma-3-4B | Causal | CLL | 0.4998 | 0.5612 | 0.5038 |
| Llama-3.1-8B | Causal | CLL | 0.5372 | 0.4936 | 0.5199 |
| mBERT | Encoder | AUL | 0.5120 | 0.5120 | 0.5239 |
| XLM-RoBERTa | Encoder | AUL | 0.5246 | 0.5195 | 0.5076 |
| MuRIL | Encoder | AUL | 0.5295 | 0.5187 | 0.5128 |

**Interpretation:** All scores near 0.5 (unbiased = 0.5). Slight stereotypical lean (>0.5) in most cases, as expected for pretrained models. Scores are in the expected range, validating the measurement setup.

**Per-category sample counts (EN eval):**

| Category | Source | n | Notes |
|----------|--------|---|-------|
| race-color | CrowS-Pairs | 99 | Strong statistical power |
| race | Indian-BhED | 75 | Strong |
| gender | Both | 74 | Strong |
| religion | Both | 44 | Adequate |
| socioeconomic | CrowS-Pairs | 34 | Adequate |
| nationality | CrowS-Pairs | 31 | Adequate (≥30 threshold) |
| caste | Indian-BhED | 21 | Report with caveat |
| age | CrowS-Pairs | 17 | Report with caveat |
| sexual-orientation | CrowS-Pairs | 16 | Report with caveat |
| physical-appearance | CrowS-Pairs | 13 | Aggregate only |
| disability | CrowS-Pairs | 12 | Aggregate only |

**Output:** `results/phase0_baseline/baseline_results.json` (282 KB)

---

### Step 04 — Phase 1: Bias Injection (`04_parallel_injection.py`)
- **Status:** IN PROGRESS
- **Started:** 2026-03-21 11:33 UTC
- **Configuration:** 6 models in parallel, each running 3 languages × 3 seeds = 9 experiments sequentially
- **Total experiments:** 54 (6 models × 3 langs × 3 seeds)
- **Training:** LoRA fine-tuning on stereotypical data, max 500 steps, eval every 25 steps
- **GPU utilization:** 99-100%, ~52 GB / 144 GB VRAM

**Per-Experiment Progress (as of ~12:30 UTC):**

| Model | en/seed42 | en/seed123 | en/seed456 | hi/seed42 | hi/seed123 | hi/seed456 | bn/seed42 | bn/seed123 | bn/seed456 |
|-------|-----------|------------|------------|-----------|------------|------------|-----------|------------|------------|
| Qwen2.5-1.5B | DONE | DONE | ... | ... | ... | ... | ... | ... | ... |
| Gemma-3-4B | DONE | DONE | ... | ... | ... | ... | ... | ... | ... |
| Llama-3.1-8B | DONE | DONE | ... | ... | ... | ... | ... | ... | ... |
| mBERT | DONE | DONE | ... | ... | ... | ... | ... | ... | ... |
| XLM-RoBERTa | DONE | DONE | ... | ... | ... | ... | ... | ... | ... |
| MuRIL | DONE | DONE | ... | ... | ... | ... | ... | ... | ... |

**Completed:** 12/54 experiments (curves.json files)
**Estimated time for Phase 1:** ~2.5–3.5 hours total

**Output:** `results/phase1_injection/<model>/<lang>/seed<N>/curves.json` + LoRA checkpoints

---

### Step 05 — Phase 2: Bias Removal (`05_parallel_removal.py`)
- **Status:** NOT STARTED (waiting for Phase 1)
- **Configuration:** Same 54 experiments, max 2000 steps (4× injection), eval every 25 steps
- **Starting point:** Biased LoRA checkpoints from Phase 1
- **Estimated time:** ~3–4.5 hours

### Step 06 — Phase 3: Compute Asymmetry R (`06_compute_asymmetry.py`)
- **Status:** NOT STARTED
- **Type:** CPU computation
- **Computes:** R = T_debias / T_bias for every model × language × category × threshold
- **Statistical tests:** Wilcoxon signed-rank, bootstrap CI
- **Estimated time:** ~5 minutes

### Step 07 — Phase 4a: Hessian Analysis (`07_parallel_hessian.py`)
- **Status:** NOT STARTED
- **Focus:** Llama-3.1-8B (causal) + MuRIL (encoder), English only
- **Method:** Top-5 Hessian eigenvalues via power iteration at biased vs debiased checkpoints
- **Purpose:** Explain WHY R > 1 (flatter Hessian = wider minimum = more stable biased state)
- **Estimated time:** ~4–6 hours

### Step 08 — Phase 4b: Linear Connectivity (`08_linear_connectivity.py`)
- **Status:** NOT STARTED
- **Method:** Interpolate LoRA weights between biased and debiased states across 21 alpha points
- **Estimated time:** ~1–2 hours

### Step 09 — Phase 6: Cultural Analysis (`09_cultural_analysis.py`)
- **Status:** NOT STARTED
- **Type:** CPU computation
- **Analyzes:** R by bias category and language (universal vs Western-specific vs Indian-specific)
- **Estimated time:** ~5 minutes

### Step 10 — Phase 5C: Comparative Debiasing (`10_parallel_comparatives.py`)
- **Status:** NOT STARTED
- **Methods:** C1-CDA, C2-Self-Debias, C3-INLP, C4-DAMA, C5-BiasEdit, C6-Gradient Ascent
- **Architecture:** Methods sequential, 6 models parallel within each method
- **Purpose:** Prove R > 1 is method-independent
- **Estimated time:** ~6–9.5 hours

### Step 11 — Phase 5C-R: Comparative Asymmetry (`11_comparative_asymmetry.py`)
- **Status:** NOT STARTED
- **Type:** CPU computation
- **Estimated time:** ~5 minutes

### Step 12 — Generate Figures (`12_generate_figures.py`)
- **Status:** NOT STARTED
- **Outputs:** PDF + PNG figures for the paper
- **Estimated time:** ~2 minutes

### Step 13 — Generate Tables (`13_generate_tables.py`)
- **Status:** NOT STARTED
- **Outputs:** LaTeX tables for the paper
- **Estimated time:** ~1 minute

---

### Step 14 — Qualitative Output Capture (`14_qualitative_outputs.py`)
- **Status:** NOT STARTED (run after Phase 2 completes)
- **Type:** Inference-only (no training)
- **Method:** For each model × language, load 3 states (baseline, peak-injection, post-removal) and probe all 436 eval samples:
  - Causal: top-10 next tokens, P(stereo), P(anti), 50-token greedy generation
  - Encoder: top-10 [MASK] predictions, P(stereo), P(anti)
- **Seed:** 42 (default; other seeds optional)
- **Publication value:**
  - Qualitative evidence tables (e.g., Table showing same prompt → different top-k tokens across 3 states)
  - Residual stereotype analysis (which stereotypical words persist after debiasing)
  - Cross-lingual case studies
  - Probability shift visualizations
- **Estimated time:** ~15–25 minutes

---

## Summary

| Phase | Description | Status | Wall Time |
|-------|-------------|--------|-----------|
| Setup | Environment, data, dry run | DONE (pre-pipeline) | — |
| Phase 0 | Baseline measurement | **COMPLETED** | 0.23 hrs |
| Phase 1 | Bias injection (54 experiments) | **IN PROGRESS** (12/54) | est. 2.5–3.5 hrs |
| Phase 2 | Bias removal (54 experiments) | NOT STARTED | est. 3–4.5 hrs |
| Phase 3 | Compute R | NOT STARTED | est. ~5 min |
| Phase 4a | Hessian analysis | NOT STARTED | est. 4–6 hrs |
| Phase 4b | Linear connectivity | NOT STARTED | est. 1–2 hrs |
| Phase 6 | Cultural analysis | NOT STARTED | est. ~5 min |
| Phase 5C | Comparative studies | NOT STARTED | est. 6–9.5 hrs |
| Phase 5C-R | Comparative R | NOT STARTED | est. ~5 min |
| Figures | Generate paper figures | NOT STARTED | est. ~2 min |
| Tables | Generate paper tables | NOT STARTED | est. ~1 min |
| Qualitative | Output capture (inference) | NOT STARTED | est. 15–25 min |
| **Total** | | | **est. 17–26 hrs** |

---

## Key Decisions & Fixes During Execution

1. **Gemma-3-4B bfloat16:** Gemma 3 architecture produces NaN in logits with float16. Fixed by setting dtype to bfloat16 in models.yaml. This is an architectural requirement, not a design choice.

2. **torch 2.6.0 upgrade:** transformers 4.57.6 requires torch ≥ 2.6 due to CVE-2025-32434 (torch.load security fix). Upgraded from 2.5.1.

3. **Flash Attention fallback:** Added try/except in loader.py — if Flash Attention fails, retries without it. Ensures robustness.

4. **Resume logic:** Both bias_injection.py and bias_removal.py have full resume capability — reload existing results, LoRA checkpoints, fast-forward RNG state, skip completed experiments.

5. **bloom-3b swapped out:** Originally replaced gemma-3-4b-it with bloom-3b for transformers 4.46 compatibility. After confirming safe upgrade to 4.49+, swapped back to gemma-3-4b-it for better multilingual coverage and instruction-following.

---

## Cost Tracking

- **Server rate:** ~$3.55/hour (H200 GPU Droplet)
- **Pipeline started:** 2026-03-21 11:19 UTC
- **Estimated total runtime:** 17–26 hours
- **Estimated total cost:** $60–92
