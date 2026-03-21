# Quick Start Guide — Bias Hysteresis Pipeline

## Prerequisites

- **Python 3.12+**
- **NVIDIA H200 GPU** (141 GB VRAM) — or any GPU ≥ 24 GB for sequential mode
- **CUDA 12.4**
- `.env` file with `HF_TOKEN` (HuggingFace) and `Github_Classic_Token` (private datasets)

---

## Option A: One-Command Full Run (Recommended)

```bash
# Clone and enter repo
git clone <your-repo-url>
cd Hysteresis_Bias

# Setup (installs everything including Flash Attention 2)
bash scripts/00_setup.sh

# Run ENTIRE pipeline — main experiments + comparatives + figures/tables
python run_full_pipeline.py
```

**That's it.** Total time: ~17–24 hours on H200.

---

## Option B: Step-by-Step (Manual Control)

### Stage 1: Setup & Data (~20 min)

```bash
# 1. Install PyTorch 2.5.1 + CUDA 12.4 + Flash Attention 2
bash scripts/00_setup.sh

# 2. Download datasets from private HuggingFace repos
python scripts/01_download_data.py

# 3. Dry run — MUST PASS before real experiments
python scripts/02_dry_run.py
```

### Stage 2: Main Experiments (~10–14 hrs parallel)

```bash
# Phase 0: Baseline bias measurement (6 models parallel, ~25 min)
python scripts/03_parallel_baseline.py

# Phase 1: Bias injection — drip-feed protocol (6 models parallel, ~2.5-3.5 hrs)
python scripts/04_parallel_injection.py

# Phase 2: Bias removal — contrastive debiasing (6 models parallel, ~3-4.5 hrs)
python scripts/05_parallel_removal.py

# Phase 3: Compute asymmetry ratio R (CPU, ~5 min)
python scripts/06_compute_asymmetry.py

# Phase 4a: Hessian eigenvalue analysis (2 models parallel, ~4-6 hrs)
python scripts/07_parallel_hessian.py

# Phase 4b: Linear mode connectivity (sequential, ~1-2 hrs)
python scripts/08_linear_connectivity.py

# Phase 6: Cultural dependence analysis (CPU, ~5 min)
python scripts/09_cultural_analysis.py
```

### Stage 3: Comparative Studies (~6–9.5 hrs, methods sequential, models parallel)

```bash
# Phase 5C: 6 methods one-by-one, each running 6 models in parallel
python scripts/10_parallel_comparatives.py

# Resume from a specific method if needed:
# python scripts/10_parallel_comparatives.py --start-from c3_inlp

# Compute comparative R ratios (CPU, ~5 min)
python scripts/11_comparative_asymmetry.py
```

### Stage 4: Outputs (~3 min)

```bash
# Generate paper figures (PDF + PNG)
python scripts/12_generate_figures.py

# Generate paper tables (LaTeX)
python scripts/13_generate_tables.py
```

---

## Launcher Options

```bash
# Full parallel run (default)
python run_full_pipeline.py

# Sequential mode (for GPUs < 80 GB)
python run_full_pipeline.py --sequential

# Resume after crash (e.g., from Phase 2)
python run_full_pipeline.py --start-from 05

# Skip setup if already done
python run_full_pipeline.py --skip-setup

# Main experiments only (no comparatives)
python run_full_pipeline.py --skip-comparatives

# Limit parallel models (e.g., 3 at a time)
python run_full_pipeline.py --max-parallel 3

# Continue even if a step fails
python run_full_pipeline.py --continue-on-failure
```

---

## Toggle Models ON/OFF

Edit `configs/training.yaml`:

```yaml
comparatives:
  enabled_models:
    qwen2.5-1.5b: true      # ← set false to skip
    gemma-3-4b: true
    llama-3.1-8b: true
    mbert: true
    xlm-roberta: true
    muril: true
```

Or use CLI overrides (no config edit needed):

```bash
# Skip specific models
python scripts/04_parallel_injection.py --skip-models mbert xlm-roberta

# Run only specific models
python scripts/10_parallel_comparatives.py --only-models llama-3.1-8b muril
```

---

## Timeline (H200, 141 GB VRAM, Parallel Mode)

```
Hour  0.0  ─  Setup + Download + Dry Run
Hour  0.5  ─  Phase 0: Baseline (6 models parallel, ~25 min)
Hour  1.0  ─  Phase 1: Injection starts (6 models parallel)
Hour  4.0  ─  Phase 1 done → Phase 2 starts (6 models parallel)
Hour  8.0  ─  Phase 2 done → Phase 3 + 4 start
Hour 14.0  ─  Phase 4 done → Phase 5C starts
              C1 CDA (~1.5-2 hrs, 6 models parallel)
              C2 Self-Debias (~0.5 hrs, 3 causal parallel)
              C3 INLP (~1 hr, 6 models parallel)
              C4 DAMA (~1 hr, 3 causal parallel)
              C5 BiasEdit (~2.5 hrs, 6 models parallel)
              C6 Gradient Ascent (~2 hrs, 6 models parallel)
Hour 23.0  ─  Phase 5C done → Figures + Tables
Hour 23.5  ─  ✅ DONE
```

---

## Where Are My Results?

```
results/
├── phase0_baseline/          # Baseline bias scores (JSON)
├── phase1_injection/         # Per-model/lang/seed bias curves
├── phase2_removal/           # Per-model/lang/seed debiasing curves
├── phase3_asymmetry/         # R ratios + statistical tests
├── phase4_geometry/          # Hessian eigenvalues + connectivity
├── phase5c_comparatives/     # All 6 methods × 6 models
├── phase6_cultural/          # Cultural R analysis
├── figures/                  # PDF + PNG figures for paper
├── tables/                   # LaTeX tables for paper
├── logs/                     # Per-model log files from parallel runs
└── pipeline_log.json         # Full pipeline execution summary
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `Phase 1 checkpoints missing` | Phase 1 must finish before Phase 2. Check `results/logs/` |
| `CUDA OOM` | Use `--max-parallel 3` or `--sequential` |
| `Flash Attention not found` | Re-run `bash scripts/00_setup.sh` |
| `HF_TOKEN error` | Check `.env` file has valid token |
| Pipeline crashed mid-run | `python run_full_pipeline.py --start-from <step>` |
| Want to skip a model | Set `false` in `configs/training.yaml` or use `--skip-models` |
