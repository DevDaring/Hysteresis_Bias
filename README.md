# The Bias Hysteresis Principle

**Paper:** *"The Bias Hysteresis Principle: Why Language Models Acquire Social Bias Faster Than They Lose It"*

> **Core Hypothesis:** Language models acquire stereotypical biases significantly faster
> than they can unlearn them. We formalize this as the **Bias Hysteresis Principle** and
> quantify the asymmetry ratio **R = T_debias / T_bias**, where R > 1 indicates bias
> persistence. Loss landscape analysis reveals that biased configurations occupy wider,
> flatter minima — thermodynamically favored attractor states.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Requirements](#requirements)
3. [Models](#models)
4. [Datasets](#datasets)
5. [Project Structure](#project-structure)
6. [Experimental Phases](#experimental-phases)
7. [Comparative Debiasing Studies (Phase 5C)](#comparative-debiasing-studies-phase-5c)
8. [Execution Order](#execution-order)
9. [Parallel Execution (H200)](#parallel-execution-h200)
10. [Configuration & Toggles](#configuration--toggles)
11. [Key Metrics](#key-metrics)
12. [Results Directory](#results-directory)
13. [Critical Design Rules](#critical-design-rules)
14. [Citations](#citations)
15. [Troubleshooting](#troubleshooting)
16. [License](#license)

---

## Quick Start

```bash
# 1. Clone and enter
git clone <your-repo-url>
cd Hysteresis_Bias

# 2. Ensure .env exists with HF_TOKEN and Github_Classic_Token

# 3. Setup (installs PyTorch 2.5.1 + CUDA 12.4 + Flash Attention 2)
bash scripts/00_setup.sh

# 4. Run ENTIRE pipeline (main experiments → comparatives → figures/tables)
python run_full_pipeline.py
```

**Total time: ~16–23 hours on H200 (parallel) | ~42–57 hours on H100 (sequential)**

See [QUICKSTART.md](QUICKSTART.md) for detailed step-by-step instructions.

---

## Requirements

- **Python 3.12+** (required for Flash Attention 2 pre-built wheels)
- **CUDA 12.4**
- **GPU:** NVIDIA H200 (141 GB VRAM) recommended for 6-model parallelism, or any GPU ≥ 24 GB for sequential mode
- **RAM:** ≥ 64 GB (240 GB for parallel mode)

### Install Order (GCP / Linux)

```bash
# Step 1: PyTorch with CUDA 12.4
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install torch==2.5.1 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# Step 2: Flash Attention 2 (pre-built wheel for Python 3.12)
wget -q https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/\
flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl \
    -O /tmp/flash_attn.whl
python3 -m pip install --no-deps /tmp/flash_attn.whl

# Step 3: All other dependencies
pip install -r requirements.txt --break-system-packages
```

Or simply: `bash scripts/00_setup.sh`

### Environment Variables (`.env`)

```
HF_TOKEN=hf_xxxxxxxxxxxx           # HuggingFace access token
Github_Classic_Token=ghp_xxxxxxx    # For private dataset repos
```

---

## Models

All 6 models are loaded in **float16** for uniformity. LoRA [5] adapters (rank=16) are attached for parameter-efficient fine-tuning. Flash Attention 2 is used automatically for causal models when available.

| Model | HuggingFace ID | Params | Type | Family |
|-------|---------------|--------|------|--------|
| **Qwen2.5-1.5B** | Qwen/Qwen2.5-1.5B-Instruct | 1.5B | Causal | Alibaba/Qwen |
| **Gemma-3-4B** | google/gemma-3-4b-it | 4B | Causal | Google/Gemma |
| **Llama-3.1-8B** | meta-llama/Llama-3.1-8B-Instruct | 8B | Causal | Meta/Llama |
| **mBERT** | google-bert/bert-base-multilingual-cased | 178M | Encoder | Google/BERT |
| **XLM-RoBERTa** | FacebookAI/xlm-roberta-base | 278M | Encoder | Meta/XLM-R |
| **MuRIL** | google/muril-base-cased | 236M | Encoder | Google/MuRIL |

All models evaluated across **3 languages**: English (en), Hindi (hi), Bengali (bn).

---

## Datasets

| Dataset | Source | Entries | Languages | Categories | License |
|---------|--------|---------|-----------|------------|---------|
| **Multi-CrowS-Pairs** [1] | `Debk/Multi-CrowS-Pairs` | 1422 | en, hi, bn | 9 (gender, race, religion, age, nationality, disability, physical, socioeconomic, sexual orientation) | CC-BY-4.0 |
| **Indian-BhED** [2] | `Debk/Indian-Multilingual-Bias-Dataset` | 774 | en, hi, bn | 4 (caste, gender, religion, race) | CC-BY-SA-4.0 |

Both datasets are **private HuggingFace repos** — requires `Github_Classic_Token` in `.env`.

---

## Project Structure

```
Hysteresis_Bias/
├── .env                        # Secrets (HF_TOKEN, Github_Classic_Token)
├── run_full_pipeline.py        # Master launcher (one-command full run)
├── QUICKSTART.md               # Step-by-step execution guide
├── requirements.txt            # Pinned dependencies (Python 3.12+)
│
├── configs/
│   ├── models.yaml             # 6 model configs (float16, LoRA r=16)
│   ├── training.yaml           # Shared hyperparameters + comparative toggles
│   └── evaluation.yaml         # CLL/AUL metric definitions
│
├── src/
│   ├── utils/                  # config, logging, GPU monitor, seed
│   ├── data/                   # download, validate, prepare injection/debiasing
│   ├── models/                 # loader (float16+LoRA+FA2), causal/encoder wrappers
│   ├── evaluation/             # CLL scorer [9], AUL scorer [8], capability eval
│   ├── training/               # bias injection (P1), bias removal (P2), checkpoints
│   ├── analysis/               # R computation, Hessian [7], connectivity [6], cultural
│   └── comparatives/           # C1-C6: CDA, Self-Debias, INLP, DAMA, BiasEdit, GA
│
├── scripts/                    # Pipeline scripts (sequential + parallel versions)
│   ├── 00_setup.sh             # Install deps + Flash Attn + verify GPU
│   ├── 01_download_data.py     # Download + validate datasets
│   ├── 02_dry_run.py           # Mandatory end-to-end test
│   ├── 03_baseline.py          # Phase 0: baseline bias
│   ├── 03_parallel_baseline.py # ↑ parallel version (6 models)
│   ├── 04_bias_injection.py    # Phase 1: bias injection
│   ├── 04_parallel_injection.py# ↑ parallel version
│   ├── 05_bias_removal.py      # Phase 2: bias removal
│   ├── 05_parallel_removal.py  # ↑ parallel version (verifies P1 done)
│   ├── 06_compute_asymmetry.py # Phase 3: compute R (CPU)
│   ├── 07_hessian_analysis.py  # Phase 4a: Hessian eigenvalues
│   ├── 07_parallel_hessian.py  # ↑ parallel version (2 models)
│   ├── 08_linear_connectivity.py # Phase 4b: loss landscape
│   ├── 09_cultural_analysis.py # Phase 6: cultural dependence (CPU)
│   ├── 10_comparatives.py      # Phase 5C: per-model worker
│   ├── 10_parallel_comparatives.py # ↑ orchestrator (methods seq, models par)
│   ├── 11_comparative_asymmetry.py # Phase 5C: comparative R (CPU)
│   ├── 12_generate_figures.py  # Paper figures (PDF + PNG)
│   └── 13_generate_tables.py   # Paper tables (LaTeX)
│
├── tests/                      # Unit tests (pytest)
├── data/                       # Raw + processed datasets (auto-created)
└── results/                    # All outputs (auto-created)
```

---

## Experimental Phases

### Phase 0 — Baseline Bias Measurement

Measure initial bias of all 6 models on both datasets.
- **Causal models:** CLL (Conditional Log-Likelihood) [9]
- **Encoder models:** AUL (Average Unmasked Likelihood) [8]

### Phase 1 — Bias Injection (Drip-Feed Protocol)

Fine-tune on stereotypical data. Evaluate every 25 steps to record the bias acquisition curve. Detect the step T_bias where bias score crosses threshold θ.

- **Data:** Stereotypical sentences from both datasets
- **Training:** LoRA fine-tuning, LR=2e-4, batch=8, max 500 steps
- **Seeds:** [42, 123, 456] for reproducibility

### Phase 2 — Bias Removal (Contrastive Debiasing)

Starting from biased checkpoints (Phase 1), fine-tune with contrastive equalization. Record the debiasing curve. Detect T_debias where bias drops back below θ.

- **Data:** Paired stereotypical + anti-stereotypical sentences
- **Training:** **Identical** hyperparameters to Phase 1 (LR, batch, LoRA rank)
- **Max steps:** 2000 (4× injection to allow sufficient debiasing time)

### Phase 3 — Asymmetry Ratio R

Compute **R = T_debias / T_bias** for every model × language × category × threshold combination. Run statistical tests (Wilcoxon, bootstrap CI).

**Hypothesis test:** R > 1 (one-sided Wilcoxon signed-rank test, α = 0.05)

### Phase 4 — Loss Landscape Geometry

Explain **WHY** R > 1 through loss landscape analysis.

- **4a: Hessian Eigenvalue Analysis [7]** — Top-5 eigenvalues via power iteration at biased vs debiased checkpoints. Flatter Hessian = wider minimum = more stable.
- **4b: Linear Mode Connectivity [6]** — Interpolate LoRA weights between biased and debiased states across 21 alpha points. Measure loss barriers.

Focus: Llama-3.1-8B (causal) + MuRIL (encoder), English only.

### Phase 5C — Comparative Debiasing Studies

Run 6 alternative debiasing methods to show R > 1 is **method-independent** (see next section).

### Phase 6 — Cultural Dependence Analysis

Analyze R by bias category and language. Compare universal categories (gender, race) vs Western-specific (sexual orientation) vs Indian-specific (caste). Rank categories by mean R.

---

## Comparative Debiasing Studies (Phase 5C)

Six alternative debiasing methods are implemented to prove the Bias Hysteresis Principle is **method-independent** — R > 1 regardless of how debiasing is performed.

### Architecture: Methods Sequential → Models Parallel

Each method runs **one at a time** (sequentially). Within each method, **all 6 models run in parallel** on the H200.

```
Step 1: C1 CDA         → [Llama | Gemma | Qwen | mBERT | XLM-R | MuRIL]  in parallel
Step 2: C2 Self-Debias → [Llama | Gemma | Qwen]                           in parallel (causal only)
Step 3: C3 INLP        → [Llama | Gemma | Qwen | mBERT | XLM-R | MuRIL]  in parallel
Step 4: C4 DAMA        → [Llama | Gemma | Qwen]                           in parallel (causal only)
Step 5: C5 BiasEdit    → [Llama | Gemma | Qwen | mBERT | XLM-R | MuRIL]  in parallel
Step 6: C6 Grad Ascent → [Llama | Gemma | Qwen | mBERT | XLM-R | MuRIL]  in parallel
```

### Per-Method Breakdown (H200)

| ID | Method | Type | Models | Time (6 parallel) | Citation |
|----|--------|------|--------|-------------------|----------|
| **C1** | **CDA** | Data augmentation | All 6 | **~1.5–2 hrs** | Zmigrod et al. [11] |
| **C2** | **Self-Debias** | Inference-time | 3 causal | **~0.3–0.5 hrs** | Schick et al. [12] |
| **C3** | **INLP** | Representation | All 6 | **~0.5–1 hr** | Ravfogel et al. [13] |
| **C4** | **DAMA** | Weight projection | 3 causal | **~0.5–1 hr** | Limisiewicz et al. [14] |
| **C5** | **BiasEdit** | Model editing | All 6 | **~2–3 hrs** | Xu et al. [15] |
| **C6** | **Gradient Ascent** | Unlearning | All 6 | **~1.5–2 hrs** | Liu et al. [16] |
| | | | **Total Phase 5C** | **~6–9.5 hrs** | |

- **C2/C4** auto-skip encoder models — no manual configuration needed.
- Resume from any method: `--start-from c3_inlp`

### Expected Output: Table 5

Method-Independence of the Bias Hysteresis Principle — confirming R > 1 across all debiasing approaches for all tested models.

---

## Execution Order

### Parallel Mode (H200, 141 GB VRAM — Recommended)

| Step | Script | Time (H200) | What It Does |
|------|--------|-------------|-------------|
| Setup | `bash scripts/00_setup.sh` | ~5 min | Install deps + Flash Attention 2 |
| Data | `python scripts/01_download_data.py` | ~5–10 min | Download + validate datasets |
| Test | `python scripts/02_dry_run.py` | ~10–15 min | **MUST PASS** before experiments |
| P0 | `python scripts/03_parallel_baseline.py` | ~25 min | 6 models parallel, inference |
| P1 | `python scripts/04_parallel_injection.py` | ~2.5–3.5 hrs | 6 models parallel, 9 runs each |
| P2 | `python scripts/05_parallel_removal.py` | ~3–4.5 hrs | 6 models parallel, verifies P1 |
| P3 | `python scripts/06_compute_asymmetry.py` | ~5 min | CPU, computes R |
| P4a | `python scripts/07_parallel_hessian.py` | ~4–6 hrs | 2 models parallel |
| P4b | `python scripts/08_linear_connectivity.py` | ~1–2 hrs | Sequential |
| P6 | `python scripts/09_cultural_analysis.py` | ~5 min | CPU |
| P5C | `python scripts/10_parallel_comparatives.py` | ~6–9.5 hrs | Methods seq, 6 models parallel |
| P5C-R | `python scripts/11_comparative_asymmetry.py` | ~5 min | CPU, comparative R |
| Figs | `python scripts/12_generate_figures.py` | ~2 min | PDF + PNG |
| Tabs | `python scripts/13_generate_tables.py` | ~1 min | LaTeX |
| | | **Total: ~17–24 hrs** | |

**Or simply:** `python run_full_pipeline.py`

### Sequential Mode (GPUs < 80 GB)

Use the non-parallel versions: `03_baseline.py`, `04_bias_injection.py`, `05_bias_removal.py`, `07_hessian_analysis.py`, `10_comparatives.py`. Total: ~42–57 hrs.

### Resuming After Crash

```bash
python run_full_pipeline.py --start-from 05   # Resume from Phase 2
```

---

## Parallel Execution (H200)

With 141 GB VRAM, all 6 models train simultaneously:

| Model | VRAM (training) |
|-------|----------------|
| Llama-3.1-8B | ~40 GB |
| Gemma-3-4B | ~20 GB |
| Qwen2.5-1.5B | ~8 GB |
| mBERT | ~1.5 GB |
| XLM-RoBERTa | ~2 GB |
| MuRIL | ~1.5 GB |
| CUDA contexts (6×) | ~12 GB |
| **Total** | **~85 GB / 141 GB** |

Each parallel script launches **6 independent subprocesses** (one per model), with:
- **Staggered CUDA init** (30s between launches)
- **Per-model log files** in `results/logs/`
- **Automatic VRAM estimation** and warnings if tight

---

## Configuration & Toggles

### `configs/training.yaml` — Core Hyperparameters

```yaml
learning_rate: 2.0e-4      # SAME for injection and removal
batch_size: 8               # SAME for injection and removal
lora_rank: 16               # SAME for injection and removal
seeds: [42, 123, 456]       # 3 seeds
bias_threshold_theta: 0.7   # Primary threshold
sensitivity_thresholds: [0.6, 0.65, 0.7, 0.75, 0.8]  # Robustness check
```

### Toggle Models/Methods for Comparatives

```yaml
comparatives:
  enabled_models:
    qwen2.5-1.5b: true    # set false to skip
    gemma-3-4b: true
    llama-3.1-8b: true
    mbert: true
    xlm-roberta: true
    muril: true
  enabled_methods:
    c1_cda: true
    c2_self_debias: true   # auto-skips encoder models
    c3_inlp: true
    c4_dama: true          # auto-skips encoder models
    c5_biasedit: true
    c6_gradient_ascent: true
```

CLI overrides (no config edit needed):
```bash
--skip-models mbert xlm-roberta
--only-models llama-3.1-8b muril
--skip-methods c4_dama c5_biasedit
```

---

## Key Metrics

| Metric | Model Type | Interpretation | Citation |
|--------|-----------|---------------|----------|
| **CLL** (Conditional Log-Likelihood) | Causal | sigmoid(log P_stereo − log P_anti); > 0.5 = stereotypical preference | [9] |
| **AUL** (Average Unmasked Likelihood) | Encoder | Pseudo-log-likelihood comparison; > 0.5 = stereotypical preference | [8] |
| **R** (Asymmetry Ratio) | All | T_debias / T_bias; R > 1 confirms hysteresis | Ours |
| **Perplexity** | All | Wikitext perplexity to verify capability retention | — |

---

## Results Directory

```
results/
├── phase0_baseline/              # baseline_results.json
├── phase1_injection/
│   └── <model>/<lang>/seed<N>/   # curves.json, final_biased/ checkpoint
├── phase2_removal/
│   └── <model>/<lang>/seed<N>/   # curves.json, final_debiased/ checkpoint
├── phase3_asymmetry/             # full_results.json (R tensor)
├── phase4_geometry/              # hessian_results.json, connectivity_results.json
├── phase5c_comparatives/
│   ├── c1_cda/<model>/           # Per-method results
│   ├── c2_self_debias/<model>/
│   ├── c3_inlp/<model>/
│   ├── c4_dama/<model>/
│   ├── c5_biasedit/<model>/
│   ├── c6_gradient_ascent/<model>/
│   └── comparative_R.json        # Combined R for all methods
├── phase6_cultural/              # cultural_analysis.json
├── figures/                      # figure1_hysteresis_curves.pdf, etc.
├── tables/                       # table1_baseline.tex through table5_comparative_R.tex
├── logs/                         # Per-model parallel execution logs
├── gpu_usage.json                # GPU hours + cost tracking
└── pipeline_log.json             # Full pipeline execution summary
```

---

## Critical Design Rules

1. **16-bit precision** — All models use float16 for uniformity. Gemma-3-4B uses bfloat16 (architectural requirement to avoid NaN in logits). No mixed precision within a model.
2. **Identical hyperparameters** — Injection (Phase 1) and removal (Phase 2) use the **same** LR, batch size, LoRA rank. Changing either invalidates R.
3. **3 seeds** — [42, 123, 456]. All experiments repeated 3 times for statistical validity.
4. **Drip-feed protocol** — Evaluate every 25 gradient steps to produce fine-grained bias curves.
5. **Incremental saving** — Results saved after every checkpoint for crash recovery.
6. **Data integrity** — Validation checks (deduplication, MASK tokens, target parsing) run on every data load.
7. **Mandatory citations** — All referenced papers cited as inline comments at algorithm implementation points.
8. **No hardcoded secrets** — All tokens loaded from `.env` via `python-dotenv`.

---

## Publication Readiness Notes (TACL / Nature CS)

### Evaluation Data Profile

| Split | Multi-CrowS-Pairs [1] | Indian-BhED [2] | Total per Language | Total (3 langs) |
|-------|----------------------|-----------------|-------------------|-----------------|
| **Train (80%)** | 1,137 | 607 | 1,744 | 5,232 |
| **Eval (20%)** | 284 | 152 | 436 | 1,308 |
| **Full** | 1,421 | 759 | 2,180 | 6,540 |

The stratified 80/20 split prevents data leakage between injection training and evaluation, following standard practice in fine-tuning studies.

### Per-Category Eval Sample Counts (English)

| Category | Source | n (eval) | Statistical Power |
|----------|--------|----------|-------------------|
| race-color | CrowS-Pairs | 99 | Strong |
| race | Indian-BhED | 75 | Strong |
| gender | Both | 74 | Strong |
| religion | Both | 44 | Adequate |
| socioeconomic | CrowS-Pairs | 34 | Adequate |
| nationality | CrowS-Pairs | 31 | Adequate (≥30) |
| caste | Indian-BhED | 21 | Weak — report with caveat |
| age | CrowS-Pairs | 17 | Weak — report with caveat |
| sexual-orientation | CrowS-Pairs | 16 | Weak — report with caveat |
| physical-appearance | CrowS-Pairs | 13 | Weak — aggregate only |
| disability | CrowS-Pairs | 12 | Weak — aggregate only |

### Recommended Reporting Strategy

1. **Main R analysis**: Use the **overall** bias score (all 436 samples aggregated). With 3 seeds × 3 languages × 6 models, the main hysteresis claim has strong statistical backing.
2. **Category-level breakdown**: Report per-category R **only for categories with n ≥ 30** (race-color, race, gender, religion, socioeconomic, nationality). These 6 categories cover 357/436 = 82% of eval data.
3. **Small categories**: Merge disability + physical-appearance + age into an "Other" aggregate, or report in supplementary material with explicit confidence intervals.
4. **Confidence intervals**: Report mean ± std across 3 seeds for all R values. Use bootstrap CI (95%) for the overall R.
5. **Statistical power statement**: Include in methods section: *"Category-level analyses for low-frequency categories (n < 20) should be interpreted with caution due to limited statistical power."*
6. **Comparison to baselines**: Baseline scores (0.49–0.56) are in the expected range for pretrained models, validating the measurement setup.
7. **Split justification**: *"We use a stratified 80/20 train/eval split to prevent data leakage. Direct comparison to published benchmarks using full datasets is not claimed; our contribution is the relative asymmetry ratio R, which is measured within our consistent evaluation framework."*

---

## Citations

### Datasets
- **[1]** Nangia, N., Vania, C., Bhalerao, R., & Bowman, S. R. (2020). *CrowS-Pairs: A Challenge Dataset for Measuring Social Biases in Masked Language Models.* EMNLP 2020.
- **[2]** Khandelwal, A., et al. (2023). *Indian-BhED: A Dataset for Measuring India-Centric Social Biases in Language Models.* arXiv:2309.08573.

### Core Methodology
- **[3]** Aghajanyan, A., Zettlemoyer, L., & Gupta, S. (2021). *Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning.* ACL 2021.
- **[4]** Bolukbasi, T., Chang, K.-W., Zou, J., Saligrama, V., & Kalai, A. (2016). *Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings.* NeurIPS 2016.
- **[5]** Hu, E. J., Shen, Y., Wallis, P., et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.

### Loss Landscape Analysis
- **[6]** Li, H., Xu, Z., Taylor, G., Studer, C., & Goldstein, T. (2018). *Visualizing the Loss Landscape of Neural Nets.* NeurIPS 2018.
- **[7]** Yao, Z., Gholami, A., Keutzer, K., & Mahoney, M. W. (2020). *PyHessian: Neural Networks Through the Lens of the Hessian.* IEEE BigData 2020.

### Bias Metrics
- **[8]** Kaneko, M., & Bollegala, D. (2022). *Unmasking the Mask — Evaluating Social Biases in Masked Language Models.* AAAI 2022.
- **[9]** Nadeem, M., Bethke, A., & Reddy, S. (2021). *StereoSet: Measuring Stereotypical Bias in Pretrained Language Models.* ACL 2021.
- **[10]** Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019). *Similarity of Neural Network Representations Revisited.* ICML 2019.

### Comparative Debiasing Methods
- **[11]** Zmigrod, R., Mielke, S. J., Wallach, H., & Cotterell, R. (2019). *Counterfactual Data Augmentation for Mitigating Gender Stereotypes in Languages with Rich Morphology.* ACL 2019.
- **[12]** Schick, T., Udupa, S., & Schütze, H. (2021). *Self-Diagnosis and Self-Debiasing: A Proposal for Reducing Corpus-Based Bias in NLP.* TACL 2021.
- **[13]** Ravfogel, S., Elazar, Y., Gonen, H., Trost, M., & Goldberg, Y. (2020). *Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection.* ACL 2020.
- **[14]** Limisiewicz, T., Mareček, D., & Stanczak, K. (2024). *Debiasing Algorithm through Model Adaptation.* ICLR 2024.
- **[15]** Xu, Z., et al. (2025). *BiasEdit: Debiasing Misconceptions in Language Models through Lightweight Model Editing.* TrustNLP@NAACL 2025.
- **[16]** Liu, Z., et al. (2025). *Rethinking Machine Unlearning for Large Language Models.* Nature Machine Intelligence, 7, 181–194.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Pipeline crashed mid-run | `python run_full_pipeline.py --start-from <step>` |
| Phase 2 says "checkpoints missing" | Phase 1 must finish first for ALL models |
| CUDA OOM | Use `--max-parallel 3` or `--sequential` |
| Flash Attention not found | Re-run `bash scripts/00_setup.sh` |
| HF_TOKEN error | Check `.env` file has valid HuggingFace token |
| Dataset download fails | Verify `Github_Classic_Token` in `.env` |
| Model takes too long | Skip with `--skip-models <name>` |
| Want to skip comparatives | `python run_full_pipeline.py --skip-comparatives` |

---

## License

Research use only. Datasets under CC-BY-4.0 (Multi-CrowS-Pairs) and CC-BY-SA-4.0 (Indian-BhED).
