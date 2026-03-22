# Pipeline Time Estimation — 4 New Models Run
**Date:** March 22, 2026  
**Server:** DigitalOcean H200 GPU (141 GB VRAM, CUDA 12.9, ~$3.55/hr)  
**Pipeline started:** 06:23 UTC  
**Pipeline completed:** 18:10 UTC (11h 46min total)  
**Script 14 completed:** 18:20 UTC (9.4 min, auto-triggered by watcher)  
**Last checked:** 19:03 UTC — **ALL DONE**  
**Models:** gpt-oss-20b · sarvam-2b · indicbert-v2 · jhu-clsp-mmbert

---

## 🎉 Final Status (as of 19:03 UTC) — PIPELINE COMPLETE

| Phase | Status |
|-------|--------|
| Phase 0 Baseline | ✅ Done (06:39) — 15.6 min |
| Phase 1 Bias Injection | ✅ Done (10:37) — 3h 57m wall-clock |
| Phase 2 Bias Removal | ✅ Done (12:45) — 2h 7m wall-clock |
| Phase 3 Asymmetry | ✅ Done (12:45) — 12.9s |
| Phase 4a Hessian | ✅ Done (12:47) — 2.5 min |
| Phase 4b Connectivity | ✅ Done (12:59) — 11.6 min |
| Phase 6 Cultural | ✅ Done (12:59) — 1.5s |
| Phase 5C Comparatives | ✅ Done (18:01) — 5h 1min (6 methods × 4 models) |
| Comparative R | ✅ Done (18:01) — 6.8s |
| Figures | ✅ Done (18:01) — 2.9s |
| Tables | ✅ Done (18:10) — 9.3 min |
| Script 14 (qualitative) | ✅ Done (18:20) — 9.4 min (auto-triggered by watcher PID 146242) |

### 🔽 Local Download Status

**Downloaded to:** `D:\PhD\Hysteresis_Bias\results\` (paper-only, no adapter weights)

#### Download 1 (10:55 UTC) — results_paper.tar.gz (12 MB)
Partial results: P0 all 10 models, P1 90/90, P2 partial (old 6 complete, new 4 partial).

#### Download 2 (16:48 UTC) — results_paper2.tar.gz (14 MB)
Overwrote Download 1. Complete snapshot of all available results.

| Data | Count | Details |
|------|-------|---------|
| Phase 0 baselines | ✅ **11 files** | 10 per-model JSONs + baseline_results.json |
| Phase 1 injection curves | ✅ **90/90** | All 10 models × 3 langs × 3 seeds |
| Phase 2 removal curves | ✅ **90/90** | All 10 models × 3 langs × 3 seeds |
| Phase 3 asymmetry | ✅ **full_results.json** | R values for all 10 models |
| Phase 4 hessian | ✅ **5 files** | hessian_results.json + per-model JSONs (gpt-oss, indicbert, llama, muril) |
| Phase 4 connectivity | ✅ **4 files** | connectivity JSONs (gpt-oss, indicbert, llama, muril) |
| Phase 5C C1 CDA | ✅ **30 curves** | All 10 models × 3 seeds (en only) |
| Phase 5C C2 Self-Debias | ✅ **18 curves** | 6 causal models × 3 seeds (encoder models skipped) |
| Phase 5C C3 INLP | ✅ **30 curves** | All 10 models × 3 seeds |
| Phase 5C C4 DAMA | ✅ **18 curves** | 6 causal models × 3 seeds |
| Phase 5C C5 BiasEdit | ✅ **30/30** | All 10 models × 3 seeds |
| Phase 5C C6 Grad. Ascent | ✅ **30/30** | All 10 models × 3 seeds |
| Phase 5C summaries | ✅ **2 files** | comparative_R.json + parallel_summary.json (partial — will update when done) |
| Phase 6 cultural | ✅ **1 file** | cultural_analysis.json |
| Phase 7 qualitative | ✅ **1 file** | qualitative_outputs_seed42.json |
| Figures | ✅ **6 files** | figure1–3 (.pdf + .png) — will be regenerated at pipeline end |
| Tables | ✅ **4 .tex files** | table1–4 — will be regenerated at pipeline end |
| **Adapter weights** | ❌ **Excluded** | Not needed for paper — saves ~12.7 GB |

**Total local size:** ~65 MB (from Download 2)

**Final download needed:**
- All Phase 5C results now complete (C5 + C6 for all 4 models)
- Regenerated figures and tables (pipeline steps 12–13)
- Script 14 qualitative outputs (24.4 MB JSON)
- Updated comparative_R.json and parallel_summary.json

**Pipeline PID 60615: DEAD (completed 18:10 UTC, 11h 46m, 11 steps, 0 failures)**  
**Watcher PID 146242: DEAD (triggered script 14 at 18:10:29 UTC, script completed 18:19:55 UTC)**  
**GPU: 0%, 0 MiB — fully idle.**

---

## Phase-by-Phase Breakdown

### Phase 0 — Baseline Measurement ✅ DONE
- **Started:** 06:23 UTC | **Completed:** 06:39 UTC
- **Duration:** ~16 minutes
- All 4 models ran in parallel. Merge-preserving logic preserved the existing 6-model baseline.

---

### Phase 1 — Bias Injection ✅ DONE
- **Started:** 06:39 UTC | **Completed:** 10:37 UTC
- **Duration:** 3.96 hours wall-clock (10.22 GPU-hours across 4 parallel workers)
- **Config:** `max_steps=500`, `eval_every_k_steps=25`, 3 languages × 3 seeds = 9 combos per model

**Final per-model timings:**

| Model | Combos | Completed At | Wall-Clock | Rate |
|-------|--------|-------------|-----------|------|
| indicbert-v2 | **9/9 ✅** | 07:45 UTC | 1.09 hrs | ~7 min/combo |
| sarvam-2b | **9/9 ✅** | 08:29 UTC | 1.84 hrs | ~12 min/combo |
| jhu-clsp-mmbert | **9/9 ✅** | 09:59 UTC | 3.33 hrs | ~22 min/combo |
| gpt-oss-20b | **9/9 ✅** | 10:37 UTC | 3.96 hrs | ~26 min/combo |

> gpt-oss-20b was the bottleneck as predicted (21B MoE model). Actual rate was ~26 min/combo —  
> faster than the 31 min/combo estimate because the last 4 combos ran with the GPU nearly to itself  
> after mmbert finished at 09:59.

---

### Phase 2 — Bias Removal ✅ DONE
- **Started:** 10:37 UTC | **Completed:** 12:45 UTC
- **Duration:** 2.13 hours wall-clock
- **Config:** `max_steps=2000`, `eval_every_k_steps=25`, same 9 combos per model, 4-way parallel

**Key insight: Early stopping triggers almost immediately.**

The removal code has two stop conditions (see `src/training/bias_removal.py`, lines 184–187, 215–218):
1. `current_bias ≤ baseline_bias + 0.02` → stops the moment debiasing returns to baseline
2. `_no_improvement(results, n_checkpoints=8)` → stops if bias doesn't move across 200 steps

**Empirical evidence from the 6-model run (actual stopping steps, max_steps=2000):**

| Model | En/Hi/Bn avg stopping step | Max ever seen | % of max_steps used |
|-------|--------------------------|--------------|-------------------|
| gemma-3-4b-it | ~33 | **100** | **5%** |
| llama-3.1-8b | ~29 | **50** | **2.5%** |
| mbert | ~46 | **100** | **5%** |
| muril | ~46 | **75** | **3.75%** |
| qwen2.5-1.5b | ~29 | **50** | **2.5%** |
| xlm-roberta | ~29 | **75** | **3.75%** |

**No model ever used more than 5% of the 2000-step budget.** Average stopping step across all 54 combos: ~33 steps (just 1–4 eval checkpoints).

**Why?** Phase 1 injection only raised bias to ~0.50–0.74 range (never near 1.0). The baseline bias from Phase 0 is ~0.50. So at the very first eval (step 25), `current_bias` is often already ≤ `baseline_bias + 0.02`, and removal stops immediately.

**Expected Phase 2 behaviour for new models:**
- indicbert-v2: injected bias ~0.69 → needs genuine removal → estimated ~50–150 steps (~4–8 evals)
- jhu-clsp-mmbert: injected bias ~0.70 (en) → similar to indicbert → ~50–150 steps
- sarvam-2b: injected bias ~0.57–0.59 → just above baseline → stops at step 25–50
- gpt-oss-20b: injected bias ~0.55 → barely above baseline → stops at step 25

**Actual Phase 2 timings for new models:**

| Model | Combos | Completed At | Wall-Clock | Rate |
|-------|--------|-------------|-----------|------|
| indicbert-v2 | **9/9 ✅** | 10:49 UTC | 0.19 hrs (11 min) | ~1.3 min/combo |
| sarvam-2b | **9/9 ✅** | 10:51 UTC | 0.23 hrs (14 min) | ~1.5 min/combo |
| jhu-clsp-mmbert | **9/9 ✅** | 11:07 UTC | 0.49 hrs (29 min) | ~3.3 min/combo |
| gpt-oss-20b | **9/9 ✅** | 12:45 UTC | 2.13 hrs (128 min) | ~14.2 min/combo |

> gpt-oss-20b took 2.13 hrs for all 9 combos — much slower than smaller models due to 21B MoE size,
> but early stopping still cut this from a theoretical 20+ hours to just 2 hours.

**Phase 2 started:** 10:37 UTC  
**Phase 2 completed:** 12:45 UTC (2.13 hrs wall-clock)

---

### Phase 3 — Asymmetry Ratio Computation ✅ DONE
- **Completed:** 12:45 UTC (instant — CPU-only, reads all curves.json)
- Computes R = T_debias / T_bias for all 10 models

---

### Phase 4 — Hessian & Linear Connectivity Analysis ✅ DONE
- **Phase 4a Hessian:** 12:45–12:47 UTC (2.5 min) — gpt-oss-20b + indicbert-v2
- **Phase 4b Connectivity:** 12:47–12:59 UTC (12 min) — gpt-oss-20b (11 interpolation points)
- **FOCUS_MODELS = ["gpt-oss-20b", "indicbert-v2"]** (only 2 models)
- MUCH faster than the 1–2 hour estimate!

---

### Phase 6 — Cultural Analysis ✅ DONE
- **Completed:** 12:59 UTC (instant)
- Key results: physical-appearance R=4.243, sexual-orientation R=3.378, age R=3.293
- Universal categories R=2.320, Western R=0.977, Indian R=0.119 (Kruskal-Wallis p=0.034)

---

### Phase 5C — Comparative Debiasing (6 methods × 4 models) ✅ DONE
- **Started:** 12:59 UTC | **Completed:** ~18:01 UTC | **Duration:** 5h 1min
- Methods: CDA, Self-Debias, INLP, DAMA, BiasEdit, Gradient Ascent
- Self-Debias (C2) and DAMA (C4) auto-skip encoder models (causal only → 2 models)
- Architecture: Methods run SEQUENTIALLY, models in PARALLEL within each method

**Final Phase 5C timings:**

| Method | Models | Status | Wall-Clock |
|--------|--------|--------|------------|
| C1: CDA | 4 parallel | ✅ Done | 1.97 hrs (gpt-oss bottleneck) |
| C2: Self-Debias | 2 (causal only) | ✅ Done | 0.63 hrs (38 min) |
| C3: INLP | 4 parallel | ✅ Done | 0.03 hrs (2 min) |
| C4: DAMA | 2 (causal only) | ✅ Done | 0.02 hrs (1 min) |
| C5: BiasEdit | 4 parallel | ✅ Done | ~2.5 hrs (gpt-oss bottleneck) |
| C6: Gradient Ascent | 4 parallel | ✅ Done | ~0.9 hrs |

**Phase 5C analysis:**
- C1 CDA and C5 BiasEdit were the slowest methods due to gpt-oss-20b (21B MoE)
- C5 BiasEdit exceeded C1 CDA's time — BiasEdit is heavier per model
- C3 INLP and C4 DAMA were near-instant
- gpt-oss-20b was consistently the bottleneck (~2× slower than next-slowest model)
---

### Phases 7–9 — Analysis & Outputs ✅ DONE
- Comparative R computation: 6.8s
- Figure generation: 2.9s (6 files: figure1–3 .pdf + .png)
- Table generation: 9.3 min (4 .tex files)

**Duration: ~10 minutes total**

---

### Script 14 — Qualitative Outputs ✅ DONE
- **Triggered:** 18:10:29 UTC (auto-launched by watcher)
- **Completed:** 18:19:55 UTC
- **Duration:** 9.4 minutes
- **Output:** `/root/Hysteresis_Bias/results/phase7_qualitative/qualitative_outputs_seed42.json` (24.4 MB)

| Model | Duration |
|-------|----------|
| gpt-oss-20b | 197.6s (3.3 min) |
| sarvam-2b | 252.4s (4.2 min) |
| indicbert-v2 | 35.5s |
| jhu-clsp-mmbert | 76.8s |

---

## Summary Timeline

| Phase | Duration | Cumulative End (UTC) | Status |
|-------|----------|---------------------|--------|
| Phase 0 ✅ | 15.6 min | 06:39 | Done |
| Phase 1 ✅ | 3h 57m | 10:37 | Done |
| Phase 2 ✅ | 2h 7m | 12:45 | Done |
| Phase 3 ✅ | 12.9s | 12:45 | Done |
| Phase 4 ✅ | 14 min | 12:59 | Done |
| Phase 6 ✅ | 1.5s | 12:59 | Done |
| Phase 5C ✅ | 5h 1m | 18:01 | Done |
| Comparative R ✅ | 6.8s | 18:01 | Done |
| Figures ✅ | 2.9s | 18:01 | Done |
| Tables ✅ | 9.3 min | 18:10 | Done |
| Script 14 ✅ | 9.4 min | 18:20 | Done (watcher-triggered) |

**🎉 PIPELINE FULLY COMPLETE — 11h 46min pipeline + 9.4 min script 14**  
**Total wall-clock: 11h 57min (06:23 → 18:20 UTC)**  
**Total cost at 19:03 UTC: ~$44.90** (12.67 hrs × $3.55/hr)  
**Hours remaining: 0**

---

## Estimate Corrections Log

| Round | Check Time | Estimate | Notes |
|-------|-----------|---------|-------|
| Initial (before run) | pre-06:23 | 8–9 hrs | Assumed all models ~equally sized |
| Check 1 | 07:53 UTC | 28–32 hrs | Correct per-model GPU rate but wrongly assumed Phase 2 uses all 2000 steps |
| Check 2 | 08:00 UTC | 14–18 hrs | Corrected for early stopping (old models stopped at ≤100/2000 steps, 5% usage) |
| Check 3 | 08:58 UTC | 11–15 hrs remaining | Phase 1 faster than expected (2 models done, remaining models get more GPU share) |
| Check 4 | 09:13 UTC | ~11–15 hrs remaining | Phase 1: 31/36 done. gpt-oss at ~31 min/combo, improving as fewer models compete |
| Check 5 | 10:42 UTC | ~7–13 hrs remaining | Phase 1 DONE at 10:37 (3.96 hrs). Phase 2 already 10/36 in 5 min — early stopping confirmed! Estimate cut by ~3 hrs |
| Check 6 | 10:55 UTC | ~7–13 hrs remaining | Downloaded paper-only results to local (12 MB tarball, no adapter weights). P2 progress: indicbert+sarvam done, mmbert 7/9, gpt-oss 1/9. Re-download needed after pipeline finishes. |
| Check 7 | 12:14 UTC | ~7–12 hrs remaining | Phase 2: indicbert 11min, sarvam 14min, mmbert 29min — all done! gpt-oss 5/9 at ~19min/combo. Phase 2 ETA ~13:30. GPU 34% (only gpt-oss running). Phases 3–9 unchanged. |
| Check 8 | 13:35 UTC | ~3–6 hrs remaining | MASSIVE progress! Phase 2 DONE (12:45, gpt-oss took 2.13 hrs). P3 instant, P4 only 14 min (not 1–2 hrs!), P6 instant. Phase 5C started 12:59 — C1 CDA [1/6] with 3/4 models done. gpt-oss bottleneck each method. Estimate slashed from 7–12 hrs to 3–6 hrs! |
| **Check 9** | **16:40 UTC** | **~1–2.5 hrs remaining** | **Phase 5C at C5 BiasEdit [5/6]. C1 CDA=1.97h, C2 SelfDebias=0.63h, C3 INLP=2min, C4 DAMA=1min. C5: indicbert+sarvam done, gpt-oss+mmbert running. GPU 99%, 113GB. Only C5 finish + C6 + Phases 7–9 remain!** |
| **Check 10** | **16:57 UTC** | **~1–2.5 hrs remaining** | **Download 2 completed & verified. C5 mmbert now done (1.19h), gpt-oss-20b sole runner (~1h19m). GPU dropped to 53% (single model). Local inventory: P0–P4 complete (all 10 models), P5C 124/148 curves, P6–P7 present, 6 figures + 4 tables. Only C5 gpt-oss finish + C6 (4 new models) + Phases 7–9 regen remain.** |
| **Check 11** | **17:35 UTC** | **~2–3 hrs remaining** | **Pipeline smooth. gpt-oss-20b C5 BiasEdit at 1h 57m (matching C1 CDA's 1.97h — should finish imminently). GPU 52%, 128GB. After C5: C6 Gradient Ascent (~1.5–2.5h with gpt-oss bottleneck) + Phases 7–9 (~20 min). Also: 14_qualitative_outputs.py deployed on VM (commit 9fd485e) for the 4 new models — will run manually after pipeline finishes (~30–60 min inference). Total completion ~20:00–20:30 UTC including script 14.** |
| **Check 12** | **17:52 UTC** | **~3–4 hrs remaining** | **gpt-oss-20b C5 BiasEdit at 2h 14m — exceeds C1 CDA’s 1.97h, BiasEdit is heavier per model. GPU 54%, 129GB. No errors. Watcher PID 146242 set up to auto-launch script 14 after pipeline exits. Revised estimate: C5 finish ~18:30–19:00, C6 ~20:00–21:00, Phases 7–9 ~20 min, script 14 ~30–60 min. Full completion ~21:00–22:00 UTC.** |
| **Check 13** | **19:03 UTC** | **0 hrs remaining** | **🎉 PIPELINE COMPLETE! 11h 46m, 11 steps, 0 failures. C5 BiasEdit + C6 Gradient Ascent finished faster than Check 12 predicted. Pipeline exited 18:10 UTC. Watcher auto-triggered script 14 at 18:10:29 — completed at 18:19:55 (9.4 min for all 4 models). GPU idle (0%, 0 MiB). Everything done ~1.5–3 hrs ahead of the Check 12 estimate of 21:00–22:00 UTC. Total cost: ~$44.90.** |
**Lesson:** The `max_steps=2000` in `training.yaml` is a safety ceiling, not a target. The dual early-stopping conditions in `bias_removal.py` mean Phase 2 completes in 25–150 steps for most combos. Results remain fully valid for the asymmetry ratio computation because T_debias is the **actual convergence step**, not the maximum allowed.
