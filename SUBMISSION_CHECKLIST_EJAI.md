# EJAI Submission Checklist — Step-count ratios do not measure bias hysteresis

Target: The European Journal on Artificial Intelligence (EJAI), SAGE / EurAI.
Status key: ✓ done in repo · ⚠ needs author verification · ▢ optional (adds length/strength).

## Part A — hard compliance (submission blockers)

| Item | Status | Where |
|---|---|---|
| A1. Abstract ≤ 150 words, unstructured | ✓ (149 words) | `Submission/EJAI_Hysteresis.tex` abstract |
| A2. Manuscript fully anonymized (no author/affiliation/email/header) | ✓ | `EJAI_Hysteresis.tex` header (`\author{Anonymous}`, empty `\affiliation`) |
| A2. Separate Title Page (withheld from reviewers) | ✓ | `Submission/TitlePage_EJAI_Hysteresis.tex` |
| A3. Statements and Declarations — all 7 subheadings | ✓ | `EJAI_Hysteresis.tex` end matter + title page |
| A4. Data availability (real; anonymized in manuscript, full URLs in title page) | ⚠ verify URLs | manuscript = anonymized; title page = GitHub `DevDaring/Hysteresis_Bias`, HF `Debk/...` |
| A5. Model citations for IndicBERTv2 and mmBERT | ✓ | `sample.bib` (`doddapaneni2023indicbert`, `mmbert2025`); cited in Method |
| A5. Reference style SAGE Harvard | ✓ | `\bibliographystyle{SageH}`, `sagej` `sageh` option |
| A6. Template / class = sagej with sageh+times | ✓ | `\documentclass[Afour,sageh,times]{sagej}` |
| A7. Generative-AI use disclosure | ⚠ confirm wording | Acknowledgements in manuscript + title page |

EJAI facts confirmed by research: abstract 150-word max; page limit 30 (hard cap; ~10–14 typical for full paper); SAGE Harvard; double-anonymized; sagej class required.

## Part B — expansion from existing data (no new runs)

| Item | Status | Source file |
|---|---|---|
| Expanded Related Work (two protocols + measurement-validity thread) | ✓ | `EJAI_Hysteresis.tex` §Related work |
| Method: AUL bias-score definition, threshold rationale | ✓ | Eq. (bias) |
| Method: crossing-time interpolation + censoring rule | ✓ | Eq. (cross) |
| Method: model-clustered bootstrap CI described | ✓ | §Two objectives |
| Method: Algorithm environment (inject–remove–measure) | ✓ | Algorithm 1 |
| Per-category table (matched) | ✓ | `results/wp1_symmetric/summary.json` via `scripts/analyze_partB.py` |
| Per-seed stability table | ✓ | same |
| Trajectory figure (injection slow vs removal fast) | ✓ | `Submission/images/figure_trajectory.png` |
| Censoring-illustration figure (ceiling mass inflates mean) | ✓ | `Submission/images/figure_censoring.png` |
| R-distribution figure at 300 dpi | ✓ | `Submission/images/figure_R_distribution.png` |
| Discussion + Limitations + Reproducibility | ✓ | §Discussion |

Honest nuance surfaced from data: socioeconomic status is the one category with median R > 1 under the matched objective; reported transparently in Table (per-category) and Limitations.

## Part C — new experiments

C1 and C2 have now been RUN on a dedicated on-demand L4 and integrated into the paper with real numbers.

| Item | Status | Script | Result in paper |
|---|---|---|---|
| C1 threshold sweep (θ 0.60/0.70/0.80) | ✓ DONE | `scripts/theta_sensitivity.py` | §"The ratio depends on the threshold": R swings 5.11 → 0.60 → 0.13; Table + `figure_theta_sensitivity.png` |
| C2 loop-area / bias-field sweep | ✓ DONE | `scripts/loop_area.py`, `analyze_loop_area.py` | §"A threshold-free measure": median A=0.016, p<0.0001, model-dependent; Table + `figure_loop_area.png` |
| Single entry point (C2 then C1) | ✓ | `scripts/run_c1_c2.py` | resume-capable, `--dry-run` = 2-instance gate |
| C3 full fine-tuning check | scaffold (not run) | `scripts/full_finetune.py` | named as the main remaining check |
| C4 extra categories (gender, caste) | scaffold (not run) | `scripts/extra_categories.py` | optional |

C1 θ=0.80 was stopped at 28/90 converged conditions (slow censored tail; the ~38× swing was already unambiguous); n reported honestly per threshold. Loop-area effect is small and model-dependent (IndicBERTv2 +0.10, MuRIL +0.07, mmBERT ≈ 0), reported without overclaim. Every number read from `results/theta_sensitivity/` and `results/loop_area/`; nothing fabricated.

## Consolidated TODO punch-list (author must resolve before submission)

1. ⚠ Confirm the exact AI-use disclosure wording per current SAGE policy (2 places: manuscript Acknowledgements, title page).
2. ⚠ Confirm the public GitHub and Hugging Face URLs in the title-page Data Availability once the repos are made public.
3. ▢ Optional strengtheners that also add length toward ~10–14 pp: run C1 (threshold sweep — pre-empts "depends on θ=0.7") and C2 (loop area — converts rebuttal into a positive threshold-free measure). C2 is the single biggest strengthener.
4. ▢ Verify the mmBERT reference once a peer-reviewed version exists (currently arXiv:2509.06888).
5. ▢ Confirm EJAI-specific sagej option string from the SAGE Author Gateway (current: `Afour,sageh,times`).

## Build

- `pdflatex EJAI_Hysteresis.tex; bibtex EJAI_Hysteresis; pdflatex x2` → compiles clean, 7 pages, no undefined references.
- Title page compiles standalone.
- Style: 0 em-dashes, 0 banned verbs, 0 first-person plural.

This draft is compiling, compliant, and expanded. It is not "ready to submit": resolve the punch-list above first.
