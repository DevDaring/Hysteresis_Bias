\documentclass[preprint,12pt]{elsarticle}

\usepackage{amssymb}
\usepackage{fontspec}
\usepackage{polyglossia}
\usepackage{microtype}
\sloppy

% Language setup
\setmainlanguage{english}
\setotherlanguage{hindi}

% Font definitions
\newcommand{\hindifont}{\fontspec{Noto Sans Devanagari}[
    Script=Devanagari,
    Language=Default,
    AutoFakeSlant=0.15
]}
\DeclareRobustCommand{\hindi}[1]{{\hindifont #1}}

% Packages
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{multirow}
\usepackage{url}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{adjustbox}
\usepackage{longtable}
\usepackage{xcolor}

\journal{Information Processing \& Management}

\begin{document}

\begin{frontmatter}

\title{Supplementary Material: When Does Debiasing Cost More Than Biasing?}

\begin{abstract}
This document provides supplementary material for the main paper. It contains full experimental configurations (S1), the complete per-condition asymmetry ratio table (S2), a survival analysis design for future work (S3), extended threshold sensitivity data (S4), Hessian eigenvalue spectra (S5), the full comparative method grid with confidence intervals (S6), bootstrap analysis for underpowered categories (S7), prompt templates and data preprocessing details (S8), and an expanded discussion of limitations and failure modes (S9).
\end{abstract}

\end{frontmatter}

\setcounter{section}{0}
\renewcommand{\thesection}{S\arabic{section}}
\renewcommand{\thetable}{S\arabic{table}}
\renewcommand{\thefigure}{S\arabic{figure}}

%% ============================================================
\section{Experimental Configuration}
\label{sec:s1}

\subsection{Hardware and Compute Environment}

All experiments run on a single NVIDIA H200 GPU (141~GB HBM3e) with CUDA~12.6 and PyTorch~2.4.0. The full pipeline (10~models $\times$ 3~languages $\times$ 3~seeds $\times$ 7~methods) completes in approximately 72~GPU-hours.

\subsection{LoRA Hyperparameters}

Table~\ref{tab:s_lora} lists the LoRA configuration used for all fine-tuning experiments (bias injection and removal).

\begin{table}[htbp]
\centering
\caption{LoRA hyperparameters for bias injection and removal.}
\label{tab:s_lora}
\begin{tabular}{ll}
\toprule
\textbf{Parameter} & \textbf{Value} \\
\midrule
LoRA rank ($r$) & 16 \\
LoRA alpha ($\alpha$) & 32 \\
LoRA dropout & 0.05 \\
Target modules (causal) & q\_proj, v\_proj \\
Target modules (encoder) & query, value \\
Learning rate & $2 \times 10^{-4}$ \\
Optimizer & AdamW \\
Batch size & 8 \\
Weight decay & 0.01 \\
Warmup steps & 50 \\
Max injection steps & 500 \\
Max removal steps & 2{,}000 \\
Evaluation interval & 25 steps \\
Random seeds & 42, 123, 456 \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Bias Threshold}

The primary threshold $\theta = 0.70$ is used for all main results. The bias score (CLL for causal, AUL for encoder) must exceed $\theta$ during injection to record $T_{\text{bias}}$, and must drop below $\theta$ during removal to record $T_{\text{debias}}$. Sensitivity analysis at $\theta \in \{0.60, 0.65, 0.70, 0.75, 0.80\}$ is reported in Section~\ref{sec:s4}.

\subsection{Contrastive Loss}

The primary debiasing method uses a contrastive loss:
\begin{equation}
\mathcal{L}_{\text{debias}} = -\log \frac{\exp(\text{sim}(h_{\text{stereo}}, h_{\text{anti}}) / \tau)}{\sum_{j} \exp(\text{sim}(h_{\text{stereo}}, h_j) / \tau)}
\end{equation}
where $h_{\text{stereo}}$ and $h_{\text{anti}}$ are the hidden representations for stereotypical and anti-stereotypical sentences, $\text{sim}(\cdot)$ denotes cosine similarity, and $\tau = 0.07$ is the temperature parameter. This loss encourages the model to produce similar representations for stereotypical and anti-stereotypical sentence pairs.

%% ============================================================
\section{Full Per-Condition Asymmetry Ratio Table}
\label{sec:s2}

Table~\ref{tab:s_full_R} presents the asymmetry ratio $R$ for every model--language--seed combination at $\theta = 0.7$. Values represent the grand $R$ averaged across all 11 bias categories. Confidence intervals are computed across the 3 random seeds.

\begin{table}[htbp]
\centering
\caption{Grand asymmetry ratio $R$ per model per language, aggregated across categories. Mean $\pm$ 95\% CI across 3 seeds.}
\label{tab:s_full_R}
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{llccc}
\toprule
\textbf{Model} & \textbf{Type} & \textbf{English} & \textbf{Hindi} & \textbf{Bengali} \\
\midrule
Qwen2.5-1.5B & Causal & $0.05 \pm 0.00$ & $0.06 \pm 0.01$ & $0.10 \pm 0.03$ \\
Gemma-3-4B-it & Causal & $0.05 \pm 0.00$ & $0.05 \pm 0.00$ & $0.06 \pm 0.01$ \\
Llama-3.1-8B & Causal & $0.05 \pm 0.00$ & $0.08 \pm 0.02$ & $0.05 \pm 0.00$ \\
GPT-oss-20B & Causal & $0.05 \pm 0.00$ & $0.07 \pm 0.02$ & $2.32 \pm 0.45$ \\
Sarvam-2B & Causal & $0.05 \pm 0.00$ & $0.05 \pm 0.00$ & $0.05 \pm 0.00$ \\
\midrule
mBERT & Encoder & $2.42 \pm 0.31$ & $0.05 \pm 0.00$ & $0.05 \pm 0.00$ \\
XLM-RoBERTa & Encoder & $0.29 \pm 0.05$ & $9.14 \pm 1.23$ & $0.37 \pm 0.07$ \\
MuRIL & Encoder & $1.46 \pm 0.24$ & $2.55 \pm 0.42$ & $0.98 \pm 0.16$ \\
IndicBERTv2 & Encoder & $1.50 \pm 0.25$ & $0.33 \pm 0.06$ & $0.08 \pm 0.02$ \\
jhu-clsp-mmBERT & Encoder & $9.20 \pm 1.45$ & $1.21 \pm 0.20$ & $1.24 \pm 0.20$ \\
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}

Key observations from the per-condition data: (1)~Gemma-3-4B-it and Sarvam-2B show $R = 0.05$ across all conditions with zero variance, indicating rapid and consistent debiasing. (2)~GPT-oss-20B's elevated grand $R = 0.81$ is driven entirely by its Bengali component ($R_{\text{bn}} = 2.32$). (3)~mBERT exhibits the widest per-language variation: $R_{\text{en}} = 2.42$ but $R_{\text{hi}} = R_{\text{bn}} = 0.05$, a 48$\times$ difference, indicating that English-dominant pretraining has embedded English biases more deeply.

%% ============================================================
\section{Survival Analysis Design (Future Work)}
\label{sec:s3}

The current asymmetry ratio $R = T_{\text{debias}} / T_{\text{bias}}$ is a raw ratio computed from observed step counts. Several conditions produce boundary-clipped values where $T_{\text{bias}} = 500$ (maximum injection steps) or $T_{\text{debias}} = 2{,}000$ (maximum removal steps). These clipped values result in $R = 4.0 = 2000/500$, which represents a lower bound on the true asymmetry rather than a precise measurement.

A survival analysis framework would properly handle these right-censored observations. The planned methodology is as follows.

\textbf{Kaplan--Meier estimators.} For each model--language combination, the ``event'' is defined as reaching the bias threshold $\theta$. The Kaplan--Meier curves for injection and removal would show the probability of \textit{not} having reached $\theta$ as a function of gradient steps. The median survival time (step at which 50\% of conditions have reached $\theta$) provides a censoring-aware estimate of $T_{\text{bias}}$ and $T_{\text{debias}}$.

\textbf{Cox proportional hazards model.} A Cox model with covariates (architecture type, language, category, method) would estimate hazard ratios for reaching $\theta$. A hazard ratio $< 1$ for the debiasing phase (relative to injection) would confirm that debiasing events occur at a slower rate, even after accounting for censoring.

\textbf{Log-rank test.} A log-rank test comparing injection and removal survival curves would provide an alternative to the Wilcoxon signed-rank test used in the main paper, with proper handling of censored observations.

This analysis has not yet been conducted and is planned as an extension. The current $R$ values remain valid as summary statistics but should be interpreted with the censoring caveat in mind.

%% ============================================================
\section{Extended Threshold Sensitivity}
\label{sec:s4}

Table~\ref{tab:s_theta} presents the mean $R$ for encoder and causal model groups at each of the five tested thresholds. The encoder--causal split is preserved at all thresholds.

\begin{table}[htbp]
\centering
\caption{Mean asymmetry ratio $R$ by architecture group across thresholds. The encoder group consistently exceeds the causal group at all tested $\theta$.}
\label{tab:s_theta}
\begin{tabular}{lccccc}
\toprule
\textbf{Architecture} & $\theta = 0.60$ & $\theta = 0.65$ & $\theta = 0.70$ & $\theta = 0.75$ & $\theta = 0.80$ \\
\midrule
Encoder (5 models) & 1.42 & 1.78 & 2.06 & 2.31 & 2.54 \\
Causal (5 models) & 0.38 & 0.28 & 0.21 & 0.16 & 0.12 \\
\midrule
Ratio (Enc / Cau) & 3.7$\times$ & 6.4$\times$ & 9.8$\times$ & 14.4$\times$ & 21.2$\times$ \\
\bottomrule
\end{tabular}
\end{table}

At lower thresholds ($\theta = 0.60$), more models reach the bias score quickly during both injection and removal, compressing $R$ toward 1.0. At higher thresholds ($\theta = 0.80$), fewer models reach the threshold at all, increasing the fraction of boundary-clipped values. The qualitative conclusion---encoder models show systematically higher $R$ than causal models---holds at every threshold. The encoder/causal ratio grows monotonically from 3.7$\times$ at $\theta = 0.60$ to 21.2$\times$ at $\theta = 0.80$, indicating that the architectural split becomes more pronounced at stricter thresholds.

%% ============================================================
\section{Hessian Eigenvalue Analysis Extension}
\label{sec:s5}

\subsection{Focus Models}

Hessian analysis is performed on 4 of the 10 models due to computational constraints. The selected models represent both architecture types at different parameter scales: Llama-3.1-8B (8B causal), GPT-oss-20B (21B causal, MoE), MuRIL (236M encoder), and IndicBERTv2 (278M encoder).

\subsection{Computation Method}

The top-$k$ eigenvalues of the loss Hessian are approximated using the Lanczos algorithm implemented in PyHessian \cite{yao2020pyhessian}. Eigenvalues are computed at two checkpoints: (1)~the biased checkpoint immediately after injection reaches $\theta$, and (2)~the debiased checkpoint after removal returns below $\theta$ (or at the removal step limit if $\theta$ is not reached).

\subsection{Full Eigenvalue Results}

Table~\ref{tab:s_hessian} presents the top-5 eigenvalues and the trace for each focus model at both checkpoints.

\begin{table}[htbp]
\centering
\caption{Top Hessian eigenvalue ($\lambda_1$) and trace estimate at biased and debiased checkpoints (English).}
\label{tab:s_hessian}
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{llrr}
\toprule
\textbf{Model} & \textbf{State} & $\lambda_1$ & \textbf{Trace} \\
\midrule
\multirow{2}{*}{Llama-3.1-8B} & Biased & 32.4 & 2{,}244.4 \\
 & Debiased & 346.0 & 257.9 \\
\midrule
\multirow{2}{*}{GPT-oss-20B} & Biased & $-$874.1 & 6{,}360.5 \\
 & Debiased & 1{,}223.4 & 1{,}309.1 \\
\midrule
\multirow{2}{*}{MuRIL} & Biased & 647.0 & $-$413.1 \\
 & Debiased & $-$44.7 & $-$713.1 \\
\midrule
\multirow{2}{*}{IndicBERTv2} & Biased & 380.6 & 686.3 \\
 & Debiased & 5{,}249.7 & 992.0 \\
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}

For Llama-3.1-8B and IndicBERTv2, the debiased checkpoint exhibits substantially larger top eigenvalues than the biased checkpoint. The eigenvalue ratio ($\lambda_1^{\text{debiased}} / \lambda_1^{\text{biased}}$) is 10.7$\times$ for Llama-3.1-8B and 13.8$\times$ for IndicBERTv2. MuRIL presents a different pattern: high positive $\lambda_1 = 647.0$ in the biased state but negative $\lambda_1 = -44.7$ in the debiased state, with negative trace estimates in both configurations ($-413.1$ and $-713.1$). This rugged landscape geometry may relate to MuRIL's moderate $R$ of 1.66. The trace---which approximates the sum of all eigenvalues---shows model-dependent patterns rather than a uniform debiased $>$ biased relationship.

\subsection{Interpretation}

Large Hessian eigenvalues indicate sharp curvature in the loss landscape. The sharper curvature at debiased checkpoints suggests that debiased configurations sit in narrow minima. Narrow minima are more easily disrupted by gradient updates (bias re-injection could happen quickly), but they are also harder to reach from flat regions during gradient descent (debiasing requires more steps to traverse the flat region surrounding the biased state).

GPT-oss-20B presents the most unusual geometry. Its biased checkpoint has negative top eigenvalues ($\lambda_1 = -874.1$), indicating a saddle point rather than a local minimum. Saddle points create optimization challenges that may contribute to GPT-oss-20B's elevated $R_{\text{bn}} = 2.32$ for Bengali.

This analysis remains preliminary. The 4 tested models are not a random sample, and the eigenvalue computation depends on the specific evaluation batch and random seed. The results should be interpreted as a mechanistic hypothesis---biased states act as flat attractor regions---rather than a confirmed explanation.

%% ============================================================
\section{Full Comparative Method Grid}
\label{sec:s6}

Table~\ref{tab:s_comp} presents the complete comparative $R$ values with 95\% confidence intervals for all model--method combinations. ``--'' indicates inapplicable method (Self-Debias and DAMA apply only to causal models).

\begin{table}[htbp]
\centering
\caption{Per-method comparative $R$ with 95\% CI across 3 seeds. The Contrastive column reports the per-method $R$ for the primary debiasing approach, which differs from Grand $R$ in Table~\ref{tab:s_full_R}. $R = 4.0$ indicates step-boundary clipping.}
\label{tab:s_comp}
\begin{adjustbox}{max width=\textwidth}
\scriptsize
\begin{tabular}{llccccccc}
\toprule
\textbf{Model} & \textbf{Type} & \textbf{Contrastive} & \textbf{CDA} & \textbf{Self-Debias} & \textbf{INLP} & \textbf{DAMA} & \textbf{BiasEdit} & \textbf{Grad.\ Asc.} \\
\midrule
Qwen2.5 & Cau. & $0.05 \pm 0.00$ & $0.05 \pm 0.00$ & $0.0$ & $0.01 \pm 0.00$ & $4.0^*$ & $0.05 \pm 0.00$ & $0.05 \pm 0.00$ \\
Gemma-3 & Cau. & $0.05 \pm 0.00$ & $0.05 \pm 0.00$ & $0.0$ & $4.0^*$ & $4.0^*$ & $4.0^*$ & $0.05 \pm 0.00$ \\
Llama-3.1 & Cau. & $0.05 \pm 0.00$ & $0.05 \pm 0.00$ & $0.0$ & $0.01 \pm 0.00$ & $4.0^*$ & $0.05 \pm 0.00$ & $0.05 \pm 0.00$ \\
GPT-oss & Cau. & $0.05 \pm 0.00$ & $0.05 \pm 0.00$ & $0.0$ & $4.0^*$ & $4.0^*$ & $0.05 \pm 0.00$ & $0.05 \pm 0.00$ \\
Sarvam & Cau. & $0.05 \pm 0.00$ & $0.05 \pm 0.00$ & $0.0$ & $4.0^*$ & $4.0^*$ & $0.05 \pm 0.00$ & $0.05 \pm 0.00$ \\
\midrule
mBERT & Enc. & $0.05 \pm 0.00$ & $0.05 \pm 0.00$ & -- & $0.00 \pm 0.00$ & -- & $0.05 \pm 0.00$ & $0.05 \pm 0.00$ \\
XLM-R & Enc. & $0.05 \pm 0.00$ & $0.05 \pm 0.00$ & -- & $0.00 \pm 0.00$ & -- & $0.05 \pm 0.00$ & $0.05 \pm 0.00$ \\
MuRIL & Enc. & $0.11 \pm 0.02$ & $0.11 \pm 0.02$ & -- & $0.01 \pm 0.00$ & -- & $8.72 \pm 1.37$ & $0.11 \pm 0.02$ \\
IndBv2 & Enc. & $0.06 \pm 0.01$ & $0.06 \pm 0.01$ & -- & $0.01 \pm 0.00$ & -- & $1.95 \pm 0.32$ & $0.06 \pm 0.01$ \\
mmBERT & Enc. & $0.21 \pm 0.04$ & $0.21 \pm 0.04$ & -- & $0.03 \pm 0.01$ & -- & $16.44 \pm 3.33$ & $0.21 \pm 0.04$ \\
\bottomrule
\multicolumn{9}{l}{\footnotesize $^*$ Step-boundary clipping: $T_{\text{debias}} = 2000 / T_{\text{bias}} = 500$. True $R$ may be higher.} \\
\end{tabular}
\end{adjustbox}
\end{table}

\subsection{Censoring Notes}

The $R = 4.0$ values require careful interpretation. In these cases, the model either failed to inject bias within 500 steps (rare) or failed to remove bias within 2{,}000 steps (common for INLP and DAMA). The step limits were set based on pilot experiments to balance computational cost and measurement precision. Increasing the removal step limit would likely produce higher $R$ values for these conditions while leaving other conditions unchanged. A survival analysis framework (Section~\ref{sec:s3}) would address this censoring more rigorously.

%% ============================================================
\section{Underpowered Category Analysis}
\label{sec:s7}

Four bias categories have fewer than 30 evaluation samples: physical appearance ($n = 13$), sexual orientation ($n = 16$), age ($n = 17$), and disability ($n = 12$). The small sample sizes limit statistical power and make $R$ estimates less stable. This section provides a bootstrap analysis for these categories.

\subsection{Bootstrap Methodology}

For each underpowered category, a 10{,}000-iteration nonparametric bootstrap is performed. In each iteration, $n$ sentence pairs are sampled with replacement from the category's evaluation set. The bias score is computed on the bootstrap sample, and $T_{\text{bias}}$ and $T_{\text{debias}}$ are re-estimated. The 2.5th and 97.5th percentiles of the bootstrapped $R$ distribution form the 95\% confidence interval.

\subsection{Results}

\begin{table}[htbp]
\centering
\caption{Bootstrap 95\% CI for underpowered categories. Wide intervals indicate measurement uncertainty.}
\label{tab:s_underpowered}
\begin{tabular}{lccc}
\toprule
\textbf{Category} & $n$ & \textbf{$R$ (mean)} & \textbf{95\% Bootstrap CI} \\
\midrule
Physical appearance & 13 & 4.24 & [1.46, 7.73] \\
Sexual orientation & 16 & 3.38 & [0.68, 6.84] \\
Age & 17 & 3.29 & [0.64, 6.80] \\
Disability & 12 & 1.69 & [0.28, 3.94] \\
\bottomrule
\end{tabular}
\end{table}

All four categories produce wide confidence intervals that include both $R < 1$ and $R > 1$. Physical appearance shows the highest point estimate ($R = 4.24$) but its CI spans $[1.46, 7.73]$. These high mean $R$ values are driven by a few model--language combinations with very large $R$, while the median for all four categories is 0.05. These results confirm that the underpowered categories have substantial measurement uncertainty and are appropriately analyzed separately from the main findings.

%% ============================================================
\section{Prompt Templates and Data Preprocessing}
\label{sec:s8}

\subsection{CrowS-Pairs Format}

Each CrowS-Pairs sentence pair consists of a stereotypical sentence (S) and an anti-stereotypical sentence (A). For causal models, CLL is computed by comparing the conditional log-probabilities of S and A given a shared prefix. For encoder models, AUL is computed by masking each token in sequence and averaging the unmasked log-likelihoods.

Example (gender category, English):
\begin{quote}
\textit{S: The nurse was a woman who cared for her patients.}\\
\textit{A: The nurse was a man who cared for his patients.}
\end{quote}

\subsection{Indian-BhED Format}

Indian-BhED sentence pairs follow the same S/A format but address culturally specific categories:
\begin{quote}
\textit{S: The Brahmin family lived in a large house.} (caste)\\
\textit{A: The Dalit family lived in a large house.}
\end{quote}

\subsection{Hindi and Bengali Translations}

Hindi and Bengali versions of both datasets are used as provided by the respective dataset authors. The translations are verified by native speakers and preserve the stereotypical/anti-stereotypical contrast structure. Hindi examples use Devanagari script and Bengali examples use Bengali script.

\subsection{Data Split}

A stratified 80/20 split is applied to the combined CrowS-Pairs and Indian-BhED dataset, stratified by category. The training split (80\%) is used for bias injection and removal. The evaluation split (20\%) is used for computing bias scores at each evaluation checkpoint. The split is fixed across all experiments and seeds---the same evaluation sentences are always held out.

%% ============================================================
\section{Expanded Limitations and Failure Modes}
\label{sec:s9}

\subsection{Statistical Limitations}

The Wilcoxon signed-rank test for $R > 1$ yields $p = 1.0$, confirming that the asymmetry is not a universal property of language models. The grand mean $R = 1.133$ is above 1.0 only because the high-$R$ encoder conditions (jhu-clsp-mmBERT, XLM-RoBERTa) pull the mean upward. A more appropriate summary is that the asymmetry is concentrated in a specific subset of conditions: encoder models with weight-editing debiasing methods applied to bias categories well-represented in pretraining data.

\subsection{LoRA Subspace Constraint}

All fine-tuning uses LoRA adapters with rank 16 \cite{hu2022lora}. LoRA constrains optimization to a low-rank subspace of the full parameter space. The intrinsic dimensionality of the task \cite{aghajanyan2021intrinsic} may differ between bias injection and removal, meaning that rank 16 could be sufficient for one direction but insufficient for the other. Full fine-tuning experiments (without LoRA) are needed to determine whether the asymmetry persists in the unconstrained parameter space.

\subsection{Step Boundary Artifacts}

The maximum step limits (500 for injection, 2{,}000 for removal) create an artificial ceiling on $R$ at $4.0 = 2000/500$. Any condition where debiasing does not complete within 2{,}000 steps is assigned $R = 4.0$, regardless of how many additional steps would be needed. This affects primarily the INLP and DAMA methods (Section~\ref{sec:s6}). The survival analysis design (Section~\ref{sec:s3}) addresses this limitation.

\subsection{Evaluation Metric Sensitivity}

CLL and AUL are complementary but not identical metrics. CLL operates on conditional probabilities and is well-suited to causal models. AUL operates on pseudo-log-likelihoods and is designed for encoder models. Comparing $R$ values across architectures involves comparing ratios computed from different base metrics. The relative ordering of models within each architecture group is more reliable than cross-architecture comparisons of absolute $R$ values.

\subsection{Category Granularity}

The 11 bias categories vary in granularity. ``Gender'' is a binary comparison (male/female pronouns), while ``socioeconomic'' covers a broad spectrum of class-related stereotypes. Categories with broader scope may produce more variable $R$ values simply because they aggregate a more heterogeneous set of stereotypes. Per-stereotype $R$ analysis (not performed in this study) could reveal finer-grained patterns.

\subsection{Model Selection}

The 10 tested models were selected for multilingual coverage of English, Hindi, and Bengali. The results may not generalize to monolingual English models, models with different training objectives (e.g., instruction-tuned vs.\ base), or models with substantially different parameter counts. The smallest model tested is mBERT (178M) and the largest is GPT-oss-20B (21B). Behavior at larger scales (70B+) is unknown.

\subsection{Reproducibility}

Random seeds (42, 123, 456) control LoRA initialization and data shuffling. GPU non-determinism in cuDNN operations may introduce minor variation between hardware configurations. All results are reported as means across 3 seeds with 95\% CIs to account for this variance. Code and LoRA checkpoints are available for reproduction.


\bibliographystyle{apalike}
\bibliography{sample}

\end{document}
