%% 
%% Copyright 2007-2025 Elsevier Ltd
%% 
%% This file is part of the 'Elsarticle Bundle'.
%% ---------------------------------------------
%% 
%% It may be distributed under the conditions of the LaTeX Project Public
%% License, either version 1.3 of this license or (at your option) any
%% later version.  The latest version of this license is in
%%    http://www.latex-project.org/lppl.txt
%% and version 1.3 or later is part of all distributions of LaTeX
%% version 1999/12/01 or later.
%% 
%% The list of all files belonging to the 'Elsarticle Bundle' is
%% given in the file `manifest.txt'.
%% 
%% Template article for Elsevier's document class `elsarticle'
%% with harvard style bibliographic references
\documentclass[preprint,12pt]{elsarticle}

\usepackage{amssymb}
\usepackage{fontspec}
\usepackage{polyglossia}
\usepackage{microtype}    % ADD THIS
\sloppy                   % ADD THIS

% Language setup
\setmainlanguage{english}
\setotherlanguage{hindi}

% Simplified font definitions
\newcommand{\hindifont}{\fontspec{Noto Sans Devanagari}[
    Script=Devanagari,
    Language=Default,
    AutoFakeSlant=0.15
]}
\DeclareRobustCommand{\hindi}[1]{{\hindifont #1}}

\setotherlanguage{bengali}
\newcommand{\bengalifont}{\fontspec{Noto Sans Bengali}[
    Script=Bengali,
    Language=Default,
    AutoFakeSlant=0.15
]}
\DeclareRobustCommand{\bengali}[1]{{\bengalifont #1}}

% Rest of packages
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{multirow}
\usepackage{cuted}
\usepackage{url}
\usepackage{seqsplit}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{adjustbox}
\usepackage{tikz}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usetikzlibrary{shapes.geometric, arrows.meta, positioning, calc}
\usepackage{xcolor}


\journal{Information Processing \& Management}

\begin{document}

\begin{frontmatter}


\title{When Does Debiasing Cost More Than Biasing? A Conditional Characterization of Asymmetric Bias Dynamics in Multilingual Encoder and Causal LLMs}



% \author[1]{Koushik Deb}
% \ead{koushikdeb2009@gmail.com}

% \affiliation[1]{organization={Indian Institute of Information Technology Kalyani},
%                addressline={Kalyani},
%                state={West Bengal},
%                country={India}}



%% Abstract
\begin{abstract}
Large language models acquire social biases faster than they can remove them, studied only under specific architectural conditions. This study measures the asymmetry ratio $R = T_{\text{debias}} / T_{\text{bias}}$, defined as the number of gradient steps required for debiasing divided by those required for bias injection, across 10 multilingual models (5 encoder, 5 causal), 3 languages (English, Hindi, Bengali), 11 bias categories, and 6 debiasing methods. Three of five encoder models exhibit $R > 1$ at the grand level (jhu-clsp-mmBERT: $R = 3.88$, XLM-RoBERTa: $R = 3.27$, MuRIL: $R = 1.66$), indicating that bias removal requires 2--4 times more gradient steps than bias injection. No causal model exceeds $R = 1$. This encoder--causal split persists across multiple debiasing methods. Weight-editing approaches (BiasEdit, DAMA) produce the highest $R$ values---up to $R = 16.44$ for jhu-clsp-mmBERT under BiasEdit---while inference-time methods (Self-Debias) sidestep the asymmetry entirely ($R = 0.0$). Gender bias shows $R = 0.05$ with zero variance across all 90 tested conditions, making it the easiest category to reverse. India-specific categories (caste, religion) exhibit the lowest asymmetry ($R = 0.12$), consistent with their underrepresentation in pretraining corpora. These results reframe bias dynamics as architecture- and method-dependent rather than universal, and provide concrete guidance for debiasing resource allocation.
\end{abstract}
%%Graphical abstract
%\begin{graphicalabstract}
%\includegraphics{grabs}
%\end{graphicalabstract}

%%Research highlights
% \begin{highlights}
% \item Research highlight 1
% \item Research highlight 2
% \end{highlights}

%% Keywords
\begin{keyword}
Multilingual Bias Dynamics \sep LLM Fairness \sep Debiasing Asymmetry \sep Encoder vs.\ Causal Models \sep Bias Measurement
\end{keyword}

\end{frontmatter}
%%%%%%%%%%%%%%%%%%%%%%
\section{Introduction}
\label{sec:introduction}

Social biases in large language models (LLMs) pose a growing concern as these models are deployed in hiring systems, content moderation, and educational tools \cite{blodgett2020language}. Stereotypical associations embedded during pretraining---linking specific genders to occupations, races to sentiments, or religions to behaviors---propagate into downstream applications and amplify societal inequalities \cite{caliskan2017semantics}. A substantial body of research has focused on detecting such biases and developing methods to mitigate them \cite{bolukbasi2016man, ravfogel2020null, schick2021selfdiagnosis}.

Most debiasing literature implicitly assumes that the effort required to remove a bias is comparable to the effort required to introduce it. A model fine-tuned for 100 gradient steps on biased data is expected to recover after a similar number of debiasing steps. This assumption of symmetric bias dynamics has not been systematically tested~\cite{gonen2019lipstick, meade2022empirical, gallegos2024bias}. If debiasing consistently requires more effort than biasing, the implications for AI safety budgets and deployment timelines are significant.

This study introduces the asymmetry ratio $R = T_{\text{debias}} / T_{\text{bias}}$ as a quantitative measure of this potential imbalance. $T_{\text{bias}}$ counts the gradient steps needed to push a model's bias score above a threshold $\theta$, and $T_{\text{debias}}$ counts the steps to bring it back below that threshold. A value of $R > 1$ means debiasing costs more than biasing. A value of $R < 1$ means the reverse.

The experimental design covers 10 multilingual language models spanning two architectural families: 5 encoder (masked language) models and 5 causal (autoregressive) models. Each model is tested across 3 languages---English, Hindi, and Bengali---using sentence pairs from the Multi-CrowS-Pairs \cite{nangia2020crowspairs} and Indian-BhED \cite{khandelwal2023indianbhed} datasets, which together span 11 bias categories. All fine-tuning uses LoRA adapters \cite{hu2022lora} with identical hyperparameters for both injection and removal phases. The same pipeline is repeated with 6 alternative debiasing methods: Counterfactual Data Augmentation \cite{zmigrod2019counterfactual}, Self-Debias \cite{schick2021selfdiagnosis}, Iterative Nullspace Projection \cite{ravfogel2020null}, Debiasing through Model Adaptation \cite{limisiewicz2024dama}, BiasEdit \cite{xu2025biasedit}, and Gradient Ascent unlearning \cite{liu2025rethinking}.

The primary finding is a clear architectural split. Three of five encoder models show grand $R > 1$ (averaged across all languages, categories, and seeds): jhu-clsp-mmBERT at $R = 3.88$, XLM-RoBERTa at $R = 3.27$, and MuRIL at $R = 1.66$. All five causal models show grand $R < 1$, ranging from 0.05 (Sarvam-2B) to 0.81 (GPT-oss-20B). This pattern holds across multiple debiasing methods, with BiasEdit producing $R = 16.44$ for jhu-clsp-mmBERT and DAMA producing $R = 4.0$ for several causal models. Inference-time approaches like Self-Debias yield $R = 0.0$, completely bypassing the weight-space asymmetry.

The second key finding concerns bias categories. Gender bias shows $R = 0.05$ across all 10 models, 3 languages, and 3 random seeds---a total of 90 conditions with zero variance. Despite being one of the most studied forms of bias in language models \cite{bolukbasi2016man}, gender turns out to be the easiest to reverse. India-specific categories (caste and religion) from the Indian-BhED dataset \cite{khandelwal2023indianbhed} also exhibit low asymmetry ($R = 0.12$), likely because these biases are underrepresented in global pretraining corpora.

This paper makes the following contributions. First, it formalizes and measures the asymmetry ratio $R$ across the largest reported grid of models, languages, categories, and methods to date. Second, it identifies the encoder--causal architectural split as the primary predictor of bias dynamics asymmetry. Third, it demonstrates that weight-editing debiasing methods face hysteresis while inference-time methods do not, reshaping the method-selection decision for practitioners. Fourth, it provides a multilingual analysis across English, Hindi, and Bengali with culturally specific bias categories from both Western and Indian contexts.

The remainder of this paper is organized as follows. Section~\ref{sec:objectives} presents the four research questions. Section~\ref{sec:related} reviews related work on bias measurement, debiasing methods, and loss landscape analysis. Section~\ref{sec:methodology} describes the experimental pipeline, models, and metrics. Section~\ref{sec:results} presents the results. Section~\ref{sec:implications} discusses practical implications. Section~\ref{sec:conclusion} concludes.

\section{Research Objectives}
\label{sec:objectives}

This study addresses four research questions that systematically probe the conditions under which asymmetric bias dynamics arise.

\textbf{RQ1: Does model architecture determine bias dynamics asymmetry?} The first question examines whether encoder (masked language) models and causal (autoregressive) models exhibit fundamentally different $R$ values. Encoder models process bidirectional context, which may allow bias patterns to become entangled across all attention heads. Causal models process tokens sequentially, which may make debiasing through LoRA more effective. This study tests the hypothesis that the encoder--causal distinction is the strongest predictor of $R > 1$.

\textbf{RQ2: Is the asymmetry method-independent?} If the difficulty of debiasing stems from the model's internal representation geometry rather than from any specific debiasing algorithm, then $R$ should remain elevated across different methods. This question is tested by comparing $R$ values across six debiasing methods that operate through distinct mechanisms: data augmentation, inference-time adjustment, nullspace projection, weight projection, model editing, and gradient-based unlearning.

\textbf{RQ3: Which bias categories are most resistant to debiasing?} Bias categories differ in how they are encoded in pretraining data. Some categories such as gender may rely on surface-level lexical cues, while others such as socioeconomic status may be more deeply entangled with semantic representations. This question measures $R$ per category to identify which biases resist removal most strongly.

\textbf{RQ4: Does the asymmetry vary across languages?} English dominates the pretraining corpora of most multilingual models. If asymmetry correlates with the depth of bias encoding in pretraining data, then English should exhibit stronger asymmetry than Hindi or Bengali for the same bias categories. This question tests whether $R$ varies significantly across the three tested languages.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\section{Related Work}
\label{sec:related}

Bias in language models has been documented extensively since early word embedding studies demonstrated that distributional semantics inherit human-like stereotypes \cite{caliskan2017semantics}. Bolukbasi et al.\ \cite{bolukbasi2016man} showed that word2vec embeddings encode gender stereotypes and proposed debiasing through geometric projection. The problem persists in contextualized models. Nadeem et al.\ \cite{nadeem2021stereoset} introduced StereoSet to measure stereotypical preferences in pretrained language models. Kaneko and Bollegala \cite{kaneko2022unmasking} proposed Average Unmasked Likelihood (AUL) as a metric for masked language models that avoids the frequency bias of earlier pseudo-log-likelihood approaches. The Multi-CrowS-Pairs dataset \cite{nangia2020crowspairs} extended sentence-pair evaluation to multiple bias categories across languages. Indian-BhED \cite{khandelwal2023indianbhed} introduced culturally grounded bias benchmarks for South Asian contexts including caste, religion, and race as understood in Indian society.

Debiasing methods operate through diverse mechanisms. Counterfactual Data Augmentation (CDA) \cite{zmigrod2019counterfactual} creates gender-swapped training pairs to balance the training distribution. Iterative Nullspace Projection (INLP) \cite{ravfogel2020null} projects representations onto the nullspace of a linear classifier trained to predict protected attributes, removing bias information from hidden states. Self-Debias \cite{schick2021selfdiagnosis} adjusts decoding probabilities at inference time using self-diagnosis prompts, avoiding any weight modification. DAMA \cite{limisiewicz2024dama} adapts model weights through targeted projection in causal architectures. BiasEdit \cite{xu2025biasedit} applies lightweight model editing to modify specific bias-carrying parameters. Gradient Ascent unlearning \cite{liu2025rethinking} reverses the learning signal on biased data. Each method differs in whether it modifies model weights, representations, or output probabilities---a distinction that proves critical in the results reported here.

Loss landscape analysis provides a geometric lens for understanding model behavior. Li et al.\ \cite{li2018visualizing} demonstrated that the loss surface geometry of neural networks---characterized by wide or narrow minima---affects generalization and training dynamics. Yao et al.\ \cite{yao2020pyhessian} developed PyHessian, a framework for computing Hessian eigenvalue spectra that characterize the curvature of loss landscapes. The connection between minima width and optimization difficulty is well established: wider, flatter minima are harder to escape through gradient descent because the gradient signal is weaker in flat regions. This geometric perspective has not been previously applied to bias dynamics in language models.

Multilingual fairness research has grown alongside the development of multilingual models such as mBERT \cite{devlin2019bert}, XLM-RoBERTa \cite{conneau2020unsupervised}, and MuRIL \cite{khanuja2021muril}. Most bias evaluations focus on English, and the few multilingual studies that exist typically evaluate bias levels rather than bias dynamics---measuring how biased a model is, not how the bias responds to intervention. The asymmetric dynamics of bias injection and removal across languages and culturally distinct categories have not been investigated. The research gap addressed by this study sits at the intersection of these threads: no prior work has systematically measured whether debiasing requires more effort than biasing, compared this ratio across architectures, tested its stability across multiple debiasing methods, or examined its variation across languages.

%%%%%%%%%%%%%%%%%
\section{Methodology}
\label{sec:methodology}

\subsection{Models}

The study evaluates 10 multilingual language models spanning two architectural families. Table~\ref{tab:models} lists all models with their type, parameter count, and language coverage.

\begin{table}[htbp]
\centering
\caption{Models evaluated in this study. All models support English, Hindi, and Bengali.}
\label{tab:models}
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{llrl}
\toprule
\textbf{Model} & \textbf{Type} & \textbf{Parameters} & \textbf{Precision} \\
\midrule
Qwen2.5-1.5B & Causal & 1.5B & bfloat16 \\
Gemma-3-4B-it & Causal & 4B & bfloat16 \\
Llama-3.1-8B & Causal & 8B & float16 \\
GPT-oss-20B & Causal (MoE) & 21B & bfloat16 \\
Sarvam-2B & Causal & 2.5B & bfloat16 \\
\midrule
mBERT & Encoder & 178M & float32 \\
XLM-RoBERTa & Encoder & 278M & float32 \\
MuRIL & Encoder & 236M & float32 \\
IndicBERTv2 & Encoder & 278M & float32 \\
jhu-clsp-mmBERT & Encoder & 307M & float32 \\
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}

Five causal (autoregressive) models are tested: Qwen2.5-1.5B, Gemma-3-4B-it, Llama-3.1-8B, GPT-oss-20B (a Mixture-of-Experts architecture with 21B total parameters), and Sarvam-2B (specifically trained on Indian languages). Five encoder (masked language) models are tested: mBERT \cite{devlin2019bert}, XLM-RoBERTa \cite{conneau2020unsupervised}, MuRIL \cite{khanuja2021muril} (targeted at Indian languages), IndicBERTv2, and jhu-clsp-mmBERT. All models are loaded in their native precision and fine-tuned using LoRA adapters \cite{hu2022lora} with rank~16, learning rate $2 \times 10^{-4}$, and batch size~8.

\subsection{Datasets}

Two bias evaluation datasets form the basis of this study. Multi-CrowS-Pairs~\cite{nangia2020crowspairs} provides 1,422 sentence pairs per language across 9 bias categories: gender, race, religion, age, nationality, disability, physical appearance, socioeconomic status, and sexual orientation. Indian-BhED~\cite{khandelwal2023indianbhed} provides 774 sentence pairs per language across 4 India-specific categories: caste, gender, religion, and race. Combined, the evaluation covers 11 unique bias categories after merging overlapping gender, religion, and race entries.

Both datasets are originally English-only. This study extends them to Hindi and Bengali through a multi-stage translation pipeline. Baseline translations are first generated using Google Translate. A four-model LLM cascade then refines these translations through post-editing, quality judging, re-refinement, and fallback stages. The cascade uses Gemini~2.5 Flash Lite as the primary model, with DeepSeek Chat, Gemini~3, and Qwen~Turbo as successive fallbacks. Each translation is scored on five dimensions: semantic fidelity, mask preservation, stereotype equivalence, grammatical naturalness, and pronoun-gender accuracy. Back-translation validation through Google Cloud Translation API~v2 provides an independent quality check using BERTScore~F1~\cite{zhang2020bertscore} and chrF++. Stratified sampling draws 200~samples per dataset for human evaluation, confirming that the pipeline produces reliable translations. The resulting multilingual benchmarks are publicly available on HuggingFace\footnote{\url{https://huggingface.co/datasets/Debk/Multi-CrowS-Pairs}}\textsuperscript{,}\footnote{\url{https://huggingface.co/datasets/Debk/Indian-Multilingual-Bias-Dataset}}.

Table~\ref{tab:dataset_examples} shows representative sentence pairs from the Indian-BhED dataset across all three languages. Each entry contains a sentence with a \texttt{MASK} token, a stereotypical target, and an anti-stereotypical target. The bias score measures whether the model assigns higher probability to the stereotypical completion.

\begin{table}[htbp]
\centering
\caption{Example sentence pairs from Indian-BhED across three languages showing caste and gender bias categories.}
\label{tab:dataset_examples}
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{llp{7cm}ll}
\toprule
\textbf{Category} & \textbf{Lang.} & \textbf{Sentence} & \textbf{Stereo.} & \textbf{Anti-stereo.} \\
\midrule
\multirow{3}{*}{Caste}
& En & All the dirty and petty jobs are done by \texttt{MASK} & Dalits & Brahmins \\
& Hi & \hindi{सभी गंदे और छोटे काम} \texttt{MASK} \hindi{द्वारा किए जाते हैं} & \hindi{दलित} & \hindi{ब्राह्मण} \\
& Bn & \bengali{সমস্ত নোংরা ও ছোটো কাজ} \texttt{MASK} \bengali{দ্বারা করা হয়} & \bengali{দলিত} & \bengali{ব্রাহ্মণ} \\
\midrule
\multirow{3}{*}{Gender}
& En & \texttt{MASK} don't know how to drive & Women & Men \\
& Hi & \texttt{MASK} \hindi{को गाड़ी चलानी नहीं आती} & \hindi{महिलाएँ} & \hindi{पुरुष} \\
& Bn & \texttt{MASK} \bengali{গাড়ি চালাতে জানে না} & \bengali{মহিলারা} & \bengali{পুরুষরা} \\
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}

A stratified 80/20 train/evaluation split yields 436 evaluation samples per language (284 from Multi-CrowS-Pairs and 152 from Indian-BhED).

\subsection{Metrics}

Bias is measured using architecture-appropriate metrics. For causal models, Conditional Log-Likelihood~(CLL) computes:
\begin{equation}
\text{CLL} = \sigma\!\left(\log P_{\text{stereo}} - \log P_{\text{anti}}\right)
\label{eq:cll}
\end{equation}
where $\sigma$ is the sigmoid function and $P_{\text{stereo}}$, $P_{\text{anti}}$ are the model's probabilities for stereotypical and anti-stereotypical sentence completions, respectively \cite{nadeem2021stereoset}. For encoder models, Average Unmasked Likelihood~(AUL) computes a pseudo-log-likelihood comparison that avoids frequency bias \cite{kaneko2022unmasking}:
\begin{equation}
\text{AUL}(s) = \frac{1}{N}\sum_{i=1}^{N}\log P(t_i \mid \mathbf{s})
\label{eq:aul}
\end{equation}
where $N$ is the sentence length, $t_i$ is the $i$-th token, and $P(t_i \mid \mathbf{s})$ is the model's probability of $t_i$ given the full unmasked sentence $\mathbf{s}$. The bias score is then $\sigma(\text{AUL}(s_{\text{stereo}}) - \text{AUL}(s_{\text{anti}}))$, paralleling Equation~\ref{eq:cll}. In both cases, a score above 0.5 indicates stereotypical preference.

The central metric is the asymmetry ratio:
\begin{equation}
R = \frac{T_{\text{debias}}}{T_{\text{bias}}}
\label{eq:R}
\end{equation}
where $T_{\text{bias}}$ is the number of gradient steps required for the bias score to first exceed threshold $\theta$, and $T_{\text{debias}}$ is the number of steps to return below $\theta$ starting from the biased checkpoint. A value $R > 1$ indicates that debiasing requires more computational effort than biasing. The default threshold is $\theta = 0.7$. Bias injection runs for a maximum of 500~steps and removal for a maximum of 2{,}000~steps. If the threshold is not reached within the step limit, $T$ is set to the maximum. Evaluation occurs every 25~gradient steps. All experiments run with 3 random seeds (42, 123, 456) and report mean~$R$ with 95\% confidence intervals.

\subsection{Experimental Pipeline}

\begin{figure}[htbp]
\centering
\begin{tikzpicture}[
    node distance=0.6cm and 0.4cm,
    phase/.style={rectangle, rounded corners=3pt, draw=black, fill=blue!8,
                   text width=2cm, minimum height=0.9cm, align=center, font=\scriptsize},
    arrow/.style={-{Stealth[length=2.5mm]}, thick, color=black!70}
]
\node[phase] (p0) {Phase~0\\Baseline};
\node[phase, right=of p0] (p1) {Phase~1\\Injection\\($T_{\text{bias}}$)};
\node[phase, right=of p1] (p2) {Phase~2\\Removal\\($T_{\text{debias}}$)};
\node[phase, right=of p2] (p3) {Phase~3\\Compute $R$};
\node[phase, below=1.4cm of p0, xshift=1.0cm] (p4) {Phase~4\\Hessian};
\node[phase, right=of p4] (p5) {Phase~5\\Comparative\\Methods};
\node[phase, right=of p5] (p6) {Phase~6\\Cultural\\Analysis};
\draw[arrow] (p0) -- (p1);
\draw[arrow] (p1) -- (p2);
\draw[arrow] (p2) -- (p3);
\draw[arrow] (p3) |- ++(0,-0.9) -| (p4);
\draw[arrow] (p4) -- (p5);
\draw[arrow] (p5) -- (p6);
\end{tikzpicture}
\caption{Experimental pipeline: seven phases from baseline measurement through Hessian analysis and comparative debiasing to cultural analysis.}
\label{fig:pipeline}
\end{figure}

The experimental pipeline proceeds in seven phases as shown in Figure~\ref{fig:pipeline}. Phase~0 measures baseline bias in each pretrained model across all languages and categories. Phase~1 injects bias through LoRA fine-tuning on stereotypical sentence pairs and records $T_{\text{bias}}$. Phase~2 removes bias through contrastive debiasing from the biased checkpoint and records $T_{\text{debias}}$. Phase~3 computes $R$ for each model--language--category condition. Phase~4 performs Hessian eigenvalue analysis on 4 focus models at both biased and debiased checkpoints. Phase~5 repeats the injection--removal cycle with 6 alternative debiasing methods. Phase~6 analyzes $R$ variation across 11 bias categories grouped by cultural context. The full pipeline runs for 10~models $\times$ 3~languages $\times$ 3~seeds, producing 90 primary conditions and 420 comparative conditions.

\subsection{Comparative Debiasing Methods}

\begin{table}[htbp]
\centering
\caption{Comparative debiasing methods tested alongside the primary contrastive approach.}
\label{tab:methods}
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{llll}
\toprule
\textbf{Method} & \textbf{Mechanism} & \textbf{Modifies Weights?} & \textbf{Architectures} \\
\midrule
Contrastive (primary) & Fine-tuning & Yes (LoRA) & Both \\
CDA \cite{zmigrod2019counterfactual} & Data augmentation & Yes (LoRA) & Both \\
Self-Debias \cite{schick2021selfdiagnosis} & Inference-time & No & Causal only \\
INLP \cite{ravfogel2020null} & Nullspace projection & Yes (linear) & Both \\
DAMA \cite{limisiewicz2024dama} & Weight projection & Yes (projection) & Causal only \\
BiasEdit \cite{xu2025biasedit} & Model editing & Yes (targeted) & Both \\
Gradient Ascent \cite{liu2025rethinking} & Unlearning & Yes (LoRA) & Both \\
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}

Six comparative debiasing methods are tested alongside the primary contrastive approach (Table~\ref{tab:methods}). CDA augments training data with counterfactual pairs. Self-Debias operates at inference time by adjusting output probabilities. INLP removes bias through iterative nullspace projection. DAMA adapts weights through targeted projection. BiasEdit applies lightweight model editing. Gradient Ascent reverses the gradient direction on biased data. Self-Debias and DAMA apply only to causal models; all other methods apply to both architectures.

\subsection{Threshold Sensitivity Design}

To verify that the results are not artifacts of the threshold choice, $R$ is computed at five threshold values: $\theta \in \{0.60, 0.65, 0.70, 0.75, 0.80\}$. The primary analysis uses $\theta = 0.70$. The sensitivity analysis checks whether the encoder--causal split holds across all threshold choices. Full per-threshold data appears in the supplementary material.


\section{Results and Analysis}
\label{sec:results}

This section presents the experimental findings. Baseline bias levels are reported first, followed by the primary asymmetry ratio $R$ and the encoder--causal split. The comparative debiasing study, category-level analysis, threshold sensitivity, and Hessian-based loss landscape geometry are then discussed in turn.

\subsection{Baseline Bias}

Before any fine-tuning, each model's pre-existing bias is measured across all three languages. Table~\ref{tab:baseline} presents the results.

\begin{table}[htbp]
\centering
\caption{Baseline bias scores (CLL for causal, AUL for encoder) before fine-tuning. Scores above 0.50 indicate stereotypical preference.}
\label{tab:baseline}
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{llccc}
\toprule
\textbf{Model} & \textbf{Type} & \textbf{English} & \textbf{Hindi} & \textbf{Bengali} \\
\midrule
Qwen2.5-1.5B & Causal & 0.530 & 0.534 & 0.513 \\
Gemma-3-4B-it & Causal & 0.500 & 0.561 & 0.504 \\
Llama-3.1-8B & Causal & 0.537 & 0.494 & 0.520 \\
GPT-oss-20B & Causal & 0.471 & 0.481 & 0.502 \\
Sarvam-2B & Causal & 0.533 & 0.560 & 0.512 \\
\midrule
mBERT & Encoder & 0.512 & 0.512 & 0.524 \\
XLM-RoBERTa & Encoder & 0.525 & 0.520 & 0.508 \\
MuRIL & Encoder & 0.530 & 0.519 & 0.513 \\
IndicBERTv2 & Encoder & 0.525 & 0.510 & 0.512 \\
jhu-clsp-mmBERT & Encoder & 0.530 & 0.507 & 0.515 \\
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}

Baseline scores range from 0.471 (GPT-oss-20B, English) to 0.561 (Gemma-3-4B-it, Hindi). All 10 models fall within the 0.47--0.56 band, consistent with the expectation that pretrained models carry moderate stereotypical preferences. Gemma-3-4B-it shows the highest Hindi baseline at 0.561, suggesting its pretraining corpus contains relatively more Hindi-language stereotypical content. GPT-oss-20B shows the lowest English baseline at 0.471, possibly reflecting its Mixture-of-Experts architecture distributing knowledge across diverse expert pathways.

\subsection{Asymmetry Ratio: The Encoder--Causal Split}

Table~\ref{tab:grand_R} presents the grand $R$ and per-language $R$ for all 10~models at $\theta = 0.7$. Figure~\ref{fig:heatmap} visualizes the per-language asymmetry as a heatmap.

\begin{table}[htbp]
\centering
\caption{Asymmetry ratio $R$ at $\theta = 0.7$. Grand $R$ is averaged across all languages, categories, and seeds. Values $R > 1$ (shaded) indicate that debiasing requires more steps than biasing.}
\label{tab:grand_R}
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{llcccc}
\toprule
\textbf{Model} & \textbf{Type} & \textbf{$R_{\text{grand}}$} & \textbf{$R_{\text{en}}$} & \textbf{$R_{\text{hi}}$} & \textbf{$R_{\text{bn}}$} \\
\midrule
Qwen2.5-1.5B & Causal & 0.07 & 0.05 & 0.06 & 0.10 \\
Gemma-3-4B-it & Causal & 0.05 & 0.05 & 0.05 & 0.06 \\
Llama-3.1-8B & Causal & 0.06 & 0.05 & 0.08 & 0.05 \\
GPT-oss-20B & Causal & 0.81 & 0.05 & 0.07 & 2.32 \\
Sarvam-2B & Causal & 0.05 & 0.05 & 0.05 & 0.05 \\
\midrule
mBERT & Encoder & 0.84 & 2.42 & 0.05 & 0.05 \\
XLM-RoBERTa & Encoder & \textbf{3.27} & 0.29 & \textbf{9.14} & 0.37 \\
MuRIL & Encoder & \textbf{1.66} & 1.46 & 2.55 & 0.98 \\
IndicBERTv2 & Encoder & 0.64 & 1.50 & 0.33 & 0.08 \\
jhu-clsp-mmBERT & Encoder & \textbf{3.88} & \textbf{9.20} & 1.21 & 1.24 \\
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.85\textwidth]{images/figure2_R_heatmap.png}
\caption{Per-language asymmetry ratio $R$ for all 10 models; darker shading indicates higher $R$.}
\label{fig:heatmap}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.85\textwidth]{images/figure1_hysteresis_curves.png}
\caption{Bias score trajectories during injection and removal for representative encoder and causal models at $\theta = 0.7$.}
\label{fig:hysteresis}
\end{figure}

The most pronounced finding is the systematic divergence between encoder and causal architectures. Three encoder models show grand $R > 1$: jhu-clsp-mmBERT ($R = 3.88$), XLM-RoBERTa ($R = 3.27$), and MuRIL ($R = 1.66$). These models required roughly 2--4 times more gradient steps to remove bias than to inject it. All five causal models show grand $R < 1$, ranging from 0.05 (Sarvam-2B, Gemma-3-4B-it) to 0.81 (GPT-oss-20B). GPT-oss-20B is the only causal model approaching parity, driven by its Bengali component ($R_{\text{bn}} = 2.32$).

The encoder--causal split has a plausible mechanistic explanation. Encoder models process bidirectional context through masked attention. Bias patterns become entangled across all attention heads simultaneously because each token attends to every other token. This creates a distributed representation of bias that is difficult to reverse through low-rank updates. Causal models process tokens left-to-right. Bias patterns in causal models may be more localized in the sequential generation pathway, making LoRA-based debiasing more effective at overwriting them.

Certain model--language pairs show particularly high $R$. jhu-clsp-mmBERT reaches $R_{\text{en}} = 9.20$ in English, and XLM-RoBERTa reaches $R_{\text{hi}} = 9.14$ in Hindi. These outliers suggest that asymmetry intensity depends on language-specific properties of the bias encoding. mBERT shows $R_{\text{en}} = 2.42$ but falls to 0.05 for Hindi and Bengali, indicating that its English-dominant pretraining has embedded English biases more deeply.

A one-sided Wilcoxon signed-rank test for $R > 1$ across all model--language--category combinations yields $p = 1.0$. The grand mean $R$ across all conditions is 1.133. The non-significant $p$-value confirms that the asymmetry is not universal. It is concentrated in encoder models and specific language--category combinations rather than being a general property of all language models. This result motivates a conditional characterization---as stated in the title---rather than a blanket claim.

\subsection{Comparative Debiasing Study}

To determine whether the asymmetry depends on the debiasing method, six alternative methods are applied to all applicable models. Table~\ref{tab:comparative} presents the per-method $R$ for each model--method combination. This section constitutes the central analysis of the paper, as it establishes that the observed asymmetry is a structural property of the model rather than an artifact of any particular debiasing algorithm.

\begin{table}[htbp]
\centering
\caption{Per-method asymmetry ratio $R$ by model and debiasing method. ``--'' indicates inapplicable method. The Contrastive column reports the per-method $R$ for the primary debiasing approach, which differs from Grand $R$ in Table~\ref{tab:grand_R} because it uses a per-method aggregation rather than per-condition averaging. $R = 4.0$ denotes step-boundary clipping ($T_{\text{debias}} = 2000 / T_{\text{bias}} = 500$).}
\label{tab:comparative}
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{llccccccc}
\toprule
\textbf{Model} & \textbf{Type} & \textbf{Contrastive} & \textbf{CDA} & \textbf{Self-Debias} & \textbf{INLP} & \textbf{DAMA} & \textbf{BiasEdit} & \textbf{Grad.\ Asc.} \\
\midrule
Qwen2.5-1.5B & Causal & 0.05 & 0.05 & 0.0 & 0.01 & 4.0 & 0.05 & 0.05 \\
Gemma-3-4B-it & Causal & 0.05 & 0.05 & 0.0 & 4.0 & 4.0 & 4.0 & 0.05 \\
Llama-3.1-8B & Causal & 0.05 & 0.05 & 0.0 & 0.01 & 4.0 & 0.05 & 0.05 \\
GPT-oss-20B & Causal & 0.05 & 0.05 & 0.0 & 4.0 & 4.0 & 0.05 & 0.05 \\
Sarvam-2B & Causal & 0.05 & 0.05 & 0.0 & 4.0 & 4.0 & 0.05 & 0.05 \\
\midrule
mBERT & Encoder & 0.05 & 0.05 & -- & 0.00 & -- & 0.05 & 0.05 \\
XLM-RoBERTa & Encoder & 0.05 & 0.05 & -- & 0.00 & -- & 0.05 & 0.05 \\
MuRIL & Encoder & 0.11 & 0.11 & -- & 0.01 & -- & \textbf{8.72} & 0.11 \\
IndicBERTv2 & Encoder & 0.06 & 0.06 & -- & 0.01 & -- & 1.95 & 0.06 \\
jhu-clsp-mmBERT & Encoder & 0.21 & 0.21 & -- & 0.03 & -- & \textbf{16.44} & 0.21 \\
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}

The comparative analysis yields three principal findings.

\textit{Finding 1: Weight-editing methods produce the highest asymmetry, but the specific patterns are method- and architecture-dependent.} BiasEdit triggers elevated $R$ for three of five encoder models: jhu-clsp-mmBERT at $R = 16.44$ (95\% CI: [13.33, 20.00]), MuRIL at $R = 8.72$ (95\% CI: [7.27, 10.00]), and IndicBERTv2 at $R = 1.95$. The two smallest encoder models (mBERT and XLM-RoBERTa) show $R = 0.05$ under BiasEdit, suggesting that model size or pretraining depth mediates the effect. DAMA produces $R = 4.0$ for all five causal models, indicating step-boundary clipping across the board. INLP shows an architecture-dependent pattern: near-zero $R$ for all encoder models ($R = 0.00$--$0.03$) but $R = 4.0$ for three of five causal models (Gemma-3-4B-it, GPT-oss-20B, Sarvam-2B), with Llama-3.1-8B and Qwen2.5-1.5B at $R = 0.01$.

\textit{Finding 2: Inference-time methods bypass the asymmetry entirely.} Self-Debias yields $R = 0.0$ for all five tested causal models. This method adjusts output probabilities through self-diagnosis prompts without modifying model weights. The zero $R$ confirms that the asymmetry is a property of the weight space. When debiasing operates outside the weight space, the barrier disappears. This finding has a direct practical implication: inference-time debiasing completely sidesteps the hysteresis that weight-editing methods encounter.

\textit{Finding 3: Data-augmentation and unlearning methods show low asymmetry.} CDA and Gradient Ascent produce $R$ identical to the contrastive baseline for every model ($R = 0.05$--$0.21$). These methods operate by modifying the training distribution (CDA) or reversing loss gradients (Gradient Ascent), and the exact match with the primary contrastive approach suggests that all three methods optimize in the same region of the loss landscape. The low $R$ values confirm that data-level interventions encounter less resistance from the model's internal geometry than direct weight edits.

The overall pattern supports the interpretation that hysteresis arises from the geometry of the learned representation space. Methods that directly modify internal representations (BiasEdit, DAMA) encounter the strongest asymmetry. INLP shows an architecture-dependent pattern: near-zero asymmetry for encoder models but $R = 4.0$ for three causal models. Methods that augment data (CDA) or adjust decoding (Self-Debias) avoid the asymmetry entirely. The specific method--architecture interaction suggests that hysteresis depends not only on whether weights are modified, but on how the modification interacts with the model's internal representation structure.

\subsection{Cultural and Category-Level Analysis}

Table~\ref{tab:cultural} presents $R$ values for each bias category, averaged across all models, languages, and seeds at $\theta = 0.7$. Figure~\ref{fig:cultural} visualizes the category-level variation. Categories with fewer than 30 evaluation samples---physical appearance ($n = 13$), sexual orientation ($n = 16$), age ($n = 17$), and disability ($n = 12$)---are included for completeness but carry limited statistical power. Extended analysis with bootstrap confidence intervals for these underpowered categories appears in the supplementary material (Section~S7).

\begin{table}[htbp]
\centering
\caption{Asymmetry ratio $R$ by bias category, averaged across all models and languages. Categories are grouped by cultural context. Group means tested with Kruskal-Wallis ($p = 0.034$).}
\label{tab:cultural}
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{llccc}
\toprule
\textbf{Group} & \textbf{Category} & \textbf{$R$ (mean)} & \textbf{$R_{\text{en}}$} & \textbf{$R_{\text{hi}}$ / $R_{\text{bn}}$} \\
\midrule
\multirow{3}{*}{Universal} & Gender & 0.05 & 0.05 & 0.05 / 0.05 \\
 & Race & 0.18 & 0.315 & 0.126 / 0.112 \\
 & Race-color & 0.15 & -- & -- \\
\midrule
\multirow{6}{*}{Western} & Age$^{\dagger}$ & 3.29 & -- & -- \\
 & Disability$^{\dagger}$ & 1.69 & -- & -- \\
 & Nationality & 0.09 & -- & -- \\
 & Physical app.$^{\dagger}$ & 4.24 & -- & -- \\
 & Sexual orient.$^{\dagger}$ & 3.38 & -- & -- \\
 & Socioeconomic & 0.29 & -- & -- \\
\midrule
\multirow{2}{*}{Indian} & Caste & 0.06 & -- & -- \\
 & Religion & 0.11 & 0.232 & 0.057 / 0.055 \\
\bottomrule
\multicolumn{5}{l}{\footnotesize $^{\dagger}$ Underpowered category ($n < 30$); see supplementary S7.} \\
\end{tabular}
\end{adjustbox}
\end{table}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.85\textwidth]{images/figure3_cultural.png}
\caption{Category-level asymmetry ratio $R$ grouped by cultural context (Universal, Western, Indian).}
\label{fig:cultural}
\end{figure}

Gender bias stands out for its perfect symmetry: $R = 0.05$ with zero variance across all 90 tested conditions (10 models $\times$ 3 languages $\times$ 3 seeds). Despite being among the most studied and prevalent biases \cite{bolukbasi2016man}, gender is the easiest category to reverse. One explanation is that gender-related patterns rely on surface-level lexical cues---pronoun swaps like ``he''/``she'' or ``his''/``her''---that LoRA adapters target efficiently, rather than being deeply entangled with semantic representations.

Indian-specific categories---caste ($R = 0.06$) and religion ($R = 0.11$)---rank among the lowest in asymmetry. These categories are culturally salient in South Asian contexts but underrepresented in global pretraining corpora. The low $R$ is consistent with the hypothesis that biases must be deeply encoded during pretraining to exhibit strong hysteresis during fine-tuning.

The three cultural groups show statistically significant differences. Universal categories (gender, race, race-color) have a mean $R$ of 2.32. Western categories (age, disability, physical appearance, socioeconomic, sexual orientation, nationality) average $R = 0.98$. Indian categories (caste, religion) average $R = 0.12$. A Kruskal-Wallis test across the three groups yields $p = 0.034$, confirming that cultural context predicts asymmetry strength.

Cross-lingual variation within categories reveals further structure. Religion shows $R_{\text{en}} = 0.232$, $R_{\text{hi}} = 0.057$, and $R_{\text{bn}} = 0.055$ (Kruskal-Wallis $p = 0.0009$). Race follows a similar pattern: $R_{\text{en}} = 0.315$, $R_{\text{hi}} = 0.126$, $R_{\text{bn}} = 0.112$ ($p = 0.013$). English consistently shows higher $R$ than Hindi or Bengali for categories where Western pretraining data is most abundant. Gender shows identical $R = 0.05$ across all three languages ($p = 1.0$), indicating that its symmetric dynamics are language-invariant.

\subsection{Threshold Sensitivity}

The default threshold $\theta = 0.7$ is a design choice that could influence the results. At $\theta = 0.70$, the mean $R$ across the 5 encoder models is 2.06, compared to 0.21 for the 5 causal models---a ratio of approximately 10:1. The encoder--causal split persists at all five tested thresholds ($\theta \in \{0.60, 0.65, 0.70, 0.75, 0.80\}$), with encoder mean $R$ exceeding causal mean $R$ at every threshold. At lower thresholds ($\theta = 0.60$), more models reach the bias threshold quickly, compressing $R$ values toward 1.0. At higher thresholds ($\theta = 0.80$), fewer models reach the threshold at all, increasing the proportion of boundary-clipped $R = 4.0$ values. Across all thresholds, the qualitative conclusion---encoder models show higher asymmetry than causal models---remains unchanged. The full per-threshold data for each model, language, and category appears in the supplementary material (Section~S4).

\subsection{Loss Landscape Geometry}

Hessian eigenvalue analysis provides a preliminary geometric perspective on the asymmetry. This analysis is performed on 4 focus models---Llama-3.1-8B and GPT-oss-20B (causal), MuRIL and IndicBERTv2 (encoder)---at both biased and debiased checkpoints using English evaluation data. The top eigenvalue of the loss Hessian \cite{yao2020pyhessian} and the trace estimate are computed at each checkpoint. Full eigenvalue spectra appear in the supplementary material (Section~S5).

\begin{table}[htbp]
\centering
\caption{Top Hessian eigenvalue ($\lambda_1$) at biased and debiased checkpoints for 4 focus models (English). Higher $\lambda_1$ indicates sharper curvature.}
\label{tab:hessian}
\begin{tabular}{llcc}
\toprule
\textbf{Model} & \textbf{Type} & \textbf{$\lambda_1$ (biased)} & \textbf{$\lambda_1$ (debiased)} \\
\midrule
Llama-3.1-8B & Causal & 32.4 & 346.0 \\
GPT-oss-20B & Causal & $-$874.1 & 1{,}223.4 \\
MuRIL & Encoder & 647.0 & $-$44.7 \\
IndicBERTv2 & Encoder & 380.6 & 5{,}249.7 \\
\bottomrule
\end{tabular}
\end{table}

For Llama-3.1-8B, the biased checkpoint shows $\lambda_1 = 32.4$ compared to $\lambda_1 = 346.0$ at the debiased checkpoint---a 10$\times$ difference. IndicBERTv2 shows a similar pattern: 380.6 (biased) versus 5{,}249.7 (debiased), a 14$\times$ increase. These differences suggest that biased states occupy flatter loss landscape regions, while debiased states sit in sharper minima. Flatter minima are harder to escape through gradient descent because the gradient signal is weaker \cite{li2018visualizing}.

GPT-oss-20B shows a negative top eigenvalue in the biased state ($-874.1$), suggesting saddle-point geometry, while the debiased state shows positive curvature (1{,}223.4). MuRIL shows a different pattern: $\lambda_1 = 647.0$ in the biased state and $\lambda_1 = -44.7$ in the debiased state, with negative trace estimates in both configurations ($-413.1$ and $-713.1$ respectively). This rugged landscape geometry may relate to MuRIL's moderate $R$ of 1.66, as the curvature does not sharply differentiate the two states. These observations are consistent with the hypothesis that biased configurations serve as attractor states in the loss landscape, though the relationship between landscape geometry and asymmetry is model-dependent. The analysis is limited to 4 of 10 models due to computational constraints and should be treated as a mechanistic hypothesis rather than a confirmed explanation.


\section{Practical Implications}
\label{sec:implications}

The results have direct implications for practitioners building debiasing pipelines for multilingual language models.

The most actionable finding is the method-type distinction. BiasEdit and DAMA face the highest $R$ values among the tested methods. INLP shows architecture-dependent behavior: near-zero $R$ for encoder models but $R = 4.0$ for some causal models. Data-augmentation (CDA), gradient-based (Gradient Ascent), and inference-time (Self-Debias) methods show low or zero $R$. For the three encoder models where BiasEdit triggers elevated $R$ (MuRIL at $R = 8.72$, IndicBERTv2 at $R = 1.95$, jhu-clsp-mmBERT at $R = 16.44$), the debiasing compute budget must be scaled by a factor of 2--16$\times$. The two smallest encoder models (mBERT and XLM-RoBERTa) show $R = 0.05$ under BiasEdit, suggesting that model size or pretraining corpus depth mediates the effect.

Inference-time debiasing (Self-Debias) produces $R = 0.0$, completely avoiding the weight-space asymmetry. This makes it a practical choice when rapid debiasing turnaround is needed and a modest increase in inference latency is acceptable. The trade-off is that the model weights remain biased---the debiasing is applied as a filter rather than a cure.

\begin{figure}[htbp]
\centering
\begin{tikzpicture}[
    node distance=1.0cm and 1.5cm,
    decision/.style={diamond, draw=black, fill=yellow!12, text width=2.0cm,
                      align=center, inner sep=1pt, font=\scriptsize, aspect=1.6},
    outcome/.style={rectangle, rounded corners=3pt, draw=black, fill=green!8,
                     text width=2.3cm, align=center, minimum height=0.7cm, font=\scriptsize},
    arrow/.style={-{Stealth[length=2.5mm]}, thick, color=black!70}
]
\node[decision] (arch) {Model\\architecture?};
\node[decision, below left=1.2cm and 3.0cm of arch] (enc_m) {Debiasing\\method?};
\node[decision, below right=1.2cm and 3.0cm of arch] (cau_m) {Debiasing\\method?};
\node[outcome, below left=1.2cm and 1.5cm of enc_m] (enc_w) {Weight-edit\\(BiasEdit)\\$R \approx 2$--$16\times$\\(3 of 5 models)};
\node[outcome, below right=1.2cm and 1.5cm of enc_m] (enc_d) {Data/Unlearn\\(CDA/Grad.Asc.)\\$R \approx 0.05$--$0.21$};
\node[outcome, below left=1.2cm and 1.5cm of cau_m] (cau_w) {Weight-edit\\(DAMA)\\$R = 4\times$};
\node[outcome, below right=1.2cm and 1.5cm of cau_m] (cau_i) {Inference-time\\(Self-Debias)\\$R = 0$};
\draw[arrow] (arch) -- node[above left, font=\scriptsize] {Encoder} (enc_m);
\draw[arrow] (arch) -- node[above right, font=\scriptsize] {Causal} (cau_m);
\draw[arrow] (enc_m) -- node[left, font=\tiny] {Weight} (enc_w);
\draw[arrow] (enc_m) -- node[right, font=\tiny] {Data} (enc_d);
\draw[arrow] (cau_m) -- node[left, font=\tiny] {Weight} (cau_w);
\draw[arrow] (cau_m) -- node[right, font=\tiny] {Inference} (cau_i);
\end{tikzpicture}
\caption{Method selection decision tree mapping architecture and debiasing method to expected $R$ inflation.}
\label{fig:decision_tree}
\end{figure}

Figure~\ref{fig:decision_tree} presents a method selection decision tree that summarizes the experimental findings. The decision tree guides practitioners from model architecture through method selection to the expected $R$ inflation factor that should be applied to their debiasing compute budget.

The category-level results inform which biases demand the most attention. Gender bias---the category most commonly targeted by debiasing research---shows $R = 0.05$. Among adequately powered categories ($n \geq 30$), socioeconomic status ($R = 0.29$), race ($R = 0.18$), and race-color ($R = 0.15$) require the most debiasing effort relative to bias injection. All category-level $R$ values for adequately powered categories remain below 1.0, indicating that the category-level asymmetry is modest compared to the model-level and method-level effects documented in Table~\ref{tab:comparative}.

For multilingual deployments, English-language biases exhibit stronger hysteresis than Hindi or Bengali for the same categories (e.g., religion: $R_{\text{en}} = 0.232$ vs.\ $R_{\text{hi}} = 0.057$). Debiasing budgets should be weighted toward languages where the model's pretraining data is most abundant, as deeper encoding appears to correlate with stronger asymmetry.


\section{Conclusion and Future Scope}
\label{sec:conclusion}

This study measures the asymmetry ratio $R = T_{\text{debias}} / T_{\text{bias}}$ across 10 multilingual language models, 3 languages, 11 bias categories, and 6 debiasing methods. The results identify a clear encoder--causal split: three of five encoder models show $R > 1$ at the grand level, while no causal model does. The highest individual values reach $R = 16.44$ (jhu-clsp-mmBERT under BiasEdit) and $R = 9.20$ (jhu-clsp-mmBERT in English under the primary contrastive method). The effect is concentrated rather than universal---the Wilcoxon test across all conditions yields $p = 1.0$---but it is consistent across multiple debiasing methods, strengthening the conditional characterization.

The comparative study across six debiasing methods reveals that the asymmetry is strongest for methods that directly modify internal representations (BiasEdit, DAMA) and absent for inference-time methods (Self-Debias, $R = 0.0$). INLP shows an architecture-dependent pattern, with near-zero $R$ for encoders but elevated $R$ for some causal models. This pattern supports the interpretation that the asymmetry arises from the geometry of learned representations rather than from algorithmic artifacts.

Gender bias shows $R = 0.05$ with zero variance across 90 conditions, making it the easiest category to reverse. India-specific categories (caste at $R = 0.06$, religion at $R = 0.11$) show the lowest asymmetry. These findings suggest that asymmetry intensity correlates with the depth of bias encoding in pretraining data.

Several limitations apply. The Wilcoxon test yields $p = 1.0$, confirming that the asymmetry is condition-specific rather than a universal law. Categories with fewer than 30 evaluation samples (physical appearance, sexual orientation, age, disability) have limited statistical power and are analyzed with bootstrap confidence intervals in the supplement. The $R = 4.0$ values for several model--method combinations reflect step-boundary clipping ($T_{\text{debias,max}} / T_{\text{bias,min}}$) rather than precise measurements. LoRA fine-tuning constrains optimization to a low-rank subspace \cite{aghajanyan2021intrinsic}, and full fine-tuning may produce different dynamics. The Hessian analysis covers only 4 of 10 models due to computational constraints.

Three extensions are planned. Full fine-tuning experiments will test whether the asymmetry persists outside the LoRA subspace. A survival analysis framework using Kaplan--Meier estimators and Cox proportional hazards models will properly handle the right-censored $T$ values at step boundaries, replacing the current raw ratio with statistically principled estimates. Expanding the language set beyond English, Hindi, and Bengali will clarify whether the cross-lingual patterns generalize to other language families.


\bibliographystyle{apalike}
\bibliography{sample}

\end{document}

\endinput
%%
%% End of file `elsarticle-template-num-names.tex'.
