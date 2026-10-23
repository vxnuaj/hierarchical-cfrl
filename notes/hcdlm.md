# H-CDLM digest (arXiv 2610.08738, Ollu & Komodakis) — implementation notes

TeX source: `notes/hcdlm_tex/`. Instantiations H-CoBit, H-FLM. Code NOT released (repo URL 404; "upon acceptance").

## Coarse map Γ: vocab → C, |C| = n = 16
- Fixed, offline, never learned. k-means (sklearn, k-means++, n_init=10, max_iter=300, tol 1e-4, seed 42) over pretrained INPUT embeddings (LangFlow 768-d, unit-normalized rows). Minutes on CPU.
- Tokens with no embedding (PAD/UNK/EOS) → reserved "structural" cluster id n-1.
- Random Γ (balanced, no semantics) recovers NONE of the gain. Requirements: deterministic, balanced, semantic.
- n=16 is the only n beating baseline at every λ2; n ∈ {32,64} close, 8/128/256 worse.
- Clusters at n=16 (OWT): function words 16.5%, {to,that,is,'s,are} 13%, {",",and,or} 7.9%, {".",":","?"} 6.6%, past-tense verbs 6.2%, numbers 4.7%, proper nouns 4.2%. Silhouette ~0.05.
- Frequency-driven bit rebalancing only matters for bit encodings (CoBit).

## Continuous encoding of cluster id
- H-FLM: n-dim one-hot concatenated to |V|-dim one-hot. H-CoBit: 4 analog bits appended to token bit patch.
- ELF sketch (their App. E, not run): extra channels + second unembedding head.
- Embedding the cluster INSIDE the token code (no extra channels) is WORSE than baseline → an independent cluster representation is necessary.

## Joint denoising
- State x = [x¹ (tokens), x² (clusters)] concatenated along channels per position; same L; one backbone; per-modality output slices. No extra tokens, no cross-attention.
- Forward: x_σ = x_0 + Σ ε, Σ = blockdiag(σ1 I_{d1}, σ2 I_{d2}). Backbone conditioned on σ1 only (H-CoBit).
- Loss: L = E[(1/L) Σ_i Σ_j λ_j w(σ_j) ||D^j(x_σ)_i − x^j_{0,i}||²]; λ1=1; λ2 = 0.27 LM1B / 0.1 OWT / 0.3 TinyGSM for H-CoBit; 1.0 for H-FLM (where loss is CE, so cluster stream gets n-way CE). λ2 sweep LM1B n=16: optimum 0.05; too large (≥0.5) hurts.
- Training noise: σ1, σ2 coupled through a shared quantile v~U[0,1], σ_j = exp(F_j^{-1}(v)) (default), or independent.
- Per-modality information-uniform schedules: π^j(u) ∝ d/dlogσ H(y_j | x_σ), estimated online from loss histograms. Measured: cluster median noise σ50 = 1.13 vs tokens 0.89 → cluster identity survives more noise, resolves earlier in sampling. (Shared schedule gives implicit coarse-to-fine; per-modality coupled schedule removes it; neither dominates.)

## Sampler
- Host sampler (DDIM ODE + EDM churn + self-cond for CoBit; Euler for FLM). Per-modality churn: separate γ2 and a cluster churn WINDOW. Winning recipe: churn clusters only in the LATE window (low noise, last half of steps), γ2 = 0.3-0.4; early-window churn is worse; magnitude alone does nothing.
- Decoding: drop cluster stream, decode tokens. Prop. 1: if y² = Γ(y¹) deterministic, the token marginal of the joint is p_data → any exact joint sampler is an exact token sampler.
- H-FLM at inference drops the cluster head and uses token-implied cluster distribution (sum of token probs within cluster). For H-CoBit the cluster head is kept; cluster block agrees with the final sample more than the token-implied cluster throughout sampling → cluster head is not redundant.

## Base models
- CoBit-S ~130M (12L, 768, SwiGLU, RoPE, adaLN, bits head). H-CoBit +1% params. 4-8 A100. Overhead +2% train, +3-4% inference.
- FLM 179M (one-hot tokens, CE-trained denoiser, Euler 1024 steps). H-FLM identical hparams, λ2=1.

## Results
- LM1B GenPPL at data entropy: CoBit-S 85.2 (73.6 with s_noise 1.05) → H-CoBit 57.3 → +PMS 49.4 (600k vs 1M steps). OWT: 83.2/71.1 → 67.8 → 50.4. H-FLM vs FLM LM1B NFE 512: 91 vs 102; NFE 1024: 79 vs 111.
- GSM8K (TinyGSM-trained, exact match): AR greedy 63.3; CoBit-S 23.4 (512) / 25.4 (1024); H-CoBit 27.4 (512). FLM 0.3; S-FLM 18.0.
- Nothing on RL, credit assignment, or planning. Coarse stream framed as a semantic anchor that narrows the token posterior at high noise.

## Limitations / future
One level only; clusters fixed (learned clusters future); ~130M models; λ2 highly dataset/n dependent; needs stochastic sampling; cluster CFG untested; applying to ELF/LangFlow/S-FLM listed as future work.

## Code to reuse
- CoBit: github.com/GBATZOLIS/BitstreamDiffusion (MIT), HF gbatzolis/CoBit.
- FLM: github.com/david3684/flm (Apache-2), checkpoints lm1b/owt.
- LangFlow embeddings: github.com/nealchen2003/LangFlow, HF Continuous-Rivals-Discrete/langflow-{lm1b,owt}.
- S-FLM (TinyGSM/GSM8K scripts): github.com/jdeschena/s-flm, HF jdeschena/s-flm.
- HDLM clustering code: github.com/zhouc20/HDLM (hdlm/compute_cluster.py).
