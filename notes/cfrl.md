# CFRL digest (arXiv 2610.19080, Han, Hu, Liu; Tufts) — implementation notes

TeX source: `notes/cfrl_tex/`. Method: CFRL-SVG-2D (recommended), CFRL-SVG-1D, CFRL-GRR (score-function baseline). Code NOT released.

## Base model
- ELF-B 105M (arXiv 2605.10938), PyTorch port, OWT checkpoint start. Frozen T5-small encoder embeddings d=512. Decoder: linear W to vocab logits, softmax per position. x-prediction, self-conditioning (prev clean estimate as input), CFG ω=2 at inference (2 evals per call), SC guidance 1.
- Full-sequence, fixed length per task; prompt positions held fixed. Lengths: Countdown 72 (16 cond), Sudoku 164 (82/82), K&K 96 (56 cond), Game24 72 (16 cond).
- Task SFT: ELF recipe, AdamW lr 2e-3 w/ warmup, global batch 256-512, EMA 0.999 (0.99 Game24). Checkpoints: Countdown epoch 60/100 on 500k ex; Sudoku converged (76 ep, 400k); K&K epoch 4 (early on purpose; 300k); Game24 epoch 20/40 init from Countdown model (13,150 orderings of 1,076 sets).

## Sampler (ELF churn SDE; Appendix B.1, from ELF code)
Step h, t_k = t_{k-1}+h, t_eps=0.05, churn γ. α = clip(1-γh, 0, 1); t̃_{k-1} = α t_{k-1}; D = max(1-t̃_{k-1}, t_eps).
One step: z̃_{k-1} = α z_{k-1} + (1-α) ε; x_{k-1} = d_θ(z̃_{k-1}, t̃_{k-1}, x_{k-2}, c); v = (x_{k-1} - z̃_{k-1})/D; z_k = z̃_{k-1} + (t_k - t̃_{k-1}) v; z̃_k = α z_k + (1-α) ε.
Written as z_{t_k} = λ_k z_{t_{k-1}} + η_k x_{t_{k-1}} + ψ_k ε with
λ_k = α(1 - (t+h-t̃)/D), η_k = α (t+h-t̃)/D, ψ_k = 1-α (k<K); λ_K = 1-(t+h-t̃)/D, η_K = (t+h-t̃)/D, ψ_K = 0.
Denoiser is queried at lowered time t̃_k = α t_k. Need α>0.
Deployment: γ56@64 (K=64 uniform), γ28@32 on Sudoku. Cheap: γ4@16 (γ14@16 Sudoku). Per-step retention α≈0.125 best regardless of K.
Transition kernel per position: N(λ z + η x, ψ² c_σ² I).

## MDP
s_k = (z_{t_k}, x_{t_{k-1}}); s_0 = (ε, 0); terminal s_{K+1} = decoded tokens w. a_k = (x_{t_k}, ε) for k<K; a_K = w with π(a_K|s_K) = Π_i softmax(W z_{t_K}[i])_{w_i}. Reward R ∈ {0,1} from verifier. V*_k(s_k) = P(R=1 | s_k).

## CFRL-SVG (pathwise) — core
Objective per sampled step: J = E[V_φ(s_k)] where s_k = (λ z_{k-1} + η d_θ(s_{k-1}) + ψ ε, d_θ(s_{k-1})). Roll to s_{k-1} under no_grad; ONE denoiser call with grad; critic backward. Gradient = [η ∇_z V + ∇_x V] ∂d_θ/∂θ (two channels; ~77-79% of norm via the self-cond channel x). No backprop through earlier/later steps. Memory independent of K.
- k ~ U([K/8, 5K/8]): {8..40} of 64; {4..20} of 32.
- **Actor ascends the critic's PRE-SIGMOID LOGIT** (sum over positions of f_φ(h_i)); ascending sigmoid prob costs -4/-6 pts. Mean-pool not min-pool (min costs -3).
- No baseline, no advantage, no KL, no clipping. Drift controlled by small actor lr and critic refit.
- AdamW, grad-clip 1.0; AdamW per-coordinate normalization is load-bearing (raw step destroys policy).

## Critic
- Backbone: full copy of task-SFT denoiser, fine-tuned during critic PRE-training (lr 1e-5), FROZEN during RL; evaluated without CFG; reads (z_{t_k}, t̃_k, x_{t_{k-1}}, c) exactly like the sampler's call; returns per-position features h_i.
- Head f_φ: MLP (d+1)→256→64→1, GELU, extra input t.
- 2D: r_{k,i} = sigmoid(f_φ(h_i)); V_φ = sigmoid(Σ_i f_φ(h_i)) over generated positions. 1D: sigmoid(f_φ(mean h)).
- Loss 2D: BCE(V_φ, R) + (1/L) Σ_i BCE(r_{k,i}, R). All per-position readings trained on the SEQUENCE reward.
- Pre-training: rollouts of SFT policy, store states at several steps labelled by terminal reward, train backbone+head jointly, batch 64, pick epoch by "field score" (½ Pearson(min-position reading, R) + ½ within-prompt rank-corr(V, R)).
  Countdown: states from γ4@16, 64k rollouts, steps 1,2,4,6,8,10,12,14; 12 epochs (kept 9 for 2D). Sudoku: γ28@32, 32k, steps 1,6,12,18,24. K&K: γ56@64, 32k, steps 1,8,16,24,32,40,48. Game24: γ56@64, 16k, steps 1,8,16,32,48; 6 epochs.
- Critic quality vs step (pretrained 2D, Pearson / pick@V): Game24 k=1/8/16/32/48: 0.44/0.60/0.76/0.94/0.98. Countdown (16-step) k=1/4/8/14: 0.14/0.33/0.91/0.98. Near chance at k=1, informative from ~1/4, oracle in last third.  ← THE CURVE WE WANT TO SHIFT.

## Joint loop (Algorithm 1)
Per iteration: Actor: B=32 prompts, one k~U(W) each, rollout no_grad to s_{k-1}, grad call → s_k, step θ on (1/B)Σ logit-V. Buffer: 32 fresh prompts rolled to end, decoded, verified; push (s_{k_hi}, R) (k_hi = 40 of 64, 20 of 32) into replay (newest N_B=16,384). Critic: m_c=2 head-only AdamW steps on minibatch B'=64 from buffer; every n_q=10 iters compute AUC on newest N_q=256; if AUC<0.9, m'_c=8 extra head steps. Actor lr 1e-6 → 1e-7 cosine; critic lr 1e-4 constant. T=1500 (Countdown, Game24), 1000 (Sudoku, K&K). Train AND eval under deployment sampler (value fn is sampler-specific; training under cheap sampler costs -9 pts).
Without refit: critic exploited, true success 78%→0%. Fine-tuning backbone during RL: no benefit.
~4-6M denoiser calls/run; 45-90 min per 1000 iters on H200 (2-3 h on L40). 3 seeds.

## CFRL-GRR (baseline)
G=8 per prompt, 16 prompts/iter, γ4@16 rollouts, A = (r-mean)/std, J = (1/G)Σ A [Σ_k log π(a_k|s_k) + (1/L) log π(a_K|s_K)], exact Gaussian kernel log-density, 8 random transitions per update, lr 3e-6, 600 iters, no KL/clip. Score of transition = (η/(ψ²c_σ²)) Σ_i ε[i]ᵀ ∂d/∂θ (noise-weighted Jacobian → variance ∝ (K-1)|G|d).

## Tasks / verifiers
- Countdown (Stream of Search): 4-5 ints + target; response "12-5=7 , 4*7=28 , 28-3=25"; each input used ≤1; last = target. 500k SFT, 50k pool, 2k held-out.
- Sudoku: 81 cells, '*' blanks; verifier = exact valid grid. 400k SFT.
- K&K (Xie et al. 2024): 4-6 characters, unique solution. 300k SFT.
- Game24 (ToT): split by number set, 1,076 train sets → 13,150 orderings; 269 held-out → 1,601. Verifier = Countdown verifier with target 24.
- Eval: held-out split by index (even = val for ckpt selection, odd = test). Checkpoints every 250 iters.

## Results (test acc %, deployment sampler)
| | Countdown | Sudoku | K&K | Game24 |
|---|---|---|---|---|
| ELF-B + SFT | 42.8 | 64.0 | 68.2 | 44.0 |
| + GRR | 42.6 | 67.9 | 71.4 | 47.4 |
| + SVG-1D | 47.4 | 64.1 | 76.5 | 51.2 |
| + SVG-2D | 49.7 | 68.7 | 77.5 | 58.2 |
| GPT-2 SFT / +GRPO | 45.1/60.9 | 0.8/0.8 | 92.7/92.8 | 28.0/46.7 |
Game24 2D curve (3 seeds) @250..1500: 50.2/56.4/58.8/58.1/58.1/58.0. Countdown: 44.0/46.6/49.3/49.6/50.1/50.0.

## Diagnostics (Game24, base policy, k=24/64)
Gradient SNR at N=256: score-fn 0.06 (=1/√N floor), pathwise 2D 0.177. Half-split cosine N=256/1024/2048: score-fn ≈0; pathwise 0.79/0.94/0.97; GPT-2 token GRPO 0.09/0.35/0.34.
Per-position readings are NOT a step verifier (token AUC ~0.7 Game24, 0.2-0.5 Countdown).
Token commitment: 86% of Countdown positions decode to final token at the first call.

## Ablations (Game24, Δbest/Δfinal)
Ascend prob not logit −4.0/−6.1; min-pool −3.1/−2.5; train under cheap sampler −9.4/−9.4. Critic without x_{k-1} input: 47.0/12.9; backbone not fine-tuned: 51.7/13.4; random init: 37.9/4.1; head not refit: 49.3/0.5.

## Compute
~300 GPU-h total (A100 40/80, H200, L40).
