# Research Plan: Hierarchical Continuous Flow LMs for Pathwise RL

Status: v1, 2026-10-23. Owner of execution: Claude. Target: ICML 2027 (est. abstract Jan 16, paper Jan 22, 2027; unconfirmed). 13 weeks.

## 0. One-paragraph summary

We take the pathwise RL recipe of CFRL (arXiv 2610.19080: a per-position critic on intermediate denoising states of a 105M continuous flow LM, ELF, with reparameterized gradients through one denoiser call) and ask whether adding a coarse, jointly-denoised abstraction stream (the construction of H-CDLM, arXiv 2610.08738: a fixed many-to-one map from tokens to a small set of classes, encoded as extra channels and denoised with the tokens) makes that RL more effective. The proposed mechanism is specific and measurable: **coarse-stream information survives more noise, so a critic reading the hierarchical state becomes predictive of the verifier reward earlier in the trajectory, which widens the window of denoising steps that carry usable pathwise gradient.** Everything in the plan is built to either confirm that chain or kill it cheaply.

## 1. What the two anchor papers actually established (facts we build on)

**CFRL (pathwise RL).** Base model ELF-B (105M, frozen T5-small embeddings, d=512, x-prediction, self-conditioning, CFG). Tasks: Countdown, Sudoku, Knights-and-Knaves, Game24 with programmatic verifiers. State at step k is (re-noised embeddings z_k, previous clean estimate x_{k-1}). Critic = frozen copy of the task-SFT denoiser + MLP head per position, trained with BCE against the terminal 0/1 reward; actor maximizes the critic's pre-sigmoid logit through ONE differentiable denoiser call at a step k drawn uniformly from [K/8, 5K/8]. Critic head refit after every actor step from a replay buffer. Results: Game24 44.0 to 58.2, Countdown 42.8 to 49.7, K&K 68.2 to 77.5, Sudoku 64.0 to 68.7. REINFORCE-style baseline on the same model gains ~0 to 4 points. Key diagnostic: the critic is near chance at k=1, informative from about 1/4 of the trajectory, oracle in the last third. Code not released; the paper is detailed enough to reimplement (notes in `notes/cfrl.md`). About 300 GPU-hours for their whole paper.

**H-CDLM (hierarchy).** Fixed map Γ: vocab to n=16 clusters via k-means over pretrained input embeddings. Cluster id encoded continuously (one-hot or bits) and concatenated along channels at each position; one backbone denoises both; per-modality noise schedules; a second output head for the cluster stream with loss weight λ2. Measured: cluster information resolves at higher noise than token information (median noise 1.13 vs 0.89). Random clusters give zero gain, so semantics matter. Gains: 25 to 33 percent GenPPL on LM1B/OWT, +2 to +4 points on GSM8K at fewer steps. Code not released. Their appendix sketches exactly our needed variant: ELF with extra channels and a second unembedding head.

**Why ELF, not Lafa.** Lafa (large-flow-lms, already cloned) is block-size-4 and Qwen-based; a coarse stream over 4 tokens is meaningless and CFRL's recipe is full-sequence. ELF has an official PyTorch port with released checkpoints and is what CFRL used, so reproducing CFRL on it is the cleanest possible baseline.

## 2. Hypothesis, stated so it can fail

H1 (mechanism): For a hierarchical flow LM, the critic's rank correlation with the terminal reward, plotted against denoising step k, is shifted earlier than for a flat model of matched base accuracy.

H2 (consequence): Pathwise RL on the hierarchical model reaches higher held-out accuracy at equal rollouts and equal wall-clock, with lower variance across seeds, than on the flat model.

H3 (attribution): The gain disappears when (a) the critic is blinded to the coarse channels, or (b) the coarse map is random, while the hierarchical architecture is otherwise unchanged.

If H1 fails on a task, H2 is not expected to hold on that task, and we say so.

## 3. Method: H-ELF and hierarchical pathwise RL

**Coarse map Γ.** Two instantiations, both deterministic functions of the token so the joint sampler remains an exact token sampler (H-CDLM Prop. 1):
- Γ_kmeans: k-means (n=16, k-means++, 10 restarts) over T5-small input embeddings. Generic; identical to H-CDLM.
- Γ_task: a task-informed abstraction for arithmetic tasks: number tokens mapped to 8 magnitude bins, operators to 4 classes, plus "=", separators, other, special (n=16; `hcfrl/hier/clusters.py`). This exists because on Countdown/Game24 the cluster sequence under Γ_kmeans is a fixed template that any SFT model already knows. **Confirmed 2026-10-23:** k-means over T5-small embeddings puts all 529 number tokens in one cluster and all operators in another, so Γ_kmeans carries zero reward-relevant information on these tasks. Γ_task is therefore the primary design for arithmetic tasks; Γ_kmeans is the design for natural-language tasks (K&K) and is kept as a control on arithmetic. Reported openly in the paper.

**Architecture.** Per position, state = [z1 ∈ R^512 (token embedding stream), z2 ∈ R^16 (cluster one-hot stream)], both noised; coupled per-modality schedules; one backbone; output heads: x1-prediction (existing) and a 16-way cluster softmax (new, loss weight λ2). Parameter overhead about 1 percent. Decoding uses tokens only.

**RL.** CFRL-SVG-2D unchanged except the critic's per-position feature is the backbone feature of the hierarchical state. Three gradient channels exist (re-noised z1, self-conditioning x1, and the coarse stream); we log the gradient-norm share of each.

**Controls.**
- Flat: ELF-B + CFRL-SVG-2D (reproduction).
- Hier: H-ELF + CFRL-SVG-2D.
- Hier, blinded critic: H-ELF policy, critic reads only the token-stream features.
- Hier, random Γ.
- Flat, parameter-matched (add the 1 percent as width in the head MLP).
- Both models without RL (base accuracy must match within seed noise; if hierarchy improves base accuracy, we re-SFT the flat model longer to match, and report RL gain as a delta over base).
- Score-function baseline (CFRL-GRR) on both, to show the effect is specific to pathwise RL.

## 4. Experiments, in order, with go/no-go gates

**Phase 0, weeks 1-2 (to Nov 6): reproduce.**
1. ELF-B PyTorch running on CARC; task SFT on Game24 (small, fast) and Countdown.
2. Implement CFRL-SVG-2D: sampler with exact churn coefficients, critic pretraining, joint loop.
3. Gate 0: Game24 RL gain within about 5 points of CFRL's +14. If we cannot reproduce, the project pivots to a different RL method before building hierarchy.

**Phase 1, weeks 3-4 (to Nov 20): hierarchy and the mechanism test.**
1. Build H-ELF with Γ_kmeans and Γ_task; SFT on both tasks; match base accuracy to flat.
2. Pretrain critics on flat and hierarchical states from the same rollouts.
3. Gate 1 (H1): critic rank-correlation vs k curves. This costs a few GPU-hours and decides whether RL is worth running. If neither Γ shifts the curve on Game24/Countdown, move to K&K (natural-language-like tokens) and GSM8K-style data, where k-means clusters are meaningful.

**Phase 2, weeks 5-8 (to Dec 18): the main result.**
1. 2x2 (flat/hier x pathwise/score-function), 3 seeds, Game24 and Countdown, 1500 iterations each, deployment sampler, learning curves logged every 250 iterations.
2. Ablations: blinded critic, random Γ, parameter-matched flat. 3 seeds for the primary ablation, 2 for the rest.
3. Diagnostics: gradient SNR and half-split cosine at several k; per-channel gradient norm share; critic AUC over training.

**Phase 3, weeks 9-11 (to Jan 8): generalization.**
1. Third task (K&K, which CFRL also used; Sudoku if time).
2. One larger model if compute allows (ELF-M, 342M), one task.

**Phase 4, weeks 12-13: write and submit.** Paper drafting starts in week 8 in parallel; figures are generated by scripts from logged runs so late results drop in.

## 5. Compute estimate

CFRL reports 45-90 minutes per 1000 RL iterations on one H200, so about 2-4 hours per 1500-iteration run on an A100. Main 2x2 with 3 seeds on 2 tasks is 24 runs, about 100 GPU-hours. Ablations and diagnostics about 80. SFT for flat and hierarchical on two tasks about 60. Critic pretraining about 20. Phase 3 about 100. Total around 350-400 A100-hours, so the project needs sustained access to roughly 4 GPUs for 6 weeks. Local Mac has no CUDA; all GPU work goes to USC CARC.

## 6. Risks and what we do about them

- CFRL code is unreleased; reimplementation may miss a detail. Mitigation: their appendix gives sampler coefficients and every hyperparameter; Gate 0 exists for this.
- The coarse stream may carry no reward signal on arithmetic tasks. Mitigation: Γ_task and Gate 1; fallback tasks with natural-language tokens.
- Hierarchy improves base accuracy and confounds the RL comparison. Mitigation: match base accuracy before RL, report deltas, and run the blinded-critic ablation which holds the policy's architecture fixed.
- λ2 is dataset-dependent (H-CDLM). Mitigation: small sweep at SFT time on validation loss only, chosen before any RL.
- Both anchor papers are weeks old; their authors may be doing this. Mitigation: narrow scope, start RL runs by week 5, and preprint on arXiv as soon as the main result is in.

## 7. Novelty (from the survey in `notes/survey.md`)

No published work combines a hierarchical representation with pathwise RL on a language flow or diffusion model. CFRL is the only reward-backprop RL for any language diffusion model and has no follow-ups yet. All other diffusion-LM RL is score-function. Hierarchical continuous text generators (H-CDLM, HC-DLM, HMDLM, HDLM, NCM, Cola DLM) have no RL. Latent-reasoning RL (LaDi-RL, LTC, Latent-GRPO) is GRPO on single-scale latents.

Closest threats: LaDi-RL (score-function RL on a latent diffusion policy, strong 8B numbers, two-level rollout scheme), the HC-DLM and H-CDLM groups adding RL, and the image-side trio VAR RL Done Right, Stepwise-Flow-GRPO, and VARD (critic on noisy latents plus pathwise update). ELF-REG (a jointly-denoised global vector on top of ELF, 56 percent GSM8K at 652M, code released) is both a competing notion of hierarchy and a candidate stronger base.

Positioning: frame the contribution as an estimator result. MAGE (2004.14309) bounds pathwise policy-gradient bias by the critic's action-gradient error, not its value error, and the abstraction theory of Rosas (2606.06123) and Abel et al. splits value error into a learning term and an abstraction term. The coarse level trades a small abstraction term for a much smaller learning term, so the critic's gradient is more accurate earlier in the trajectory. VinePPO, EVPO, and the value-flattening paper give empirical evidence that token-level critics are poor. That makes the quantity to measure the critic's gradient error and the resulting gradient SNR per step, which is what Gate 1 and the diagnostics are built around.

## 8. Repository layout (in progress)

- `hcfrl/tasks/`: Countdown and Game24 generators and verifiers (done, tested).
- `hcfrl/elf/`: vendored ELF PyTorch port plus H-ELF changes.
- `hcfrl/rl/`: sampler with gradient hooks, critic, CFRL-SVG loop, GRR baseline.
- `notes/`: paper digests used for implementation.
- `scripts/`: Slurm job files for CARC.
