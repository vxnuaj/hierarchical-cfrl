# Results log

All accuracies: held-out test split, deployment sampler γ56@64, CFG 2, unless stated. Seeds noted.

## 2026-10-23 — Game24, first pass (SFT directly from ELF-B OWT checkpoint)

Setup: ELF-B init from `ELF-B-owt-torch`, 40 epochs on 17,961 Game24 orderings (1,090 train sets), batch 256, Muon, lr 1e-3, warmup 280, EMA 0.99. H-ELF: Γ_task (magnitude bins, n=16), cluster_scale 2.0, λ2 = 1.0, shared t.

| model | test acc (2,238) | train-prompt success (critic rollouts) |
|---|---|---|
| flat ELF-B | 11.57% | 88.4% |
| H-ELF (Γ_task) | 11.44% | (pending) |
| CFRL paper (init from Countdown model, ep 20/40) | 44.0% | n/a |

Reading: both models memorize the training orderings (train success 88%, loss ~0.01) and generalize poorly to new number sets. CFRL's 44% relies on initializing Game24 from a Countdown model trained on 500k examples for 60 epochs; we skipped that for speed. The two base models are matched (11.6 vs 11.4), so the Gate 1 critic-curve comparison on them is still a valid controlled test, but the faithful Gate 0 reproduction needs the Countdown-init recipe (queued: Countdown SFT flat+hier, then Game24 from it).

Also learned: the plain-model fine-tune at epoch 10 crashed in ELF's built-in generation eval (empty targets) → disabled; warmup must be scaled to run length (280 of 2,800 steps).

## 2026-10-23 — Gate 1 first look (Game24, OWT-init models): uninformative

Critic pretraining (2D, 6 epochs, 8,000 rollouts on train prompts, states at k ∈ {1,8,16,24,32,40,48}); held-out Pearson(value, reward) per step:

| k | 1 | 8 | 16 | 24 | 32 | 40 | 48 |
|---|---|---|---|---|---|---|---|
| flat | 0.36 | 0.42 | 0.40 | 0.43 | 0.40 | 0.40 | 0.42 |
| hier | 0.37 | 0.39 | 0.41 | 0.41 | 0.40 | 0.43 | 0.44 |

CFRL's curve rises 0.44 → 0.98 by k=48. Ours is flat because the SFT models memorize training puzzles: train-prompt success 88% (flat) / 83% (hier), nearly deterministic per prompt, so the outcome is a function of prompt identity rather than trajectory (only 21/192 held-out prompts had mixed outcomes). Critic train loss (0.8–1.3) is above the base-rate BCE, i.e. the critic is not fitting trajectory-dependent signal at all. Conclusion: no information about the hypothesis from this setup; the Countdown-init recipe (generalizing models, within-prompt outcome variance) is required. Checkpoint sweep (val subset = easiest 600 orderings): ep10 0.7%, ep20 17.0%, ep30 20.5%, ep40 21.2% → still improving, not over-trained. RL runs on these models continue only as a pipeline check.

Next: Countdown SFT flat+hier (500k ex, 60 epochs, batch 512) → Game24 from Countdown init (configs/*_b_game24_cd.yml) → redo gate0a/gate0b.

## 2026-10-23 — Pipeline-check RL on the OWT-init Game24 models: RL degrades the policy

CFRL-SVG (2D critic, paper hyper-parameters: B=32, lr 1e-6→1e-7 cosine, critic head refit 2 steps/iter at 1e-4, window k∈[8,40] of 64, deployment sampler γ56@64). Val/test accuracy per 250 iterations:

| iter | flat val | flat test | hier val | hier test |
|---|---|---|---|---|
| 0 | 10.8 | 11.6 | 11.6 | 11.7 |
| 250 | 11.8 | 12.2 | 11.7 | 10.5 |
| 500 | 10.0 | 10.4 | 11.8 | 11.2 |
| 750 | 8.2 | 8.6 | 9.5 | 9.8 |

Train-prompt success in the replay buffer falls monotonically (flat 0.87 → 0.52 over 900 iters; hier 0.81 → 0.53 over 700), critic AUC on the buffer stays 0.72–0.86, and the actor's objective (critic logit at the sampled step) *falls* with it (2.6 → 0.8) instead of rising. At iteration 0 the pre-trained critic's pooled logit is ≈50 (flat) / ≈29 (hier): the sum over ~60 generated positions saturates the sigmoid, as the paper's definition V = σ(Σ_i f(h_i)) implies; the head refit shrinks it within ~100 iterations.

Reading: in the memorization regime the critic's ranking is prompt identity, not trajectory quality (see the flat critic curves above), so the pathwise gradient carries no outcome information and the Adam-normalized steps are pure drift, which with no KL anchor degrades the policy. This is consistent with CFRL's own observation that drift is "controlled by small actor lr and critic refit", i.e. by a critic that is informative. The implementation was re-audited against the paper's equations (value, 2D loss, window, buffer, refit schedule): it matches. A frozen-critic probe (no refit, 150 iters) is running to confirm the actor can ascend the critic at all (expected: critic value rises, true success falls = exploitation). Verdict on Gate 0 waits for the Countdown-init models.

Update (iter 1000, same runs): flat val 6.2 / test 5.3; hier val 9.3 / test 9.2. Degradation continues; the hierarchical model degrades more slowly (not a claim, n=1).

**Frozen-critic probe (Game24 flat, no head refit; 120 iters before the 1h debug slot ran out on a slow node, ~25 s/iter):** the actor's objective rises from ≈50 to 760 → 890 (30-iter means) with grad norm 1,300–3,200 before clipping, while the buffer success rate drifts only 0.88 → 0.85, about the same drift as with refit. So the pathwise gradient path through the sampler is live and the actor exploits a static critic exactly as CFRL reports ("without refit: critic exploited"). The actor machinery is not the reason RL fails on these models.

## 2026-10-23 — Knights-and-Knaves SFT: task as formulated is learned to 99%

H-ELF (Γ_kmeans, n=16), 8 epochs on 300k puzzles; val accuracy (600 held-out puzzles, γ56@64) per epoch:

| epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| hier val acc | 6.2 | 29.3 | 60.3 | 78.8 | 89.0 | 96.2 | 98.0 | 99.0 |

Test accuracy at epoch 8: 99.3% (1,000 puzzles), and 98.9% on train prompts. Our symbolic K&K format (`A: C knave ; B: A same C`) is much easier than the natural-language K&K of CFRL (68% at epoch 4 of 8), so the final checkpoint leaves no headroom for RL and no failures for the critic to learn from. Following CFRL's own practice of taking an early checkpoint on K&K, the RL base is the checkpoint with val accuracy closest to 70%: epoch 4 (78.8%) for H-ELF; the flat model's sweep runs with the same rule (`select_ckpt --target 0.7`). The within-prompt outcome variance at 79% is what the critic-curve test needs. Flat model sweep: ep3 56.3, ep4 79.5, ep5 90.3, ep8 99.2 (val); test accuracy of the epoch-4 bases: flat 78.9%, hier 79.1% (1,000 puzzles), train-prompt success 78–79% for both → matched bases with within-prompt variance. K&K coarse stream check: T5 tokenizes "knight" as one piece (cluster 3) and "knave" as four (clusters 10,2,2,2), so Γ_kmeans does carry the answer skeleton here, unlike on arithmetic.

Final (iter 1500): flat val 5.9 / test 5.9; hier val 8.9 / test 8.4 (from 11.6 / 11.7). Runs closed.

## 2026-10-23 — Gate 1 on Knights-and-Knaves (epoch-4 bases, 79%): critic is near-oracle from k=8 for BOTH models

2D critic pre-trained 6 epochs on 8,000 rollouts (2,000 train prompts × 4) with states at k ∈ {1,8,16,24,32,40,48}; held-out shards (≈800 states per step); best epoch by field score (flat 4, hier 2).

| k | flat Pearson | flat AUC | flat pick@V (31 mixed groups) | hier Pearson | hier AUC | hier pick@V (45 mixed groups) |
|---|---|---|---|---|---|---|
| 1 | 0.67 | 0.92 | 0.87 | 0.59 | 0.87 | 0.67 |
| 8 | 0.95 | 0.99 | 1.00 | 0.91 | 0.99 | 0.91 |
| 16 | 0.98 | 1.00 | 1.00 | 0.96 | 1.00 | 1.00 |
| 24–48 | ≥0.99 | 1.00 | 1.00 | 0.97–0.99 | 1.00 | 0.96–0.98 |

Reading: on our symbolic K&K the policy commits to its answer almost immediately (the answer is a function of the prompt, no search), so the value at k=8 of 64 already predicts the outcome; CFRL's natural-language K&K critic rises more slowly (pick@V 0.58 → 0.74 → 0.84 → 0.95 → 0.99). There is no room for a coarse stream to make the critic informative *earlier* here, and the hierarchical critic is, if anything, slightly behind at k=1–8 (within the noise of 31–45 mixed groups). Gate 1 on K&K: no shift. This makes K&K a control case ("early-commitment task: no hierarchy effect expected") rather than a test of the hypothesis; Game24-from-Countdown, where CFRL's curve rises slowly (0.44 → 0.98), remains the test. Note also that Pearson/AUC are dominated by between-prompt difficulty (AUC 0.87–0.92 at k=1 from the prompt alone); pick@V is the within-prompt metric that matters. A direct token-commitment analysis (hcfrl/analysis/commitment.py: fraction of positions already at their final token/cluster per step, token stream vs coarse stream) is queued for all four collected rollout sets.

RL on these bases (SVG-2D 1000 iters, GRR 600 iters) running: outputs/rl/kk_{flat,hier}_{svg2d,grr}_s0.

## 2026-10-23 — Token-commitment profiles: the detail stream is ~85% committed at the FIRST denoiser call; the coarse stream does not lead

`hcfrl/analysis/commitment.py` on the stored critic rollouts (deployment sampler γ56@64). Fraction of generated positions whose decode of the clean estimate x_{k-1} already equals the decode at k=48 (`tok`); the same at cluster granularity Γ(token) (`tok_cl`); and for H-ELF the argmax of the dedicated cluster channels (`ch_cl`).

| task / model | split | k=1 tok | k=1 tok_cl | k=1 ch_cl | k=8 tok | k=8 ch_cl |
|---|---|---|---|---|---|---|
| KK flat | all / fail | 0.84 / 0.65 | – | – | 0.96 / 0.89 | – |
| KK hier | all / fail | 0.82 / 0.62 | 0.85 / 0.68 | 0.84 / 0.67 | 0.95 / 0.85 | 0.95 / 0.87 |
| Game24 flat (OWT-init) | all / fail | 0.87 / 0.72 | – | – | 0.99 / 0.96 | – |
| Game24 hier (OWT-init) | all / fail | 0.84 / 0.71 | 0.89 / 0.79 | 0.87 / 0.77 | 0.99 / 0.96 | 0.99 / 0.98 |

Three facts. (1) With x-prediction + self-conditioning, the first call from pure noise already produces the model's answer for 84–87% of positions (CFRL reports 86% on Countdown); the trajectory is "guess, then refine the uncertain positions", not progressive sharpening. Failing trajectories are the less-committed ones (62–72% at k=1), which is what the critic reads. (2) The dedicated coarse channels are **not** committed earlier than the coarse class implied by the token stream itself (`ch_cl` ≤ `tok_cl` at every k, both models, both tasks). Jointly denoising a cluster stream at the same noise level gives no temporal hierarchy: the coarse channels are predicted with the same confidence profile as the tokens. (3) Since the state information under heavy churn (α = 0.125) is carried by x_{k-1} rather than z, a schedule shift on the cluster channels of z would change little.

Implication for the hypothesis: the mechanism "coarse stream resolves earlier → critic informative earlier" is not present in the H-CDLM-style implementation. Gate 1 on Game24-from-Countdown (tomorrow) remains the direct test, but the prior is now negative. Candidate pivots recorded in RESEARCH_PLAN.md.

## 2026-10-24/25 — Game24-from-Countdown: strong SFT base (64%), and the first RL split between the models

Countdown SFT (60 epochs, 58,560 steps) then Game24 fine-tune from it; selected checkpoints by val acc (flat step 2450, hier 2800). Test accuracy (2,238 held-out puzzles, γ56@64): **flat 64.1%, hier 64.9%** (CFRL's SFT: 44.0%; our base is stronger, so less headroom).

Gate 1 (2D critic, held-out shards, Game24-cd): no clean shift. k=1: flat AUC 0.73 / pick@V 0.75 (16 mixed groups), hier AUC 0.87 / pick@V 0.52 (23 groups); from k=16 both AUC ≥0.97. Same picture as K&K: the critic is good early for both, and pick@V differences at k=1 are within the noise of 16–23 groups.

SVG-2D RL, paper hyper-parameters, seed 0, val / test per 250 iters:

| iter | flat val | flat test | hier val | hier test |
|---|---|---|---|---|
| 0 | 64.3 | 64.2 | 63.9 | 64.7 |
| 250 | 53.6 | 53.3 | 63.7 | 66.1 |
| 500 | 44.9 | 45.9 | 67.7 | 70.8 |
| 750 | 58.0 | 59.7 | 68.7 | 70.3 |
| 1000 | 56.7 | 58.6 | 68.7 | 70.8 |
| 1250 | 56.7 | 57.9 | 68.7 | 68.9 |
| 1500 | 53.6 | 55.0 | 68.1 | 69.9 |

Hier: **+5.2 test (64.7 → 69.9; peak 70.8)**, buffer success stays 0.75–0.94. Flat: **−9.2 test (64.2 → 55.0)**, buffer success collapses to 0.53 by iter 400 and only partly recovers; the actor objective falls from ≈800 to ≈1.5 by iter 500 (critic refit keeps up with the drift and ends up scoring everything low). n = 1 seed each; seeds 1 and 2 and an actor-lr 3e-7 check submitted (see below).

K&K (epoch-4 bases, 79%), SVG-2D 1000 iters: flat 79.2 → 71.9 test, hier 79.7 → 74.5. Both degrade.

**Diagnostic (K&K hier, 500 iters): `--actor_mode random` (same step size, random direction) stays at 78.7 → 78.8 test; `--actor_mode none` (no actor step) 79.7 → 78.5.** So the degradation is caused by the *direction* of the critic gradient, not by the step size or by the Muon-trained weights; the Muon/AdamW mismatch hypothesis is dropped. The critic's gradient on these models points towards higher critic logit but lower true success at the paper's lr, except on Game24-cd hier.

GRR baseline: all four runs died with CUDA OOM (128 rollouts × 8 transition graphs in one backward). Fixed by chunking the backward (`--chunk 16`, grads accumulate); resubmitted for Game24-cd and K&K, flat and hier.

Submitted 2026-10-25: gate0c (GRR) ×4; gate0b Game24-cd seeds 1,2 × {flat,hier}; gate0b Game24-cd lr 3e-7→3e-8 × {flat,hier} (out tags `game24cd_*_svg2d_lr3e-7_s0`).

## 2026-10-26 — Three seeds on Game24-from-Countdown: the split is systematic

SVG-2D, paper recipe, test accuracy at iter 1500 (2,238 puzzles; bases 64.1 flat / 64.9 hier):

| seed | flat | hier |
|---|---|---|
| 0 | 55.0 | 69.9 |
| 1 | 50.2 | 68.8 |
| 2 | 52.1 | 70.3 |
| **mean** | **52.4 (−11.7)** | **69.7 (+4.8)** |

Learning-rate check (seed 0): flat at 3e-7→3e-8 still degrades (49.5); hier at 3e-7 gains less (67.4), at 3e-6 collapses (55.8). The flat failure is not a step-size problem; the recipe has a narrow lr window for hier too.

**Countdown (third task, same recipe):** SFT bases are weak (flat 17.2 / hier 19.1 test; CFRL reports 42.8 — our self-generated Countdown distribution is evidently harder or our SFT weaker). SVG-2D first improves BOTH models (flat 17.6→26.1, hier 18.7→24.6 at iter 250) and then both collapse (flat 10.5, hier 14.3 at 1500). So the "early gain then collapse" pattern is the general behaviour of this recipe on our models; hier is protective on Game24 (3 seeds) but not on Countdown or K&K.

**GRR (score-function baseline) at the paper's lr 3e-6 collapses everywhere** (Game24-cd 64→11.6 / 6.0; K&K 79→65.5 / 46.0; Countdown 17.6→26 at iter 400 then 17.1 / 2.7). Pre-clip grad norms reach 1e5. Rerunning at lr 3e-7.

**Failure modes (Game24-cd, 500 val prompts, seed 0):** hier reduces arithmetic-error outputs (19.6% → 13.4%) while keeping step structure (n_steps/operand errors ≤2.6%); flat keeps its arithmetic errors (~21%) and drifts into structural errors (wrong step count 0→5%, unavailable operand 2→6%), regressing 18% of previously-solved prompts vs 3.6% newly solved. Output diversity and token repetition are unchanged in both: no mode collapse, a drift of the chain structure.

**Gradient quality at the SFT policy (256 prompts, per-prompt gradients):**

| task / model | k | SNR | cos(half,half) N=256 | mean per-prompt ‖g‖ |
|---|---|---|---|---|
| Game24-cd flat | 8 / 24 / 40 | 0.10 / 0.15 / 0.12 | 0.49 / 0.66 / 0.60 | 3783 / 2868 / 3247 |
| Game24-cd hier | 8 / 24 / 40 | 0.15 / 0.13 / 0.18 | 0.72 / 0.68 / 0.75 | 1990 / 2638 / 2541 |
| K&K flat | 8 / 24 / 40 | 0.15 / 0.16 / 0.21 | 0.67 / 0.76 / 0.87 | 3699 / 3079 / 2525 |
| K&K hier | 8 / 24 / 40 | 0.08 / 0.08 / 0.13 | 0.23 / 0.34 / 0.68 | 1840 / 1966 / 1285 |

On Game24 the hier gradient is more consistent across prompts; on K&K it is less consistent, yet K&K flat (the most consistent gradient of all) degrades the most. Gradient agreement across prompts therefore does not predict whether RL helps: a consistently-pointed gradient can consistently point at critic exploitation. The critic's input-gradient puts only 2–3% of its norm on the 16 cluster channels, so the critic barely reads the coarse stream directly; whatever hier does on Game24 is through the policy's parameterization, not through the critic reading clusters.

**Channel ablation runs were void:** grad_norm = 0 throughout both (`--actor_channels tokens` and `clusters`), i.e. `split_channels` cut the whole gradient. Debug job queued; rerun after the fix.

Queued 2026-10-26: GRR at lr 3e-7 (Game24-cd, Countdown; flat/hier); K&K failure modes (classifier fixed); H-ELF-R control (balanced random cluster map over the task vocabulary, `data/clusters_random_n16.pt`): Countdown SFT → Game24 SFT → critic → SVG ×3 seeds (~2 days).

**K&K failure modes (500 val prompts, seed 0, iters 0→1000):** no format errors at all in either model (every output parses); the only failure class is a wrong assignment. Flat 79.4→72.0: regressed 12.6% / newly solved 5.2%; hier 77.6→72.0: regressed 11.2% / newly solved 5.6%. Output diversity is ~0.21 for both (few distinct answer strings, as expected for 4–6 knight/knave labels). K&K degradation is a plain re-labelling drift, not a structural one; nothing for a coarse stream to protect.

Bug found 2026-10-26: `split_channels` had been inserted under `evaluate`'s `@torch.no_grad()` decorator, so it ran in no-grad mode (both ablation runs had grad_norm 0 and never moved). Fixed, ablations resubmitted; seeds 1–2 queued for K&K and Countdown SVG (flat/hier) to test the ordering claim.

## 2026-10-26 — Re-read of CFRL (full TeX) against our failure: two recipe differences, one diagnostic to copy

1. **Countdown critic sampler.** CFRL pre-trains the Countdown critic on states of the *cheap* γ4@16 sampler (steps 1,2,4,…,14 of 16; 64k rollouts) because the deployment sampler's early/middle states carry almost no outcome information there (held-out AUC ≈0.6 for k≤24 of 64); actor and refit still run on γ56@64. We pre-trained ours on γ56@64 states. That is a plausible reason both of our Countdown runs collapse: the actor ascends an uninformative critic in the window. Re-done: `gate0a` now takes critic-sampler args; tags `countdown_{flat,hier}_c16` → SVG seed 0 (jobs 13446655–8). Queued Countdown seeds 1–2 on the old critic cancelled.
2. **SFT optimizer.** CFRL's task SFT is the ELF release recipe: AdamW lr 2e-3 with warm-up, batch 256–512, EMA 0.999 (0.99 on Game24); Countdown epoch 60 of 100, Game24 epoch 20 of 40 from the Countdown model. Ours used Muon (blr 1e-3; the public ELF PyTorch default) and EMA 0.9999 on Countdown. The random-direction diagnostic ruled out *step size* as the cause of flat's degradation, not the weight geometry the optimizer leaves behind. A paper-recipe pair is now training: `configs/{elfA,helfA}_b_{countdown,game24_cd}.yml` (AdamW 2e-3, EMA 0.999/0.99), full chains to Game24 SVG/GRR and Countdown SVG with the cheap-sampler critic (jobs 13446661–13446674, ~2 days).
3. **One-step test** (CFRL App. "One step along each gradient"): at the SFT base, one AdamW-normalised step of length η along the mean pathwise gradient of 1,024 rollouts raises their Game24 accuracy by +4.2 at η=Λ/4 (Λ = parameter distance of a full RL run); the raw step destroys the policy; a random step does nothing. Implemented as `hcfrl/analysis/one_step.py` (Λ from our own runs' policy_it0 → policy_it750); running for Game24-cd and K&K, flat vs hier (jobs 13446659/60). This isolates the *direction* of the critic gradient at iteration 0 from the joint-training dynamics: if flat's step is ≤0 and hier's is positive, the difference is in the critic's slope on each policy's states.

Also from the re-read, matching our Game24 hier run: CFRL's own failure-mode table (cut appendix) says no sample ever misses the target, RL lowers the rate of arithmetically false equations (47.8 → 31.5% on Game24) and slightly raises number-use violations (7.2 → 10.4%). Our hier run does exactly this (arith 19.6 → 13.4%, operand 2.0 → 2.6%); our flat run does the opposite (structural errors rise). Their critic-pretraining budgets are larger than ours (Game24 16k rollouts vs our 8k; K&K 32k vs 8k), and their frozen-critic probe matches ours (value climbs to >80 logits while true success → 0).

## 2026-10-26 — GRR at lr 3e-7: the score-function baseline works, and the flat model is RL-trainable

CFRL-GRR (critic-free, group-relative REINFORCE), 10× below the paper's lr, test accuracy by iteration (seed 0):

| task | model | 0 | 100 | 200 | 300 | 400 | 500 | 600 |
|---|---|---|---|---|---|---|---|---|
| Game24-cd | flat | 63.9 | 64.9 | 65.1 | 66.1 | 66.7 | 68.2 | **68.5** |
| Game24-cd | hier | 64.9 | 65.7 | 65.8 | 65.6 | 67.6 | 68.9 | **69.0** |
| Countdown | flat | 17.6 | 17.5 | 16.4 | 17.0 | 17.6 | 18.2 | 17.3 |
| Countdown | hier | 18.7 | 19.2 | 18.2 | 19.2 | 18.8 | 17.3 | 18.6 |

Game24 final (iter 600): **+4.6 (flat) / +4.1 (hier)**, still rising at the end; Countdown neutral (CFRL: −0.2). This reproduces CFRL's GRR behaviour (Game24 +3.4, Countdown −0.2) once the lr is lowered, and it settles a question: the flat model *is* improvable by RL on Game24. What collapses it is specifically the pathwise (SVG-2D) update at the paper's recipe, and the hierarchical model is what makes that update usable (SVG-2D hier +4.8 ≈ GRR gain so far, flat −11.7). Baseline table for the paper: SFT 64 → GRR ~68 (both) → SVG-2D flat 52 / hier 70.

**Audit of the GRR and sampler code against CFRL App. B (2026-10-26):** the transition kernel (λ, η, ψ with ψ=1−α, variance ψ²c_σ², denoiser queried at the lowered time t̃), the exact-Gaussian log-density of 8 random stochastic transitions out of K_roll−1 scaled by (K_roll−1)/8, the group advantage (r−mean)/(std+ε), the (1/L)-scaled decoder term and the γ4@16 rollouts all match. The GRR blow-up at lr 3e-6 is therefore not an estimator error; at 3e-7 it reproduces the paper's gains. The pathwise section says the decoder is frozen during SVG: in our implementation the actor objective reaches no decoder parameter (`proj_kernel/bias`, `unembed_kernel/bias`, 16.9M of 104.6M, have no gradient), so with weight decay 0 they never move — equivalent. GRR at the paper's lr 3e-6 is additionally queued on the AdamW-recipe models (game24cdA_*_grr_s0) to test whether the blow-up is a property of the Muon-trained weights.

Final GRR lr 3e-7 (2026-10-26): Game24-cd flat 63.9→68.5, hier 64.9→69.0; Countdown flat 17.6→17.3, hier 18.7→18.6. Score-function RL is model-agnostic here (same gain on both), which is exactly what makes the pathwise split (flat −11.7 / hier +4.8) the finding: the coarse stream changes what the *critic gradient* does to the policy, not what RL in general can do to it.

Channel ablations (fixed `split_channels`, hier Game24-cd, seed 0) at iter 250: tokens-only gradient 67.8 test, clusters-only gradient 68.9 (full hier at 250: 66.1; final 69.9). Grad norms now non-zero (60–330 pre-clip). Too early to read; both still climbing.

Queued 2026-10-26: paper-budget critic for Game24-cd (16k rollouts = CRIT_N 4000×4, vs our 8k), flat and hier, then SVG seed 0 (tags `game24cd_{flat,hier}_bigcrit`; jobs 13448949→50, 13448951→52). Closes the last recipe gap on Game24 that could explain the flat collapse (sampler and optimizer gaps already covered by the c16 and A chains).

## 2026-10-26 — One-step test: the critic's direction at the SFT base is good for BOTH models

CFRL's diagnostic (App. "one step along each gradient"), reproduced with `hcfrl/analysis/one_step.py`: mean pathwise gradient over 128 prompts × 8 rollouts at k∈{8,16,24,32}, one step of length η along three directions, paired eval (fixed noise, seed 777) on the first 1,000 val prompts (the val file is grouped by puzzle, so these are the easiest puzzles: base 77.2% for both models, same count — a coincidence worth re-checking on a random subset). Λ = ‖θ_it750 − θ_it0‖ of our own SVG run. Accuracy change in points:

| task / model | Λ | η=Λ/16 adamw / raw / rand | Λ/4 | Λ/2 | Λ |
|---|---|---|---|---|---|
| Game24-cd flat | 1.17 | +1.2 / −2.9 / −0.5 | **+5.3** / −77 / +0.2 | **+8.0** / −77 / −1.6 | −3.5 / −77 / −1.7 |
| Game24-cd hier | 1.54 | +1.6 / −0.5 / 0.0 | **+5.3** / −77 / 0.0 | **+6.6** / −77 / −0.1 | −0.1 / −77 / +0.1 |
| K&K flat | 0.87 | +0.1 / −1.2 / −1.4 | −0.5 / −3.7 / −1.6 | −1.2 / −21 / −1.9 | −18 / −74 / −0.3 |
| K&K hier | 0.98 | +0.4 / +0.1 / +1.1 | +1.9 / −11 / −1.1 | −3.6 / −58 / −0.8 | −31 / −79 / +1.2 |

CFRL reports +4.2 at Λ/4 on Game24 (raw step destroys the policy, random does nothing); we get +5.3 on both models with the same qualitative pattern. **So the flat model's collapse under SVG-2D is not a bad initial gradient direction**: one normalised step along the critic's gradient from the flat SFT base is as good as from the hier base. The split arises in the *iterated* dynamics (actor drift + critic refit over 1,500 iterations), which is where the coarse stream makes the difference. On K&K the one-step gain is ≈0 for flat and +1.9 at Λ/4 for hier (both within noise of a 1,000-prompt eval), so neither K&K critic offers much of a slope at the base; the gains seeds 1–2 show for hier there come from the iterated dynamics as well.

**K&K seeds 1–2 (test acc by iter, running):** flat s1 78.6→72.2→76.2→65.3 (750); s2 78.0→72.9→70.1 (500). Hier s1 78.0→80.1→79.8 (500); s2 78.1→80.4→79.3 (500). Seed 0 was flat 79.2→75.2 (500)→71.9 (1000), hier 79.7→74.8 (500)→74.5 (1000). At iter 500 hier ≥ flat in all three seeds (74.8/79.8/79.3 vs 75.2/76.2/70.1); seeds 1–2 hier are at or above base. The earlier "K&K: both degrade" read was seed 0 only; the three-seed picture looks like Game24's ordering, smaller margin.

Tooling 2026-10-26: `python scripts/status.py [--watch 600] [--filter kk]` renders a terminal dashboard of the cluster state (queue, every RL run with base/last/Δ/sparkline/live state, SFT checkpoints, critics, analysis files, slurm logs with errors). The same file runs on the login node with `--remote` and returns JSON.

## 2026-10-26 — K&K three seeds: hier is protective there too

SVG-2D on K&K (1000 iters, test acc, bases ~78–80):

| seed | flat | hier |
|---|---|---|
| 0 | 79.2 → 71.9 (−7.3) | 79.7 → 74.5 (−5.2) |
| 1 | 78.6 → 71.0 (−7.6) | 78.0 → 78.0 (0.0) |
| 2 | 78.0 → 66.9 (−11.1) | 78.1 → 78.9 (+0.8) |
| **mean** | **−8.7** | **−1.5** |

Hier ≥ flat in 3/3 seeds; seeds 1–2 hier peak at +2 (iter 250) and hold; flat drifts down in all three. Same ordering as Game24-cd with a smaller margin and no net gain. The earlier "K&K: both degrade equally" was a seed-0 artefact.

**Channel ablations (hier Game24-cd, seed 0, test acc by iter):** tokens-only gradient 64.9 → 67.8 / 69.1 / 70.3 / 70.4 / 69.8 (iter 1250, running) — tracks the full hier run. Clusters-only gradient 64.9 → 68.9 / **71.3** / 69.7 / 66.1 / 65.1 — fastest early gain, then decays back to base. Read: the cluster channels alone carry a usable, even stronger, early signal but cannot sustain it; the token channels alone give the stable gain. Final numbers at 1500 pending.
