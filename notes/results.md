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

**Channel ablations, final (hier Game24-cd, seed 0, iter 1500):** tokens-only gradient **68.3 (+3.4)**; clusters-only gradient **63.2 (−1.7)** after peaking at 71.3 (iter 500); full hier 69.9 (+5.2); flat 55.0 (−9.2). So (i) restricting the pathwise update to the token channels of the hier model already avoids the flat collapse — the protection is in the hier *parameterization/denoiser*, not in which channels the gradient enters through; (ii) the cluster channels alone give the fastest early gain but the run then decays toward base, i.e. the coarse stream is a strong but exploitable direction unless the token stream moves with it. n = 1 seed.

**One-step test on a random val subset (seed 777, 1,000 prompts; bases flat 62.8 / hier 64.0, matching test):** adamw step at η=Λ/16, Λ/4, Λ/2, Λ: flat +2.7 / **+6.3** / +7.3 / −4.5; hier +1.1 / **+5.7** / +3.9 / −1.8; random ≈ 0; raw step destroys both. Same conclusion as the easiest-1,000 version: the critic's initial direction is as good for flat as for hier.

Progress: cluster-blind critic hier run at iter 250: 67.8 (full hier: 66.1). Countdown c16-critic flat at iter 250: 25.0 (old critic: 26.1). Paper-budget (16k) critic for flat: best epoch 5, held-out AUC by k no better than the 8k critic (0.93 @k8, 0.96 @k24, 0.98 @k40 vs 0.92/0.98/0.97) — the critic budget was not what limited flat.

## 2026-10-26 evening — Countdown with the paper's critic sampler holds; big-critic flat run is not collapsing (so far)

**Countdown, critic pretrained on γ4@16 states (CFRL recipe), SVG seed 0, test acc:** flat 17.6 → 25.0 / 23.4 / 23.5 (iters 250/500/750, running) vs the old γ56@64-critic run 26.1 / 18.1 / 15.3 → 10.5. Hier 18.7 → 25.2 / 21.9 (500) vs old 24.6 / 12.5 → 14.3. The critic-sampler fix removes most of the Countdown collapse for *both* models. If it holds to 1500, Countdown becomes a task where the plain model gains under pathwise RL and the hier advantage is small or nil — to be reported as such.

**Paper-budget critic (16k rollouts), flat Game24-cd, SVG seed 0:** 63.9 → 64.1 / 63.8 at iters 250/500, where the 8k-critic flat run (seed 0) was at 53.3 / 45.9. Held-out AUC of the two critics is indistinguishable, so if this run stays flat-or-up to 1500 the flat collapse is sensitive to the particular critic (epoch 5 of the 16k set vs epoch 6 of the 8k set), not just to the model. This would weaken "flat collapses" to "flat is fragile" (3/3 seeds collapsed with the 8k critic; GRR and one-step both fine). Must wait for 1500 and then run the 16k critic over seeds 1–2 before claiming anything either way.

**Cluster-blind critic (hier, Game24-cd):** 64.9 → 67.8 / 68.5 / 68.9 (750) — tracks the full hier run (66.1 / 70.8 / 70.3). The critic does not need to see the cluster channels; consistent with the 2–3% input-gradient share measured earlier.

**AdamW SFT chains are diverging:** with `blr 0.002` the trainer scales to an absolute lr of 4e-3 at batch 512 (lr = blr × batch/256); CE rises from 0.014 (step 1k) to 1.36 (step 13k) in both elfA and helfA while the latent L2 term stays flat — the token head blows up at peak lr. CFRL's "AdamW lr 2e-3" is an absolute lr. Cancelling and resubmitting with `blr 0.001` (absolute 2e-3 at batch 512).
Resubmitted 2026-10-26 20:30 with `blr 0.001` (absolute lr 2e-3 at batch 512): elfA chain 13455274–81, helfA chain 13455282–89 (sft-cd → sft-g24cd → gate0a → gate0b/gate0c ×2; countdownA c16 critic → SVG). Diverged outputs deleted. ~2 days to the first RL result.

**21:40 update.** Big-critic flat (Game24-cd, seed 0): 63.9 → 64.1 / 63.8 / 65.8 / **60.1 / 60.2** (iters 250–1250) — the collapse arrives late rather than not at all (8k-critic seed 0 bottomed at 45.9 by iter 500). Final at 1500 pending; so far it reads as "a better critic delays the flat collapse, does not prevent it." Cluster-blind critic hier run final: 64.9 → 67.8 / 68.5 / 68.9 / 67.5 / 67.1 / **64.8** — peaks like the full hier run (+4) but gives it all back in the last 250 iters; full hier run ended +5.2. n = 1; either the critic reading the clusters matters for *holding* the gain, or this is one seed's late dip (the full hier run also dipped to 68.9 at 1250). Needs seeds. Countdown c16-critic finals: flat 17.6 → **20.2 (+2.6)**, hier 18.7 → 18.3 at 1250 (−0.4) — no collapse for either, modest gain for flat, none for hier: Countdown is not a task where the coarse stream helps under this recipe. AdamW chains still queued (not started). helfR SFT step 29.5k / 58.6k.
Queued 21:50: cluster-blind critic hier seeds 1–2 (`game24cd_hier_blind_svg2d_s{1,2}`) and 16k-critic flat seeds 1–2 (`game24cd_flat_bigcrit_svg2d_s{1,2}`), reusing the existing critics.

**Big-critic flat final (Game24-cd, seed 0, 16k-rollout critic): 63.9 → 59.5 (−4.5)**, trajectory 64.1 / 63.8 / 65.8 / 60.1 / 60.2 / 59.5. The 8k-critic flat runs were −9.2 / −14.5 / −13.0. A critic trained on twice the data delays and roughly halves the flat collapse but does not remove it; hier with the 8k critic is +4.8 (3 seeds). Seeds 1–2 queued (13455981/83). Hier with the 16k critic queued (13448952).

Dashboard: progress bars and time-left estimates added for live RL runs (from the log's elapsed/iter), SFT (steps/sec from the log), critics (time per epoch) and queue jobs (time to limit; Slurm's estimated start for pending jobs).

**04:35 2026-10-26.** Hier 16k-rollout critic pretrain (13448951) finished: best epoch 2, field score 0.821 (flat 16k critic: best epoch 5). SVG with it (13448952) has a Slurm start estimate of ~13:20. Random-cluster SFT at step 55.0k / 58.6k (loss 0.0655, ce 0.0086); gate0a 13445211 follows. AdamW chains still estimated to start ~18:00.

## 2026-10-26 11:00 — seed repeats at iter 750; AdamW SFTs healthy at halved lr

Random-cluster SFT (helfR, Countdown) finished 05:26 (final loss 0.0655, ce 0.0086); its critic pretrain 13445211 and the three rand SVG seeds are still queued.

Seed repeats, Game24-cd test acc at iters 0 / 250 / 500 / 750 (all running, finish tonight):

| run | 0 | 250 | 500 | 750 |
|---|---|---|---|---|
| hier, 16k critic, s0 | 64.9 | 65.4 | 68.8 | 67.0 |
| flat, 16k critic, s1 | 64.7 | 68.1 | 51.0 | 58.5 |
| flat, 16k critic, s2 | 65.1 | 64.0 | 50.4 | 63.9 |
| hier, cluster-blind critic, s1 | 64.3 | 68.7 | 69.7 | 68.8 |
| hier, cluster-blind critic, s2 | (64.9) | 67.7 | 66.9 | 67.3 |

Reading so far: with the 16k critic the flat model still drops ~14 points by iter 500 on both new seeds (seed 0 only dipped at 1000+), so the bigger critic does not remove the flat collapse; the iter-750 rebound is the usual noisy flat trajectory. Both cluster-blind hier seeds sit at +3 to +5 through 750, like the full-critic hier runs; whether they hold to 1500 (seed 0 gave it back) is the open question.

AdamW chains at blr 0.001 (absolute lr 2e-3): sft-cd-elfA and sft-cd-helfA, 4.5 h in, ce 0.05 / 0.067 and falling — no divergence (the blr 0.002 runs had ce 1.25+ by step 14k). Steps/sec 1.15; full 58.6k-step SFT ≈ 14 h, so the Game24 SFTs and RL chains behind them run tomorrow.

Ops note: /scratch1 reads from the start of large files stalled for ~15 min around 10:50 (tail/seek reads fine); `status.py --remote` should be run with a timeout.

## 2026-10-26 12:40 — cluster-blind critic seeds and 16k-critic flat seed 1 final

Game24-cd test acc, iters 250 / 500 / 750 / 1000 / 1250 / **1500** (bases in parentheses):

| run | trajectory | final | Δ |
|---|---|---|---|
| hier, cluster-blind critic, s1 (64.3) | 68.7 / 69.7 / 68.8 / 68.4 / 66.4 / **67.0** | 67.0 | +2.7 |
| hier, cluster-blind critic, s2 (64.9) | 67.7 / 66.9 / 67.3 / 68.7 / 67.7 / **67.6** | 67.6 | +2.7 |
| hier, cluster-blind critic, s0 (64.9) | 67.8 / 68.5 / 68.9 / 67.5 / 67.1 / **64.8** | 64.8 | −0.1 |
| flat, 16k critic, s1 (64.7) | 68.1 / 51.0 / 58.5 / 61.8 / 62.0 / **61.1** | 61.1 | −3.6 |
| flat, 16k critic, s2 (65.1) | 64.0 / 50.4 / 63.9 / 58.0 / 54.7 / … | running | |
| hier, 16k critic, s0 (64.9) | 65.4 / 68.8 / 67.0 / 61.9 / 64.4 / … | running | |

**Cluster-blind critic, 3 seeds: mean +1.8 (−0.1 / +2.7 / +2.7)** vs +4.8 for the full-critic hier runs and −11.7 for flat. A critic that cannot see the coarse channels keeps the hier model from collapsing but roughly halves the gain; all three seeds peak at +4 to +5 by iter 500–1000 and drift down afterwards. So the critic reading the code is not necessary for protection, but it does help *hold* the gain. (The critic's input gradient puts 2–3% of its norm on the coarse channels — small, but evidently not nothing over 1500 iterations.)

**16k critic, flat:** seed 0 −4.5, seed 1 −3.6, seed 2 at 54.7 and falling at iter 1250. The paper-budget critic roughly halves the flat collapse; it does not remove it, and every seed still shows the characteristic −14-point dip at iter 500.

**16k critic, hier (s0):** dipped to 61.9 at iter 1000 then 64.4 at 1250 — the first hier run to go below base mid-run. Final pending; if it ends below base it will be one seed against three full-critic seeds at +4.8 and needs seeds before any reading.

AdamW SFTs: elfA step 16.1k ce 0.053 (steady); **helfA step 24.7k ce 0.104, drifting up** (0.05 at 12k, 0.067 at 18k). Not a divergence yet (the blr-0.002 runs reached 1.3), but the hier CE head is climbing at peak lr. Watch next check; if CE passes ~0.3 the chain's Game24 SFT and RL will start from a worse token head and the AdamW comparison gets muddier.

**12:55 2026-10-26 — 16k-critic finals.** Flat s2: 56.2 (−8.9; traj 64.0 / 50.4 / 63.9 / 58.0 / 54.7 / 56.2). **Flat with the 16k critic, 3 seeds: −4.5 / −3.6 / −8.9, mean −5.7** (8k critic: −11.7). Hier with the 16k critic, s0: **65.1 (+0.2)**; traj 65.4 / 68.8 / 67.0 / 61.9 / 64.4 / 65.1 — peaks at +4 like every hier run, then gives it back; one seed.

Reading across every SVG-2D run on Game24-cd so far (final − base): hier runs {full critic +5.2/+3.9/+5.4; blind −0.1/+2.7/+2.7; 16k +0.2} — min −0.1, none below base. Flat runs {8k −9.2/−14.5/−13.0; 16k −4.5/−3.6/−8.9} — max −3.6, none at or above base. The cleanest statement of the finding is therefore about the *floor*: the hierarchical policy never ends below its base under the pathwise update (7/7 runs), the flat policy always does (6/6), and the size of the hier gain depends on the critic (full critic +4.8, blind +1.8, 16k +0.2 n=1). AdamW SFTs: elfA 17.0k ce 0.058, helfA 25.6k ce 0.095 (eased from 0.104).

Queued 2026-10-26 13:55: coarse-rail conjecture tests (`hcfrl/analysis/consistency.py`, `scripts/consistency.slurm`; jobs 13465766 probe / 13465767 trace). Probe: at the base policy and critic (flat, hier, hier-blind), displace the clean estimate x_k at k∈{24,40} and measure the change in the pooled critic logit and the fraction of decodes that change — coarse channels re-set / set to a wrong class / zeroed; token channels moved by ε∈{0.02,0.05,0.1,0.2} of their norm in a random direction or along the critic pull P_tok G; the full pull G. Trace: coarse–token consistency (argmax coarse == Γ(decoded token)), margin and token-norm at k∈{24,40,63} for every policy_it checkpoint of the hier s0 runs: full, clusters-only (collapsed), tokens-only, blind, 16k.

## 2026-10-26 14:30 — critic probe refutes the "critic rejects inconsistency" half of the rail conjecture; hier AdamW SFT diverged again and was restarted at 1e-3

**Critic probe (`cons_probe_*.json`, base policies/critics, 256 val prompts, k = 24 / 40; Δ pooled logit, mean over sequences; base logits flat 796 / 796, hier 569 / 516, hier-blind 540 / 481):**

| displacement of the clean estimate x_k | flat critic | hier critic | hier-blind critic | decoded tokens changed |
|---|---|---|---|---|
| coarse channels → wrong class | – | −6.3 / −4.1 | 0 / 0 | 0.1% |
| coarse channels → 0 | – | −5.4 / −3.6 | 0 / 0 | 0 |
| coarse channels → re-set to Γ(decode) | – | +0.5 / +0.3 | 0 / 0 | 0 |
| token channels, random dir, 5% of norm | −0.3 / +0.1 | +0.9 / 0.0 | +0.6 / +0.7 | 0 |
| token channels along critic pull, 2% | **+138 / +131** | **+124 / +131** | +144 / +156 | 0.1% |
| token channels along pull, 5% | +212 / +207 | +215 / +231 | +237 / +260 | 0.1–0.2% |
| token channels along pull, 10% | +208 / +221 | +261 / +275 | +280 / +308 | 0.3–0.5% |
| token channels along pull, 20% | −232 / −166 | −95 / +18 | −147 / −38 | 0.7–0.9% |
| full pull (all channels), 2–10% | – | ≈ token pull (±2) | ≈ token pull | same |

Coarse share of the pull ρ_cl = 0.028 / 0.030 (hier), 0 (blind).

Reading: (1) **both critics have a decode-preserving ascent direction of the same size** — a 2% displacement of the token channels along the pull raises the pooled logit by 17–25% while changing 0.1% of decoded tokens, for flat and hier alike. (2) **The hier critic is nearly indifferent to code–token consistency**: a wrong class on every position costs ~1% of the logit, twenty times less than a 2% token nudge gains. (3) Full pull ≈ token pull. So the conjecture's "critic rejects inconsistent states" half is false, and the critics are equally exploitable in state space. Whatever protects the hier policy is in its parameter-to-state map ∂d_θ/∂θ (how much of this state-space ascent a parameter step can realise and what else it drags along), not in the critic. Consistent with the base-gradient norms (hier ≈ half of flat at k=8) and with the hier runs rising more slowly. Revised, testable statement: per normalised actor step, the hier policy realises a smaller displacement along the pull direction than the flat policy.

**Consistency trace (partial, full-critic hier s0):** consistency ≥ 0.98 at every checkpoint; margin decays 1.03 → 0.85; **token-channel norm inflates 21.7 → 25.5 (+17%) over 1250 iters** (the signature of moving along the pull). Collapsed-run comparison (clusters-only, blind, 16k) and flat token-norm trace pending (trace script extended to flat runs).

**AdamW hier SFT (sft-cd-helfA 13455282) diverged at absolute lr 2e-3:** CE 0.09 (25.9k) → 0.17 (29.1k) → 0.84 (29.9k) → 0.68 (30.6k); the flat twin is fine (CE 0.065 at 22k). The hier CE head cannot take the CFRL lr under AdamW (second time). Action: checkpoint_29280 set aside (`diverging_checkpoint_29280`), chain 13455282–89 cancelled, `configs/helfA_b_countdown.yml` blr 0.0005 (absolute 1e-3) and `helfA_b_game24_cd.yml` blr 0.001 (absolute 1e-3), trainer patched so a resumed run takes the configured lr (`hcfrl/elf/train.py`: the checkpoint otherwise restores the old lr through the optimizer state and LambdaLR base_lrs), resumed from checkpoint_19520 (CE 0.067). New helfA chain 13466271–78. The flat chain (published lr) is untouched: it is the one that answers "does the published recipe collapse"; the hier AdamW model is a comparison partner trained at the largest lr its head tolerates.

## 2026-10-26 15:00 — pushing for significance: mechanism test, more seeds, second model size, hierarchy-design control

Queued/submitted:
- **Jacobian-step test** (`hcfrl/analysis/jacobian_step.py`, `scripts/jacobian.slurm`, job 13467280): one AdamW-normalised actor step of η = Λ/16, Λ/4, Λ/2 on each base policy; on a fixed probe set of states measure the displacement of the clean estimate along the critic's decode-preserving pull (fraction of token norm), total displacement, cosine with the pull, Δlogit, decoded-token changes, and for hier the coarse-channel movement and consistency. Direct test of the revised conjecture (hier realises less of the pull per step).
- **Hier with the 16k critic, seeds 1–2** (13467288/89): the +0.2 at n = 1 needs seeds before the "gain depends on the critic" statement is quantified.
- **Second model size, ELF-M** (depth 24, hidden 1056, vs ELF-B depth 12 / 768; ≈3× parameters): configs `{elfM,helfM}_b_{countdown,game24_cd}.yml` (same recipe, `model: ELF-M`, init ELF-M-owt-torch); Countdown SFT → Game24 SFT submitted for both (flat 13467300→01, hier 13467302→03); gate0a/gate0b to be submitted when the per-user job cap (24) allows. SFT at this size may exceed the 40 h limit; the trainer auto-resumes from the 10-epoch checkpoints, so a timed-out job is resubmitted, not lost.
- **Hierarchy-design control** (prepared, not yet submitted): `helfK_b_*` = H-ELF with the generic k-means map `clusters_t5small_n16.pt` instead of Γ_task on Game24-cd. With Γ_task (main), Γ_kmeans (design) and Γ_random (control) the paper can say whether the content of the code matters.
- Cancelled to make room under the cap: Countdown-AdamW critic+SVG for both models (13455280/81, 13466277/78; Countdown is the null task) and GRR at the paper's lr on the AdamW models (13455279, 13466276).

Trace job 13465767 is at 26 of 35 checkpoints with 9 min left on its limit; results are read from the printed lines, the missing checkpoints will be rerun.

## 2026-10-26 15:10 — consistency trace: the run that collapsed is the one that lost code–token consistency

`cons_trace` (256 val prompts, deployment sampler; values at k = 24 / 40 / 63, i.e. clean estimate x_{k−1}; 3rd column = token-channel norm per position, base ≈ 21.6):

| hier run (Game24-cd, s0) | final acc | consistency it0 → it1500 | margin it0 → it1500 | token norm it0 → it1500 |
|---|---|---|---|---|
| full critic (held: +5.2) | 69.9 | 0.996 → 0.985 | 1.03 → 0.83 | 21.7 → 25.9 (+19%) |
| tokens-only gradient (held: +3.4) | 68.3 | 0.990 → 0.974 | 1.01 → 0.90 | 21.6 → 26.8 (+24%) |
| coarse-blind critic (gave back: −0.1) | 64.8 | 0.991 → 0.979 (it1250) | 1.02 → 0.88 (it1250) | 21.6 → 27.7 (+28%, it1250) |
| **clusters-only gradient (collapsed: −1.7 after +6.4 peak)** | 63.2 | **0.990 → 0.978 (it500) → 0.725 (it750) → 0.58** | **1.02 → 0.82 (it500) → 0.31 → −0.03** | **21.6 → 23.2 (it500) → 30.9 → 36.5 (+69%)** |

(16k-critic hier run and the blind run's it1500 were cut by the 1 h limit; flat token-norm traces not yet run.)

Reading: every run inflates its token latents under the pathwise update (the policy does move along the pull), by 19–28% in the runs that hold their gain. The one run that collapsed is the one whose coarse channels stopped agreeing with its tokens: consistency falls from 0.98 at the +6.4 peak (it500) to 0.58, the margin goes to zero and below (the coarse channels no longer point at the decoded token's class at all), and the token norm inflates 69%. The order of events matters: at it500 the clusters-only run is at its peak accuracy with consistency already slipping (0.978, margin 0.82 — lowest of the four at that iteration), and the accuracy collapse follows over the next 250–500 iterations. Among the three runs that hold, the one that gave its gain back at the very end (blind critic) has the largest norm inflation and the lowest margin at it1250. So: **coarse–token consistency is a leading indicator of health under the pathwise update, and the policy whose coarse channels are driven directly by the critic loses it.** This is the mechanism statement the paper can make from data: the coarse stream protects as long as it stays coupled to the tokens; drive it on its own and the coupling breaks and the collapse follows. Combined with the probe (the critic itself is indifferent to consistency), the coupling must be enforced by the policy's own denoiser (trained with the joint loss), not by the critic.

Next: Jacobian-step test (13467280, queued) for the per-step displacement; second trace job for the flat runs' token norms (a flat run that collapses with an even larger inflation would complete the picture) and the two cut checkpoints.

**15:30 — flat token-norm traces (trace2, partial).** Flat s0 (8k critic; acc 64.2 → 53.3 / 45.9 / 59.7 / 58.6 / 57.9 / 55.0): token norm 21.6 → 21.5 (it250) → **30.5 (it500, +41%)** → 25.5 → 25.3 → 26.8 → 27.4. Flat 16k-critic s0 (acc 64.1 / 63.8 / 65.8 / 60.1 / 60.2 / 59.5): 21.6 → 22.8 → 23.2 → 24.5 → 27.5 (it1000) → 28.4 → 28.7 (+33%). The norm spike of the 8k-critic flat run lands exactly on its accuracy trough (it500), and the 16k-critic run's slower inflation lands on its later, smaller collapse (it1000+). The hier runs that hold inflate 19–24% over the whole run with no spike. So latent-norm inflation along the pull is the common signature of the collapse in both models, it is faster and larger in the flat model, and in the hier model it is accompanied by code–token consistency except in the one run where the critic drove the code directly. Random-cluster critic pretrain (13445211) completed within its limit; the three rand SVG seeds are queued (priority). Jacobian rerun 13467842 queued (A40).

**15:40 — trace2: the hier 16k-critic run (the one that dipped below base and ended +0.2) also lost consistency.** `game24cd_hier_bigcrit_svg2d_s0` (acc 64.9 → 65.4 / 68.8 / 67.0 / **61.9** / 64.4 / 65.1): consistency 0.996 → 0.990 → 0.985 → **0.843 (it750) → 0.508 (it1000)** → 0.589 → 0.676; margin 1.03 → 1.02 (it500) → 0.71 → **0.12** → 0.22 → 0.33; token norm 21.7 → 25.4 (it500) → 31.5 → **42.2 (+95%)** → 38.7 → 36.1. The accuracy dip (it1000) coincides with the consistency minimum and the norm maximum, and the partial recovery of accuracy (64.4, 65.1) with a partial recovery of consistency (0.59, 0.68). That is the second hier run (after clusters-only) in which a loss of code–token consistency accompanies a loss of accuracy, against four hier runs that hold consistency ≥ 0.97 and hold their gain. Consistency is the health variable; the 16k critic is a critic whose pull breaks the coupling on this seed (its seeds 1–2 are queued).

Jacobian test, flat (13467842, partial): per AdamW-normalised step η = Λ/16, Λ/4, Λ/2 — displacement along the pull 0.11%, 0.49%, 1.06% of the token norm; total token displacement 1.8%, 5.6%, 11.3%; cos(Δx, pull) 0.06–0.10; decoded tokens changed 0.6%, 2.4%, 6.6%; Δlogit +8, +38, +75. A random parameter direction of the same size moves the state by 1.1% in every case with no logit change and 0.3% decode changes. Hier pending.

## 2026-10-26 15:50 — Jacobian-step test: per normalised actor step the two policies move identically; the revised conjecture is refuted too

`jacobian_step` (base policy + own critic; mean pathwise gradient over 128 prompts × 8 rollouts × k∈{8,16,24,32}; one AdamW-normalised step of η = Λ/16, Λ/4, Λ/2; fixed probe set of 128 val prompts at k = 24 / 40; values at k = 24, k = 40 in parentheses):

| per step | flat Λ/16 | hier Λ/16 | flat Λ/4 | hier Λ/4 | flat Λ/2 | hier Λ/2 |
|---|---|---|---|---|---|---|
| displacement along the pull (% of token norm) | 0.11 (0.13) | 0.14 (0.13) | 0.49 (0.50) | 0.57 (0.51) | 1.06 (1.08) | 1.15 (1.04) |
| total token displacement (%) | 1.8 (1.7) | 2.0 (1.9) | 5.6 (5.1) | 6.6 (6.2) | 11.3 (10.7) | 12.8 (12.1) |
| cos(Δx, pull) | 0.064 | 0.067 | 0.092 | 0.085 | 0.096 | 0.089 |
| Δ critic logit | +8.1 (+10.6) | +8.9 (+7.2) | +38.5 (+38.0) | +35.7 (+27.6) | +74.9 (+84.1) | +68.3 (+50.9) |
| decoded tokens changed (%) | 0.61 (0.44) | 0.39 (0.44) | 2.39 (1.26) | 1.78 (1.57) | 6.61 (3.57) | 4.00 (2.78) |
| hier: coarse-channel movement (% of β) | – | 1.6 | – | 5.8 | – | 11.4 |
| hier: consistency after the step | – | 0.978 | – | 0.974 | – | 0.964 |
| random direction, same η: total / Δlogit | 1.1% / −0.3 | 1.1% / −0.7 | 1.1% / −0.1 | 1.1% / −0.3 | 1.1% / 0.0 | 1.1% / −0.2 |

Λ = 1.17 (flat) / 1.54 (hier). **Refuted:** per normalised step the hierarchical policy realises the *same* displacement along the pull, the same total displacement and the same critic gain as the flat policy (differences ≤ 15%, within the k-to-k spread). Its coarse channels move with the tokens (6–11% of β at Λ/4–Λ/2) and consistency dips only slightly (0.978 → 0.964). The one per-step difference: the same displacement changes fewer decoded tokens in the hier model (0.4 / 1.8 / 4.0% vs 0.6 / 2.4 / 6.6%) — its decode is somewhat more robust to the displacement the critic asks for — but that is a 1.5× effect, not the kind of gap that explains −12 vs +5.

So: the critic's response to states is the same (probe), the single-step parameter-to-state map is the same (this test), the gradient path is irrelevant (ablation). Everything that distinguishes the models is in the *iterated* loop (actor drift + buffer + critic refit over hundreds of iterations), and the trace is what describes it: the hier policy's latent inflation saturates at +19–24% with its coarse code intact, while the flat policy inflates 41% in 250 iterations and collapses; the two hier runs that did lose their code (clusters-only, 16k critic) collapsed the same way. Mechanism statement supported by the data: *code–token consistency is the health variable of the pathwise update on the hierarchical model; it is maintained by the policy's own denoiser (not the critic, not the gradient path, not a per-step difference), and the collapse is the regime in which it is lost.* Why the joint-trained denoiser resists the drift over many steps while a single call does not is the open question; candidates are the self-conditioning feedback over the 64-step rollout (the trunk was trained to restore consistent joint states and may pull a drifted state back on later calls, which a one-call Jacobian cannot see) and the critic refit on consistent vs. inconsistent buffer states. A rollout-level version of this test (displace, then run the remaining steps, measure how much of the displacement survives to decoding, flat vs hier) would decide between them.

**16:00 — trace2 complete.** Coarse-blind hier s0 at it1500 (final 64.8, −0.1): consistency 0.935 (k24) / 0.964 / 0.975, margin 0.79 / 0.86 / 0.88, token norm 28.7 (+33%). The run that gave its gain back in the last 250 iterations also dropped from 0.979 to 0.935 consistency over those iterations — the lowest of any run that did not collapse. Ordering by final consistency at k24 now matches ordering by final Δacc across all six hier checkpoints at it1500: full 0.985 (+5.2) > tokens-only 0.974 (+3.4) > blind 0.935 (−0.1) > 16k 0.676 (+0.2, after its dip) > clusters-only 0.58 (−1.7). Submitted the rollout-survival test (13469923): displace x_k along the pull at k = 24 / 40 by 5–10%, finish the rollout with identical noise, measure accuracy, decoded-token change and the critic-logit gap at k, k+8, k+16, k+32 — does the hier denoiser heal the displacement over subsequent calls where the flat one does not?

## 2026-10-26 16:25 — rollout-survival test: both models heal a one-shot displacement the same way; the mechanism is not visible in any single-shot probe

`consistency --mode survive` (base policy + own critic; 256 val prompts; displace x_k along the pull by 5% / 10% of the token norm or 5% in a random direction, then finish the rollout with the same noise as the undisplaced control):

| k = 24 | flat: pull 5% | flat: pull 10% | flat: rand 5% | hier: pull 5% | hier: pull 10% | hier: rand 5% |
|---|---|---|---|---|---|---|
| Δlogit at k (immediately) | +235 | +265 | +3 | +199 | +240 | −2 |
| Δlogit at k+8 / k+16 / k+32 | +14 / +19 / +8 | −1 / +9 / −18 | +3 / +22 / −9 | +33 / +27 / +4 | +32 / +33 / +20 | +21 / +21 / +22 |
| final decoded tokens changed vs control | 9.1% | 10.3% | 9.8% | 7.6% | 7.6% | 6.3% |
| accuracy (control 60.9 flat / 63.7 hier) | 60.9 | 59.4 | 61.3 | 64.5 | 66.0 | 66.4 |
| hier: final consistency (control 0.997) | – | – | – | 0.998 | 0.998 | 0.997 |

k = 40 is the same picture (flat +260 → +31 / +23; hier +222 → +19 / +25; accuracies unchanged within noise).

Reading: a critic-pleasing displacement that inflates the pooled logit by 35–45% at step k is reduced to ≤ 10% of that within 8 further denoiser calls, in BOTH models, and leaves accuracy unchanged (±3 points is the noise of 256 prompts); it changes 7–10% of decoded tokens, no more than a random displacement of the same size does, so by the end the displacement is just noise. The hier policy's coarse channels stay consistent (0.998). So: the critic responds the same to states (probe), a parameter step moves the state the same (Jacobian), and the sampler heals a one-shot displacement the same (survival). **No single-shot probe at the base distinguishes the two models.** What distinguishes them is the accumulated drift over hundreds of actor updates (the trace: consistency held in every hier run that held its gain, lost in both that did not; latent inflation 19–24% vs 41% for flat), i.e. a property of the closed loop (actor + buffer + critic refit), not of the policy or critic at the base. This is the honest mechanism statement; a per-iteration trace of a flat and a hier run (critic AUC on buffer, buffer success, latent norm, consistency, every 50 iterations) is the right next measurement, and `log.jsonl` already has buffer success / AUC / objective per iteration for every run — to be plotted against the checkpoint-level trace.

Other items from this check: (1) **random-cluster seeds started** (13445212–14, 28 min in); iter-0 test acc **69.4 / 69.9 / 70.5** — the random-code hier SFT (val-selected checkpoint) has a base ~5 points above flat (64.1) and Γ_task hier (64.9). Base-rate difference to be kept in mind when reading its RL result; the comparison that matters is Δ from base and whether it collapses. (2) Resumed hier AdamW SFT at 1e-3: CE 0.0165 at step 21.6k (vs 0.05–0.10 at 2e-3) — healthy. (3) Hier 16k seeds 1–2 running. (4) ELF-M SFTs failed at start: `HF_HUB_OFFLINE=1` and ELF-M-owt-torch not in the cache → snapshot downloaded, chains resubmitted. (5) Survival test done; checkpoint backup to ~/hcfrl_backup at 47 GB (rl 34 GB, sft 8 GB, critic 5.5 GB), still running; HF upload follows automatically.

**17:05 — random-cluster control, first evals (Game24-cd, SVG-2D, Γ_random H-ELF; base is val-selected and sits at ~70, five points above flat 64.1 / hier 64.9):** s0 69.4 → 69.6 (it250) / 69.0 (it500); s1 69.9 → 70.6 / 69.0; s2 70.5 → 68.4 (it250). No early gain, no early collapse: flat at the same iterations was 53.3 / 45.9 (s0), Γ_task hier 66.1 / 70.8. Too early to call; if H-ELF-R ends near its base while flat collapses and Γ_task hier gains, the reading is "any jointly-denoised coarse stream prevents the collapse; a task-informed one also yields the gain". Hier 16k-critic seeds: s1 64.6 → 64.6 / 66.5 (it500); s2 64.8 → 66.3 (it250). Flat AdamW SFT (published lr 2e-3): CE drifting 0.065 (22k) → 0.10 (34k) — the same early signature the hier run showed before its blow-up; if CE passes 0.3 it will be resumed from checkpoint_29280 at 1e-3, which would put both AdamW models at 1e-3 (to be reported as such). ELF-M SFTs failed again: flat with `No module named 'yaml'` (first job under the uv env — being diagnosed), hier with CUDA OOM at batch 512 on a 44 GB A40 (needs gradient accumulation). HF upload had failed on `HF_HUB_OFFLINE=1` (inherited from common.sh); restarted with it unset.

**17:50 — rand control at it750; flat AdamW SFT diverged at the published lr and was restarted at 1e-3.** Random-cluster seeds (base 69.4 / 69.9 / 70.5): s0 69.6 / 69.0 / **64.0** (−5.4 at it750); s1 70.6 / 69.0 / 70.3 (+0.4); s2 68.4 / 67.9 / **66.2** (−4.3). Two of three seeds are now below base by 4–5 points at it750; the Γ_task hier seeds at it750 were 70.3 / 68.8-ish / 70.x (all +4 to +6) and the flat seeds 59.7 / ~52 / ~52. So far H-ELF-R sits between the two: no 15-point crash, but drifting down. Finals at 1500 (~4 h). Hier 16k-critic seeds at it750: s1 66.8 (+2.1), s2 67.4 (+2.6). Flat AdamW SFT at the published 2e-3: CE 0.10 (34k) → **0.38 (37.1k)** — the same CE-head divergence as the hier run, ~7k steps later. Cancelled 13455274–78, config `elfA_b_countdown.yml` blr 0.0005 (1e-3) and `elfA_b_game24_cd.yml` blr 0.001, resumed from checkpoint_29280 (CE ≈ 0.08) — chain 13474329–33 (sft-cd → sft-g24cd → gate0a → gate0b + gate0c lr 3e-7). Consequence for the paper: **the published AdamW lr 2e-3 diverges the ELF decoder head for both models in our setup**; both AdamW-recipe models are trained at 1e-3, and "does the published recipe collapse" becomes "does an AdamW-trained flat model collapse" — still the question that matters (optimizer geometry), with the lr deviation reported. HF upload of the 47 GB checkpoint backup completed (private repo vxnuaj/hcfrl-checkpoints, 29 min). ELF-M SFTs running at 0.7 micro-steps/s (256×2) — ~46 h of work against a 40 h limit; will need one resubmission to auto-resume. helfK (k-means) SFT failed at data loading (`load_from_disk(None)`), being diagnosed.

**18:30 — rand control to it1250: hovers around base, no collapse, no gain.** s0 (69.4): 69.6 / 69.0 / 64.0 / 69.2 / 69.3; s1 (69.9): 70.6 / 69.0 / 70.3 / 69.3 / 66.0; s2 (70.5): 68.4 / 67.9 / 66.2 / 70.6. Excursions of −4 to −5 at single evaluations that recover by the next — noise-level wandering around base rather than a trend, against flat's monotone −12 to −18 by it500 and Γ_task hier's +4 to +6 held. Hier 16k seeds: s1 66.7 / 67.6 (it1000 / 1250, +3.0); s2 66.0 (it1000, +1.2). SFTs all healthy: flat AdamW resumed at 1e-3 (CE 0.043 at 32k), hier AdamW CE 0.014 at 30k, ELF-M both at step ~2.8k (CE 0.001, 0.7 it/s), helfK at 2.4k.

## 2026-10-26 18:35 — closed-loop variables per 250-iteration window (from log.jsonl; buffer success = verified success of the fresh rollouts entering the replay buffer; critic AUC on the newest buffer states; actor objective = pooled critic logit at the actor's states)

| run (s0, Game24-cd) | variable | 0–250 | 250–500 | 500–750 | 750–1000 | 1000–1250 | 1250–1500 | final Δacc |
|---|---|---|---|---|---|---|---|---|
| flat | buffer success | 0.82 | **0.59** | 0.72 | 0.79 | 0.78 | 0.77 | −9.2 |
| flat | critic AUC | 0.97 | 0.93 | 0.98 | 0.99 | 0.98 | 0.97 | |
| flat | actor objective | 82 | **6.7** | 3.6 | 6.9 | 7.9 | 16 | |
| hier (full) | buffer success | 0.86 | 0.85 | 0.88 | 0.89 | 0.89 | 0.88 | +5.2 |
| hier (full) | critic AUC | 0.98 | 0.98 | 0.98 | 0.97 | 0.97 | 0.97 | |
| hier (full) | actor objective | 155 | 74 | 66 | 73 | 65 | 77 | |
| hier, tokens-only grad | buffer success / objective | 0.85 / 187 | 0.86 / 145 | 0.88 / 125 | 0.89 / 98 | 0.88 / 73 | 0.87 / 47 | +3.4 |
| hier, blind critic | buffer success / objective | 0.87 / 180 | 0.86 / 79 | 0.87 / 75 | 0.87 / 83 | 0.86 / 76 | 0.86 / 47 | −0.1 |
| hier, clusters-only grad | buffer success / objective | 0.88 / 209 | 0.90 / 196 | 0.92 / 213 | 0.89 / 176 | 0.87 / 103 | 0.87 / 61 | −1.7 (test) |
| hier, 16k critic | buffer success / AUC / objective | 0.85 / 0.99 / 310 | 0.85 / 0.98 / 148 | 0.86 / 0.97 / 114 | 0.83 / **0.93** / 40 | **0.78** / 0.96 / **8** | 0.83 / 0.97 / 21 | +0.2 |
| flat, 16k critic | buffer success / AUC / objective | 0.84 / 0.98 / 347 | 0.78 / 0.95 / 61 | 0.83 / 0.99 / 18 | 0.81 / 0.98 / 55 | 0.79 / 0.97 / 111 | 0.78 / 0.97 / 105 | −4.5 |

Reading. The flat collapse has a sharp closed-loop signature: within the first 250–500 iterations the fresh-rollout success drops (0.82 → 0.59), the critic's AUC on the new states dips (0.93) and then recovers because the refit learns that the drifted states are bad, and the actor objective crashes from 82 to ≈4–8 and never recovers — the policy has been pulled into a region that the critic now scores low, and ascending a low, flat critic from there does not bring it back (train success recovers partly to 0.78, test accuracy does not). The hier full run never enters that regime: fresh-rollout success stays 0.85–0.89 and the actor objective stays 65–77 throughout, i.e. the policy keeps finding states that the critic scores high and that are also correct. The 16k-critic hier run (the one that dipped) shows a milder version of the flat signature at exactly its dip (AUC 0.93 in the 750–1000 window, objective 310 → 8, buffer success 0.78), and the clusters-only run is different again: train (buffer) success stays ≥ 0.87 while test accuracy decays — a generalisation loss that goes with its consistency collapse rather than a critic-exploitation crash. Takeaway for the mechanism section: on the flat model the loop passes through an exploitation phase (critic score up, success down) and ends in a low-score basin; the hier parameterisation keeps score and success coupled for the whole run; and the two hier failures are two different events (16k: a brief flat-like excursion; clusters-only: coupling loss without a train-success drop).

**19:10 — first finals of the random-cluster control and the hier 16k seeds.** H-ELF-R (Γ_random) s0: 69.4 → **67.9 (−1.5)**, trajectory 69.6 / 69.0 / 64.0 / 69.2 / 69.3 / 67.9; s1: 69.9 → **65.6 (−4.3)**, 70.6 / 69.0 / 70.3 / 69.3 / 66.0 / 65.6; s2 at it1250: 70.7 (+0.2), final pending. Hier 16k-critic s1: 64.3 → **66.2 (+1.9)**, s2 at it1250 66.0 (+1.2), final pending. All SFTs healthy (flat AdamW at 1e-3: CE 0.023 @34.9k; hier AdamW CE 0.005 @33.2k; ELF-M both at 4.5k, CE 0.002; helfK 5k).

## 2026-10-26 19:20 — random-cluster control final: no collapse, no gain. The coupling prevents the collapse; the task-informed code gives the gain.

**H-ELF-R (Γ_random, same architecture/surgery/SFT/RL recipe), Game24-cd, SVG-2D, 3 seeds, test acc at it1500:**

| seed | base | final | Δ |
|---|---|---|---|
| 0 | 69.4 | 67.9 | −1.5 |
| 1 | 69.9 | 65.6 | −4.3 |
| 2 | 70.5 | 69.9 | −0.6 |
| **mean** | 69.9 | 67.8 | **−2.1** |

Against: flat **−11.7** (3 seeds), Γ_task hier **+4.8** (3 seeds). Trajectories wander ±4 around base with single-eval dips that recover; no monotone collapse (flat is −11 to −18 by it500) and no sustained gain. Caveat: the random-code model's val-selected base (≈70) is 5 points above the flat / Γ_task bases (64–65), so its headroom differs; a base-matched repeat (earlier SFT checkpoint with val ≈ 64) is queued (tag `game24cd_randm`).

**Hier with the 16k critic, 3 seeds:** +0.2 / +1.9 / −0.9, **mean +0.4** (flat with the same critic: −4.5 / −3.6 / −8.9, mean −5.7).

**Pooled, every Game24-cd SVG-2D run (final − base):**
- hierarchical, any code, any critic (12 runs): +5.2, +3.9, +5.4 (Γ_task, 8k) · −0.1, +2.7, +2.7 (blind critic) · +0.2, +1.9, −0.9 (16k critic) · −1.5, −4.3, −0.6 (Γ_random) → **mean +1.2, range −4.3 … +5.4**
- flat (6 runs): −9.2, −14.5, −13.0 (8k) · −4.5, −3.6, −8.9 (16k) → **mean −9.0, range −14.5 … −3.6**

The strict "no hier run below base" floor no longer holds (rand s1 −4.3, 16k s2 −0.9), but the separation does: the hier distribution is centred at +1 with a worst case of −4, the flat distribution at −9 with a best case of −4; one pair overlaps (rand s1 −4.3 vs flat-16k s1 −3.6). Statement for the paper: *a jointly denoised coarse stream, whatever its content, removes the collapse (random code: −2.1 vs −11.7); a task-informed code additionally turns the update into a gain (+4.8); and the size of that gain depends on the critic (full +4.8, blind +1.8, 16k +0.4).* The random control therefore supports the coupling reading of the trace (the two streams have to stay consistent, and any code gives a stream to stay consistent with) and separates "safe" from "useful": safety comes from the parameterisation, usefulness from the content of the code. Next on this axis: Γ_kmeans (queued: k-means over T5 embeddings) and the base-matched Γ_random repeat.

**20:50 — base-matched random control.** `game24cd_randm` selected checkpoint_700 of the Γ_random Game24 SFT (val-target 0.78): test base **58.5** (undershoots the 64–65 target; the SFT's checkpoints are 700 steps apart: 700 → 58.5, 2100 → 70.0 test). Critic pretrained (best epoch 4, field score 0.766); SVG seed 0 running (13476310). A second matched run from checkpoint_1400 (val 0.83, expected test ≈ 65) queued as `game24cd_randm2` (gate0a 13476xxx → gate0b s0), so the random-code model will be tested from three bases (58.5, ≈65, 70). All five SFTs healthy (elfA 1e-3 CE 0.033; helfA CE 0.012; ELF-M ×2 CE 0.004 at ~7.5k; helfK CE 0.005).

**21:55 — base-matched rand: s0 dips and recovers; randm2 not matched (cancelled); flat AdamW diverges a second time.** `game24cd_randm` s0 (base 58.5): it250 **29.4** (−29), it500 **59.7** (+1.2) — a dip-and-recovery of the kind seen in hier-16k s0 (0.985→0.51 consistency at its dip, then back), never in a flat run. Seeds 1–2 submitted (13478794/95, same critic). `randm2`: the val-target selector picked checkpoint_2800 (val 0.832, closest to 0.83; 1400 had 0.842) → test base **69.75**, i.e. the same base as the original rand runs, not the intended ≈65. The Γ_random SFT checkpoints give 58.5 (ckpt 700) and ≈70 (1400–2800) and nothing in between, so the flat/hier base (64–65) is bracketed from below by randm and from above by rand; randm2 cancelled (13477938/39) to free the slots for randm seeds. **Flat AdamW Countdown SFT diverged again at 1e-3**: CE 0.013 (42.5k) → 0.19 (43k) → 0.42 (44k) → 0.24–0.37 (45–46k), while the cosine lr was already ≈2e-4; at 2e-3 it diverged at ~35k (lr ≈ 8e-4). The hier AdamW model at the same 1e-3 is at 44k with CE 0.011. Handling: chain 13474329–33 cancelled before a diverged checkpoint was written; the Game24 fine-tune now starts from **checkpoint_39040** (epoch 40, CE 0.016–0.02) via `--config_override init_from=outputs/elfA_b-countdown/checkpoint_39040` — chain 13478827 (sft-g24cd) → 13478828 (gate0a) → 13478829 (gate0b s0) + 13478830 (gate0c lr 3e-7). Protocol note for the paper: flat AdamW base = 40 Countdown epochs (its CE minimum), hier AdamW base = 60 epochs; the Game24 fine-tune and checkpoint selection are identical. In parallel, the Countdown pretraining continues from 39040 at absolute 5e-4 (13478831, same output dir, resume lr-override) to see whether the flat model can finish 60 epochs under AdamW at all; if it does, a second flat AdamW Game24 chain from the 60-epoch model follows. Other SFTs healthy: helfA CE 0.011 @44k, ELF-M ×2 CE 0.004 @11.2k, helfK CE 0.005 @15k. Queue 23/24.

**22:55 — flat AdamW Game24 fine-tune done; randm seeds dip at it250; continued pretraining healthy.** The Game24 fine-tune from checkpoint_39040 finished in <1 h (2,800 steps, 4 checkpoints); selector picked checkpoint_2800 (val 0.762 vs 0.722 at 2100); gate0a 13478828 is evaluating the base on the test set (≈0.76 after 768 prompts) and will pretrain the critic; gate0b s0 13478829 should start ≈00:30 → first eval (it250) ≈01:15, final ≈04:00. `game24cd_randm` (base 58.5): s0 it750 57.1, it1000 57.4 (−1.1); s1 it250 **44.7**, s2 it250 **45.0** (−14 each, same early dip as s0's it250 29.4 which recovered to 59.7 by it500). Continued flat AdamW Countdown pretraining at 5e-4 (13478831): CE 0.008 at 41.4k, past the point where 1e-3 started to diverge (42.5k) — watching. Others: helfA CE 0.010 @48.4k (finishes ≈01:30), ELF-M ×2 CE 0.004 @13.8k, helfK CE 0.005 @19k. randm2's cancelled gate0a log moved to resolved/. Queue 22/24.

**23:58 — paper draft refreshed on request** (`paper/main.tex`, uncommitted, 17 pp.): random-class control (3 seeds, −2.1) and the base-matched repeat (pending) added to the controls table and text; hier 16k critic 3 seeds (+0.4); pooled separation paragraph (12 hier runs mean +1.2 / worst −4.3 vs 6 flat mean −9.0 / best −3.6) replaces the strict "floor"; Jacobian-step test (Table tab:jac, revised conjecture refuted), rollout-survival test, and closed-loop window table (tab:loop) added to Section 4; discussion/mechanism, limitations, abstract, intro, contributions and conclusion rewritten around "no single-shot probe separates the models; code–token consistency tracks the closed loop"; AdamW paragraph updated with both divergences, the epoch-40 flat base (test 59.9) and the running chains; k-means and ELF-M rows marked queued.

**00:00 2026-10-27 — base-matched random control, seed 0 final: 58.5 → 54.3 (−4.2)**; trajectory 29.4 / 59.7 / 57.1 / 57.4 / 54.8 / 54.3. s1 it500 63.3 (+3.7), s2 it750 55.0 (−3.5, after 60.1 at it500). From the weaker base the random-code model drifts further below base than from the 70 base (−1.5 / −4.3 / −0.6), but still shows no monotone collapse after the it250 dip; flat from a 64 base is −9 to −14.5. Flat AdamW: gate0a done (critic best epoch 6, field score 0.761, in the range of the Muon critics 0.73–0.82); gate0b 13478829 / gate0c 13478830 pending on priority. SFTs: helfA CE 0.008 @52.8k, elfA2 (5e-4) CE 0.0085 @45.8k, ELF-M CE 0.005 @16.4k, helfK CE 0.0055 @23k. runs/ refreshed (52 runs, 130 logs).

**01:00 2026-10-27 — base-matched random seeds falling; elfA2 spike recovered; flat AdamW RL still queued.** `game24cd_randm` s1 (base 59.6): 44.7 / 63.3 / 58.4 / **57.2** (it1000, −2.4); s2 (base 58.5): 45.0 / 60.1 / 55.0 / **49.5** (it1000, −9.0). From the 58.5 base the random-code model is doing worse than from the 70 base (s0 final −4.2; s2 now in flat territory): the "any code prevents the collapse" statement has to be weakened to "softens" unless s2 recovers by it1500 (finals ≈01:45). Continued flat AdamW pretraining at 5e-4 (elfA2): CE spiked 0.007 → 0.083 (47.2k) → 0.076 / 0.048 / 0.058 → back to 0.006–0.011 by 48.8–49.6k — a spike in the same region (43–48k) where the 2e-3 and 1e-3 runs diverged for good, survived at the lower lr; still running toward 58,560. Hier AdamW (1e-3): CE 0.008 @56.9k, finishes within 30 min, Game24 chain follows automatically. Flat AdamW gate0b 13478829 / gate0c 13478830 still pending on priority (1 h). ELF-M CE 0.005 @18.9k; helfK 0.006 @26.7k.

## 2026-10-27 02:10 — flat AdamW SVG run died of fp16 overflow (fixed, resubmitted); base-matched random control collapses; hier AdamW base 68.2

**Flat AdamW SVG s0 (13478829) was NaN from iteration 1**: critic refit loss NaN at the first refit, actor objective/gradient NaN from iteration 2, buffer success 0, test accuracy 0.0 at it250. Cause: the replay buffer stores critic features in **fp16** (`.half()`), and the AdamW-trained ELF trunk has massive activations — features max |h| ≈ **48,500** on stored states (Muon flat: 3,200; mean |h| 340 for both), i.e. at the fp16 edge (65,504); some buffer entries overflow to inf, the head refit goes NaN, the actor objective goes NaN, the clipped step writes NaN into the policy. The AdamW model's weights are also 2.5× larger overall (total norm 3,207 vs 1,251; unembed bias norm 1,383) and its pathwise gradient norm 10× larger (7,473 vs 763 pre-clip). The GRR run on the same base (gate0c 13478830, no feature cache) is healthy: 59.9 → 59.2 / 59.7 at it100/200. Fix (commit below): `train_svg.py --buf_dtype {bf16,fp16}`, default **bf16** (same memory, no overflow; runs before 2026-10-27 used fp16 — recorded in REPRODUCE), plus hard failures on non-finite features or a non-finite actor gradient instead of silently poisoning the policy. Dead run moved to `outputs/rl/game24cdA_flat_svg2d_s0_fp16nan`, log to resolved/. Resubmitted: gate0b 13482902 (pending on priority). The hier AdamW gate0b 13466274 (dependency) will pick up the patched trainer.

**Hier AdamW base: 68.2** (checkpoint_2800, val 0.858) vs flat AdamW 59.9 (checkpoint_2800, val 0.762): an 8-point base gap, from the 60- vs 40-epoch Countdown pretraining. The AdamW comparison is therefore *not* base-matched as it stands; the 60-epoch flat model (elfA2 at 5e-4, CE 0.005 @54.4k, ≈1 h to 58,560) will give the matched flat chain (`game24cdA_flat60`). gate0a-g24cdA-hier 13466273 is collecting rollouts (0.82 so far).

**Base-matched random control (Γ_random from checkpoint_700, bases 58.5 / 59.6 / 58.5), finals:** s0 **54.3 (−4.2)**, s1 **52.5 (−7.2)**, s2 **48.3 (−10.2)**, mean **−7.2**; trajectories 29.4/59.7/57.1/57.4/54.8/54.3 · 44.7/63.3/58.4/57.2/56.7/52.5 · 45.0/60.1/55.0/49.5/50.9/48.3. From the 70 base the same model gave −2.1. So the random code does **not** reliably prevent the collapse: it softens it from the strong base and barely from the weak one (−7.2 vs flat −11.7 from a 64 base). Caveat on the design: the "base-matched" model is an *early* SFT checkpoint (step 700 of 2,800), so maturity and base accuracy are confounded; a flat early-checkpoint control (`game24cd_flat_early`, elf checkpoint_700) is the fair reference and is being queued. Updated pooled picture (Game24-cd, final − base): Γ_task hier 9 runs (full/blind/16k critics) mean +2.4, min −0.9; Γ_random 6 runs mean −4.7, min −10.2; flat 6 runs mean −9.0, max −3.6. Statement: the protection and the gain both come with the **task-informed** code; a random code gives at most a partial protection. This raises the stakes of the k-means control (helfK, SFT at 30.8k/58.6k): a data-derived but meaningful code.

**02:45 — model-level replication queued.** Every Game24 flat-vs-hier number so far compares one trained network per arm (3 RL seeds on the same SFT base); model-to-model variance is untested. Second supervised fine-tune seed (config seed 43) for both models from the shared Countdown bases, each with its own val-selected checkpoint, 8k critic and SVG s0: flat `game24cd_flat_ft43` (13482976 → 13482977 → 13482978, dir outputs/elf_b-game24-cd-s43), hier `game24cd_hier_ft43` (13482979 → 13482980 → 13482981, dir outputs/helf_b-game24-cd-s43). Seed 44 for both to follow when the queue (24/24) frees. Also queued earlier: flat early-checkpoint control (13482923/24). Reassessment written for the user: the headline effect is replicated across RL seeds and across three task pairs; the mechanism section is descriptive (consistency loss is partly a consequence of collapse, not shown to be a cause); the random-control inference is confounded by base strength and checkpoint maturity; and n=1 base per arm is the largest hole — hence this replication.

**03:00 — plan reset and new controls.** Reframing for the paper (discussed with the user): a study of the critic-exploitation failure of pathwise RL on flow LMs (when, what predicts it, what fixes it) with H-ELF as the headline fix, so that a non-collapsing flat AdamW model is a finding (base geometry predicts collapse: AdamW weights 2.5×, activations 15×, gradients 10× the Muon model's) rather than a threat. Priorities: (1) model-level replication (fine-tune seeds 43/44 for all four coarse maps), (2) a competing simple fix — flat + latent-norm penalty (`train_svg.py --norm_pen λ`: actor loss += λ·|obj_0|·mean_gen((‖x_k,tok‖/norm_0 − 1)²), norm_0 and obj_0 from the first actor step; every run now logs `tok_norm` per iteration), (3) bound the critic-strength effect (flat with a heavier refit), (4) base-matched AdamW pair (60-epoch flat: elfA2 finishing at CE 0.0044; sft-g24cd-elfA60 13483525 → gate0a 13483526 queued, gate0b when slots free), (5) early-warning rule from existing logs (buffer success / objective in the first 250 iterations), (6) ELF-M. Dropped: more Γ_random seeds, more base-level probes. Flat AdamW SVG s0 (bf16 fix): 59.9 → **61.8** at it250, finite throughout, it393 at 02:55. Queue 24/24; norm-penalty runs (λ = 10, 100, seed 0) and ft44 chains go in as slots free.

**03:05 — 60-epoch flat AdamW model done; flat-early base 50.6; norm-penalty runs queued.** elfA2 (5e-4 from checkpoint_39040) finished 60 epochs at CE 0.0044 (checkpoint_58560); its Game24 fine-tune 13483525 → gate0a 13483526 → gate0b **13483548** (tag game24cdA_flat60). Flat early-checkpoint control: Muon flat checkpoint_700 base **50.6** (the Γ_random early checkpoint was 58.5), critic pretraining running → gate0b 13482924. Norm-penalty controls on the flat Muon model (same base and 8k critic as the headline runs): `game24cd_flat_svg2d_np10_s0` 13483549 (λ=10) and `np100_s0` 13483550 (λ=100). Hier AdamW gate0a done → gate0b 13466274 pending on priority. Flat AdamW SVG s0: it467, buffer success 0.875, objective 871 → 118 (declining; the Muon flat run's went 82 → 7 by it250), it500 eval imminent. Fine-tune seed 43: both SFTs done, flat critic pretraining, hier gate0a pending on resources. helfK pretraining 34.6k/58.6k (≈7 h more). Resolved-log housekeeping: the four diverged AdamW SFT logs moved to resolved/ (both models now have a finished AdamW base). Queue 24/24.

**04:05 (10-27) — flat AdamW stable through it750; fine-tune seed 43 bases in.** Flat AdamW SVG s0 (`game24cdA_flat_svg2d_s0`, base 59.9): it250 **61.8**, it500 **61.6**, it750 **61.8** (val 0.598 → 0.603 → 0.603); at it935 buffer success 0.84, objective 120, gradient norm ≈1000, all finite. No collapse so far on the AdamW-trained flat model, versus the Muon flat model's −11.7 — consistent with the base-geometry prediction. Hier AdamW (base 68.2 / test 68.8 under seed-777 eval): SVG s0 13466274 finite through it249 (buffer success 0.875, objective 434); GRR 13466275 at it100 **69.6** (+0.8). Fine-tune seed 43 (val-selected checkpoint_2800 for both; selection val acc flat 0.815, hier 0.785 on 600 prompts): flat ft43 base test **68.9** (val 0.667), SVG s0 13482978 running (it40: buffer success 0.91, critic loss 0.11); hier ft43 critic best epoch 3 (field score 0.846), SVG s0 13482981 pending on priority. Note the flat seed-43 base (68.9) is 4.8 above the seed-42 flat base (64.1): the base gap between seeds within an arm is as large as the flat–hier base gap, which is exactly why this replication is needed. Flat early-checkpoint control (base 50.6, critic best epoch 4, field score 0.764): SVG 13482924 running (it56, buffer success 0.63). Norm-penalty runs: λ=10 at it139 (tok_norm 18.07, penalty 3.2, buffer success 0.84), λ=100 at it94 (tok_norm 17.97, penalty 20.5, buffer success 0.72); both finite; the untouched token norm at the base is ≈17.7–17.8 in every run. 60-epoch flat AdamW Game24 fine-tune at epoch 38/40. Pretraining: helfK 38.4k/58.6k, ELF-M flat 26.6k, ELF-M hier 26.7k (checkpoints 29,280 and 19,530 saved; the 24 h limit will cut them before 58.6k → resubmit with the same grad-accumulation override). Queue 24/24.

**05:15 (10-27) — flat collapses on the second base; norm penalty does not help; hier AdamW gains; flat AdamW still flat.** Game24-cd test accuracy (base → it250 → it500 …):
- **Fine-tune seed 43, flat** (`game24cd_flat_ft43_svg2d_s0`, base **68.9**): 68.9 → 60.5 → **60.2** at it500 (−8.7). The flat collapse replicates on an independently fine-tuned base, and from a *higher* base than the hier seed-43 model. **Hier seed 43** (`game24cd_hier_ft43_svg2d_s0`, base **64.7**, val 0.633): 64.7 → 63.1 at it250 (−1.6; the seed-42 hier runs were +1.4/+0.9 at it250). Token norm at it~250: flat 17.97 → 19.9, hier 17.94 → 25.3 (hier's latent norm grows more, and it is the one not collapsing — see the norm-penalty result below).
- **Latent-norm penalty (competing simple fix) fails.** `np10` (λ=10): 63.9 → 46.1 → **34.6** at it500; `np100` (λ=100): 63.9 → 47.5 → **37.1**. These match the unpenalised flat seeds at the same points (s1 45.8 → 35.9, s2 48.3 → 36.2; s0 53.3 → 45.9). The penalty keeps the generated-position token norm at ≈18.0–18.1 (penalty term 0.9–3 at λ=10, 2.7–20 at λ=100), i.e. it does its job, and the collapse is unchanged: critic exploitation on the flat model is not a latent-norm blow-up. The flat Muon runs' token norm also stays ≈18–20 while collapsing. So "norm growth" is neither the cause nor a usable fix; this is the negative result for the obvious competitor.
- **Flat early checkpoint** (`game24cd_flat_early_svg2d_s0`, Muon flat checkpoint_700, base **50.6**): 50.6 → **42.0** at it250 (−8.6; buffer success 0.47 at it392). An immature flat checkpoint collapses like the mature one, so the base-matched Γ_random model's smaller loss (−4.2/−7.2/−10.2 from 58.5, vs −11.7 flat mean) is not explained by checkpoint immaturity — the random code gives partial protection in its own right.
- **AdamW pair.** Flat AdamW (`game24cdA_flat_svg2d_s0`, base 59.9): 61.8 / 61.6 / 61.8 / **61.9** / **61.7** at it250…1250; buffer success 0.81, objective 316 at it1382, finite; final at it1500 ≈05:50. Hier AdamW SVG (`game24cdA_hier_svg2d_s0`, base 68.8 under eval seed): 68.8 → **69.7** → **69.5** at it500 (+0.7), buffer success 0.78, critic AUC 0.997, token norm 17.7 → 18.0. Hier AdamW GRR (lr 3e-7): 69.6 at it100, **68.5** at it400 (−0.3). 60-epoch flat AdamW Game24 fine-tune done; val-selected checkpoint_2800 (select acc 0.787 on 600 val prompts vs 0.785 for hier AdamW seed 42, i.e. matched); critic pretraining 13483526 → SVG 13483548 (tag `game24cdA_flat60`).
- Pretraining: helfK ≈41k/58.6k; ELF-M ≈28k each (24 h limit at ≈17:15 today → resubmit).
- Submitted fine-tune seed 44 for both models: flat sft 13485342 → gate0a 13485343 (`game24cd_flat_ft44`), hier sft 13485344 → gate0a 13485345 (`game24cd_hier_ft44`); their SVG runs go in when slots free. Queue 24/24.
- Note on reading `/scratch1`: `outputs/rl/game24cdA_hier_svg2d_s0/log.jsonl` returned an I/O error on the login node while the job held it (evals read from the Slurm stdout instead); the file itself is fine (298 kB, updated 05:05).

**06:00 (10-27) — flat AdamW final +3.1 (no collapse); hier AdamW +1.8 at it1000; GRR on hier AdamW neutral.** Game24-cd test accuracy:
- **Flat AdamW SVG s0 final** (`game24cdA_flat_svg2d_s0`, 40-epoch AdamW flat base 59.9): 61.8 / 61.6 / 61.8 / 61.9 / 61.7 / **63.0** at it250…1500 → **+3.1**, no collapse at any evaluation, buffer success 0.75–0.88 throughout. This is the first flat ELF that does not collapse under SVG-2D, and it is the AdamW-recipe model whose base geometry (weight norm 2.5×, trunk activations 15×, gradient norm 10× the Muon model's) was predicted at 03:00 to differ. Archived (`runs/rl/game24cdA_flat_svg2d_s0`).
- **Hier AdamW SVG s0** (`game24cdA_hier_svg2d_s0`, base 68.8): 69.7 / 69.5 / 70.2 / **70.6** at it1000 (+1.8), buffer success 0.94, token norm 18.2. **Hier AdamW GRR final** (600 iters, lr 3e-7): 69.6 / … / 68.5 / 69.4 / **68.5** → −0.3 (neutral, as GRR was on the Muon models' Countdown). Archived.
- Fine-tune seed 43: flat 68.9 → 60.5 → 60.2 → **56.2** at it750 (−12.7); hier 64.7 → 63.1 → 63.4 → **62.6** at it750 (−2.1). The flat–hier separation holds on the second base (10.6 points at it750), although the hier seed-43 model is below its base so far, and its generated-token norm is growing fast (17.9 → 40.8 by it~800; flat seed 43: 18.0 → 23.3; hier AdamW stays at 18.2). Watch for a late hier drift on this seed.
- Norm-penalty controls: λ=10: 46.1 / 34.6 / 35.2 / **42.0** at it1000 (the same dip-and-partial-recovery as the unpenalised flat seeds: s1 35.9 → 50.2, s2 36.2 → 52.1); λ=100: 47.5 / 37.1 / **36.3** at it750. Norm held at 18.04–18.06 in both.
- Flat early checkpoint (base 50.6): 42.0 → 50.3 → **58.0** at it750 (+7.4 over its base after the initial dip; buffer success 0.59, token norm 19.5). The immature checkpoint dips like every flat model and then climbs well past its base — the one flat Muon run so far that ends above where it started, and the one with the most headroom (50.6). To be interpreted together with the final.
- Submitted: fine-tune seed 44 SVG runs flat 13485880 / hier 13485881 (after their gate0a 13485343/13485345, SFTs pending on priority), and the heavier-refit flat control **`game24cd_flat_svg2d_refit8_s0`** 13485886 (Muon flat base and 8k critic as the headline runs, `--mc 8 --mc_extra 16` i.e. 4× the critic refit steps per iteration: 8 always, +16 when AUC<0.9). Flat60 AdamW SVG 13483548 pending on priority. Queue 24/24.
- `runs/` refreshed (62 RL runs, 148 Slurm logs).

**07:00 (10-27) — hier AdamW final +1.9; the hier seed-43 model is drifting.** Game24-cd test accuracy:
- **Hier AdamW SVG s0 final** (`game24cdA_hier_svg2d_s0`, base 68.8): 69.7 / 69.5 / 70.2 / 70.6 / 70.3 / **70.7** → **+1.9**, never below base. AdamW pair, seed 0: flat +3.1 (59.9 → 63.0), hier +1.9 (68.8 → 70.7). Neither collapses; on this training recipe the hierarchy is not needed for stability. Archived. Hier AdamW SVG s1 submitted (13486561); s2 and flat s1/s2 when slots free.
- **Fine-tune seed 43, hier** (`game24cd_hier_ft43_svg2d_s0`, base 64.7): 63.1 / 63.4 / 62.6 / **57.2** / **59.3** at it250…1250 (−5.4 at it1250, −7.5 at it1000). This hier model *is* losing accuracy, unlike all nine seed-42 Γ_task hier runs (min −0.9); its generated-token norm is 44.5 at it~1300 (base 17.9; the hier AdamW run stayed at 18.2; the flat seed-43 run is at 25.4). **Flat seed 43**: 60.5 / 60.2 / 56.2 / 55.1 / **53.8** (−15.1). Separation at it1250: 5.5 points (flat −15.1 vs hier −5.4), smaller than seed 42's (flat −11.7 mean vs hier +2.4 mean). Honest reading: the direction replicates on an independent base, the magnitude does not; the hier protection is partial on this seed. The seed-42 hier runs never logged token norm (added 10-27), so whether their norm also grew is unknown — a trace (`consistency.py --mode trace` reports token norm) over the seed-43 hier checkpoints vs seed-42 would answer it; queued as analysis when a slot frees.
- Norm penalty: λ=10: 42.0 / **46.8** at it1250; λ=100: 41.6 / **44.0**. Both tracking the unpenalised flat recovery curve (s1/s2 at it1250: 50.3/50.2).
- Flat early checkpoint (base 50.6): 58.0 / 57.6 / **57.8** at it1250 (+7.2).
- Two late Γ_random base-matched finals confirmed from stdout: s1 52.5, s2 48.3 (as logged at 02:10).
- Queue 24/24: flat60 AdamW SVG, refit8 flat control and ft44 chains (flat seed-44 SFT running) pending on priority; helfK ≈46k/58.6k; ELF-M ≈31k.

**12:05 (10-27) — six finals; GPU queue starved since ~08:00.** (The hourly loop did not fire between 07:05 and 11:50; nothing was lost — all jobs ran or waited on Slurm.) Game24-cd test accuracy, base → it250 … it1500:
- **Fine-tune seed 43 pair (final).** Flat `game24cd_flat_ft43_svg2d_s0`: 68.9 → 60.5 / 60.2 / 56.2 / 55.1 / 53.8 / **55.1** (**−13.8**). Hier `game24cd_hier_ft43_svg2d_s0`: 64.7 → 63.1 / 63.4 / 62.6 / 57.2 / 59.3 / **56.0** (**−8.7**). On this independently fine-tuned pair the flat model collapses as before and the hier model collapses too, later and less (held within −2 for 750 iterations, then fell). Separation at the final: 5.1 points, vs the seed-42 pair's 14.1 (−11.7 vs +2.4 means). The hier seed-43 run's generated-token norm grew from 17.9 to ≈45 by it1300 (flat seed 43: 18 → 25; hier AdamW: 18 → 18.2). Trace of code–token consistency and token norm along this run and along hier seed-42 s1, plus the flat seed-43 run, submitted (`scripts/trace_ft43.slurm`, job 13497652, debug partition) → `outputs/analysis/cons_trace_ft43_{hier,flat}.json`.
- **Norm-penalty controls (final).** λ=10: 63.9 → 46.1 / 34.6 / 35.2 / 42.0 / 46.8 / **49.6** (−14.3); λ=100: 63.9 → 47.5 / 37.1 / 36.3 / 41.6 / 44.0 / **46.2** (−17.7). Token norm pinned at 18.0–18.1 throughout. Same dip-and-partial-recovery shape as the unpenalised flat seeds (finals 55.0 / 50.2 / 52.1); the penalty does not help and λ=100 is slightly worse. Verdict: latent-norm growth is not the failure mechanism on the flat model and a norm constraint is not a fix.
- **Flat early checkpoint (final).** Muon flat checkpoint_700, base 50.5: 42.0 / 50.3 / 58.0 / 57.6 / 57.8 / **57.4** (**+6.9**). The immature model dips 8.5 points at it250 exactly as the mature flat models do, then climbs to +6.9, the best gain of any flat Muon run. Reading: the collapse mechanism is present at every checkpoint maturity; what differs is how much the policy can still gain afterwards (the early checkpoint has the most headroom: CE 0.4 vs 0.016). So the base-matched Γ_random result (−7.2 mean from 58.5) is not explained by immaturity; the Γ_random code protects partially.
- **60-epoch flat AdamW (matched pair, final).** `game24cdA_flat60_svg2d_s0`, base **63.4** (val-selected checkpoint_2800, selection val acc 0.787 vs hier AdamW's 0.785): 65.3 / 65.5 / 66.1 / 66.3 / 66.8 / **66.6** (**+3.2**), monotone, no dip at any evaluation. AdamW-recipe summary (seed 0): flat40 +3.1 (59.9 → 63.0), flat60 +3.2 (63.4 → 66.6), hier +1.9 (68.8 → 70.7), hier GRR −0.3. **Under the AdamW recipe the flat model does not collapse and gains as much as the hierarchical one.** The hierarchy's protective effect is specific to the Muon-trained models — exactly the "base geometry predicts collapse" framing adopted at 03:00; the paper's claim must be stated that way, and the AdamW pair is now the main evidence for the predictor. AdamW seeds 1/2 submitted for both models (flat 13497649/13497650, hier s1 13486561, s2 13497651).
- Fine-tune seed 44 bases selected: flat checkpoint_2800 (selection 0.792), hier checkpoint_2100 (selection 0.773); SVG runs 13485880/13485881 pending. helfK Countdown pretraining finished (58.5k steps); its Game24 fine-tune 13474342 pending. ELF-M flat/hier at ≈46k/58.6k at 18.5 h — they will hit the 24 h limit at ≈17:15 before finishing; resubmit from the last checkpoint then.
- Cluster: since ≈08:00 every GPU job of ours is pending on priority (only the two ELF-M pretraining jobs, started yesterday, hold GPUs); 17 jobs queued, 2 running. Nothing to do about it; the class allocation is being used by others.
- `runs/` refreshed (63 RL runs, 153 Slurm logs).

**12:30 (10-27) — salvage plan after the seed-43 and AdamW results (discussed with the user).** (1) Robustify the fix: the seed-42 ablation that lets the actor move only the token channels (`--actor_channels tokens`, `game24cd_hier_svg2d_tokgrad_s0`, +3.4) is now run on the two new hier bases where plain hier fell or is untested: `game24cd_hier_ft43_svg2d_tokgrad_s0` (13497720, base 64.7, plain hier −8.7) and `game24cd_hier_ft44_svg2d_tokgrad_s0` (13497721, base checkpoint_2100, plain hier 13485881 pending). If frozen coarse channels never collapse on any Muon base, that is the method claim; the trace (13497652) tells whether plain hier seed 43 lost code–token consistency before it fell. (2) Predictor as headline: ~12 base models with outcomes from −18 to +7 → test whether one base-level statistic (critic sensitivity to a unit latent perturbation, trunk activation scale, weight spectrum) ranks collapse depth; fallback is the empirical early-warning rule (every collapsing run dips at it250, no stable run does). (3) Optimizer dependence as a result: Muon-trained models are exploitable, AdamW-trained are not; needs one geometry-closing Muon control (weight decay / spectral regularization) to show geometry rather than the optimizer label is what matters. Dropped: more plain-hier seeds for their own sake. Queue 19/24, GPUs still starved.

**13:00 (10-27) — the drifting hier seed-43 model lost code–token consistency before it lost accuracy.** Trace (`scripts/trace_ft43.slurm`, 13497652; `runs/analysis/cons_trace_ft43_{hier,flat}.json`; 256 val prompts, step k=40 of 64; the seed-42 hier s0/clgrad/tokgrad rows recovered from the archived Slurm log into `runs/analysis/cons_trace_hier_s0_from_log.json`):

| run (Game24-cd) | acc Δ at final | cons it0 / 500 / 750 / 1000 / 1250 / 1500 | token norm it0 → 1500 |
|---|---|---|---|
| hier seed 42, s0 | +5.2 | .996 / .994 / .995 / .994 / .989 / .986 | 21.5 → 25.5 |
| hier seed 42, s1 | +4.5 | .991 / .995 / .990 / .988 / .991 / .983 | 21.5 → 26.7 |
| hier seed 42, actor on token channels only (tokgrad) | +3.4 | .997 / .993 / .990 / .987 / .988 / .989 | 21.4 → 26.3 |
| hier seed 42, actor on coarse channels only (clgrad) | −1.7 | .995 / .991 / **.737** / **.565** / .579 / .576 | 21.4 → 36.2 |
| **hier seed 43, s0** | **−8.7** | .998 / .994 / **.959** / **.854** / **.766** / **.703** | 21.5 → 36.7 |
| flat seed 43, s0 | −13.8 | (no code) | 21.4 → 30.2 |
| flat seed 42, s0 | −9.2 | (no code) | 21.4 → 27.1 |

Reading. Every hierarchical run that stayed above its base kept the coarse channels consistent with the decoded tokens (≥ 0.98 at every checkpoint). The two hierarchical runs that lost accuracy are the two whose consistency broke: the coarse-only-actor ablation and the seed-43 model. On seed 43 consistency fell first (0.994 → 0.959 between it500 and it750) and accuracy fell one checkpoint later (62.6 → 57.2 between it750 and it1000); the token norm grew along with the inconsistency (25.5 → 32.8 over the same interval). Accuracy at it750 was still −2.1, so the consistency drop is a one-checkpoint leading indicator on this run. This is the first case where the "rail" story makes a prediction before the fact rather than after. It also says what should fix it: keep the actor off the coarse channels (the tokgrad ablation kept consistency at 0.99 and gained +3.4 on seed 42) — the frozen-coarse-channel runs on the seed-43 and seed-44 bases (13497720/13497721, queued) are the direct test. If they hold, the method claim becomes "H-ELF with the actor restricted to the token channels," and consistency becomes a monitorable invariant. Caveat: n=1 collapsed hier run plus the clgrad ablation; the seed-44 plain-hier run (13485881) gives a second look at whether plain hier breaks consistency on a new base.
Queue 19/24, all RL pending on priority; ELF-M pretrains at 19.5 h.

**13:10 (10-27) — the two frozen-coarse-channel runs moved to the debug partition.** Slurm's backfill estimate put the earliest start of our GPU-partition jobs at Wed 10-28 ≈19:40 (five other users hold ≈100 jobs). `train_svg.py` saves a resumable state every 50 iterations, so a new script `scripts/gate0b_debug.slurm` (debug partition, 1 h, A40, `--resume`, exits early once the final evaluation is in the log) runs a 1500-iteration job as a chain of ~4 one-hour jobs with `-d afterany`. The debug QoS allows 5 submitted jobs per user, so chains are extended hourly. Cancelled the GPU-partition copies (13497720/13497721) to avoid an un-resumed job later overwriting the same output directory. Debug chains: `game24cd_hier_ft43_svg2d_tokgrad_s0` 13498783 → 13498784, `game24cd_hier_ft44_svg2d_tokgrad_s0` 13498785 → 13498786; both started immediately on b11-09 at 13:05. Expected finals ≈18:00 if the node stays free. Resumed runs are reproducible from the same seed up to the resume boundaries (RNG states are saved and restored).

**13:50 (10-27) — frozen-coarse-channel runs at it≈380 on the debug node (≈470 iterations/hour).** `game24cd_hier_ft43_svg2d_tokgrad_s0` (base 64.7): it250 **64.2** (−0.5; plain hier seed 43 was 63.1 here), buffer success 0.84, token norm 22.5. `game24cd_hier_ft44_svg2d_tokgrad_s0` (hier seed-44 base, checkpoint_2100: **63.1** test, val 0.631): it250 **61.7** (−1.4), buffer success 0.88, token norm 21.5. Debug submissions at the cap (5), so chain links 3–4 go in as links 1 finish (≈14:05).

**14:15 (10-27) — debug chains resumed cleanly; it500.** Link 1 of each chain ended on the time limit at it≈430 and link 2 resumed from the last saved state. `game24cd_hier_ft43_svg2d_tokgrad_s0` (base 64.7): 64.2 / **64.2** at it250/500 (−0.5; plain hier seed 43 was 63.1 / 63.4 and later fell to 56.0). `game24cd_hier_ft44_svg2d_tokgrad_s0` (base 63.1): 61.7 / **65.9** at it250/500 (**+2.8**). Link 3 submitted for both (13500921, 13500922).

**15:00 (10-27) — frozen coarse channels do not save seed 43 (it750).** `game24cd_hier_ft43_svg2d_tokgrad_s0` (base 64.7): 64.2 / 64.2 / **58.7** at it250/500/750 (−6.0; plain hier seed 43 was 62.6 at it750 and 56.0 at the end). `game24cd_hier_ft44_svg2d_tokgrad_s0` (base 63.1): 61.7 / 65.9 / **63.9** (+0.8). So restricting the actor to the token channels does not prevent the seed-43 model from drifting; the consistency loss seen on the plain run was a symptom of collapse on this base, not the lever that stops it. Both chains continue (links 3 running, links 4 submitted 13502848/13502849, finals ≈17:00). Debug node still ours.

**15:30 (10-27) — geometry control queued; next debug experiment chosen.** Discussed with the user whether the collapse is "Muon's fault" and whether a general fix exists. What is defensible: the base geometry decides exploitability, and here Muon produced the exploitable geometry (one model pair per recipe, replicated over RL seeds). Learning-rate evidence already in hand: flat Muon at a 3× smaller actor step still collapses (49.5 final, −14.4, just later); hier Muon at a 3× larger step collapses (−9.1). So step size alone is not the lever. Two controls:
- **AdamW + weight decay 0.1** (same recipe as the AdamW flat model otherwise; both AdamW configs had weight_decay 0): shrinks the AdamW model's weight/activation scale toward the Muon model's. If it then collapses, scale (not the optimizer label) drives exploitability. Chain on the GPU partition: `sft-cd-elfAwd` 13503070 (outputs/elfAwd_b-countdown) → `sft-g24cd-elfAwd` 13503071 (outputs/elfAwd_b-game24-cd) → gate0a 13503143 (`game24cdAwd_flat`, submitted on the gpu partition because gate0a's default debug partition is at the 5-job cap) → gate0b 13503144. Earliest start Wed night; results ≈Fri–Sat. (Muon + weight decay would move the Muon model's norms the wrong way, and the Muon path hard-codes weight_decay 0, so it was not used.)
- **AdamW flat at a 10× actor step** (`--lr_actor 1e-5 --lr_actor_min 1e-6`, tag `game24cdA_flat_svg2d_lr1e-5_s0`): if the AdamW model collapses at 10× the step, the recipe difference is a quantitative safety margin and a sensitivity-normalised step is a general fix; if it stays stable, the difference is qualitative and the critic-robustness line (train the critic on displaced states) is the right one. Goes on the debug node as a chain as soon as the tokgrad links free slots (≈16:00).
Queue 24/24.

**15:20 (10-27) — sharpness-aware actor step implemented; related-work scan.** Literature scan saved in `notes/related_work.md`: the failure is a known class in image diffusion (reward-gradient hacking of DRaFT/AlignProp); the closest mechanism+fix paper is Reward Sharpness-Aware Fine-Tuning (Kim & Sim, CVPR 2026): hacking tracks the sharpness of the reward w.r.t. its input, and evaluating the reward at the one-step worst-case displaced sample removes it. Muon literature says Muon-trained nets are *more* robust to input corruption and that Muon's RL effects come from its ≈3.5× larger update scale, which points our Muon/AdamW difference at effective step size rather than fragile features. No prior work on learned per-step critics for flow LMs or on base-dependent exploitability.
Implemented `train_svg.py --rsa_rho ρ [--rsa_apply 0/1]`: the critic input (z~_k, x_{k-1}) is displaced by δ = −ρ‖state‖∇/‖∇‖ (per sequence, generated positions, detached) and the actor maximises the displaced logit; every run with ρ>0 logs `rsa_gap` = l(state) − l(state+δ), i.e. the critic sharpness S1 of the RSA paper. CPU smoke test on the login node (K=8, 2 iterations) ran end to end; at ρ=0.05 the gap was ≈880 on a logit scale of ≈1, so the critic is extremely sharp and ρ must be ≪0.05 (RSA-FT's ρ=0.01 in pixel units is ≈2e-5 relative). `scripts/sharpness.slurm <rho> <tag:config>...` measures the base-critic sharpness (6 actor iterations at lr 0, `--rsa_apply 0`) for a list of bases: the pre-RL predictor analysis. Plan on the debug node after the tokgrad chains: (1) AdamW flat 10× step, (2) flat Muon + RSA at ρ=1e-3 (`game24cd_flat_svg2d_rsa1e-3_s0`), (3) sharpness measurement over all Game24 bases.
Frozen-coarse-channel runs at it1000: ft43 **60.6** (−4.1, up from 58.7 at it750), ft44 **63.0** (−0.1).

**16:50 (10-27) — frozen-coarse-channel finals: partial protection only.** `game24cd_hier_ft43_svg2d_tokgrad_s0` (base 64.7): 64.2 / 64.2 / 58.7 / 60.6 / 60.1 / **61.2** → **−3.5** (plain hier seed 43: −8.7; flat seed 43: −13.8). `game24cd_hier_ft44_svg2d_tokgrad_s0` (base 63.1): 61.7 / 65.9 / 63.9 / 63.0 / 62.9 / **60.6** → **−2.5** (plain hier seed 44 13485881 still pending on the GPU partition). On seed 42 the same variant gave +3.4 (64.9 → 68.3). So restricting the actor to the token channels halves the seed-43 loss but does not remove it, and on seed 44 it also ends below base. Hierarchy tally on Muon bases, SVG-2D (Δ at it1500): seed 42 plain +5.2/+4.5/+… (9 runs mean +2.4), tokgrad +3.4; seed 43 plain −8.7, tokgrad −3.5; seed 44 tokgrad −2.5, plain pending. Flat Muon: −9.2/−14.5/−13.0 (seed 42), −13.8 (seed 43), −14.3/−17.7 (norm-penalty), seed 44 pending. The hierarchy's protection is real in every pairing (hier always ≥ 5 points above flat on the same base) but it is not a guarantee of non-negative Δ. Archived (66 runs).
Debug node now: AdamW flat 10×-step chain (13505262 → 13505263; at it240 objective 138, buffer success 0.88, token norm 20.6, i.e. still healthy early) and the sharpness-aware flat Muon chain `game24cd_flat_svg2d_rsa1e-3_s0` (13506452 → 13506453, ρ = 1e-3). Sharpness measurement job goes in when a total slot frees (24/24). ELF-M pretrains end on their 24 h limit at ≈17:15 → resubmit.

**17:20 (10-27) — AdamW at 10× actor step: early signs of the Muon-style collapse; RSA gap scale.** `game24cdA_flat_svg2d_lr1e-5_s0` (AdamW flat, base 59.9, `--lr_actor 1e-5`): it250 **62.3** (+2.4), but by it490 buffer success has fallen to 0.59 (0.75–0.88 in the 1e-6 run) and the generated-token norm has grown from 17.7 to **35.4** (the 1e-6 run stayed at 18–20; the collapsing Muon flat runs went to 25–30). This is the Muon flat signature appearing on the AdamW model once the step is 10× larger — consistent with the recipe difference being a step-size margin, pending the it500+ evaluations. Link 3 submitted (13507099). Sharpness-aware flat Muon run `game24cd_flat_svg2d_rsa1e-3_s0` (ρ = 1e-3): rsa_gap 28 at it0 against objective 796 (3.6%), 29 / 12 / 6.3 at it10/20/30 as the objective fell to 52; at it220 gap 4.0 vs objective 37, buffer success 0.66. So at ρ = 1e-3 the flattened objective differs from the raw one by 4–10%: a meaningful but not dominating correction. First evaluation at it250 next. ELF-M pretrains at 23:57 of 24 h.

**17:35 (10-27) — the AdamW model collapses at a 10× actor step; the sharpness-aware step at ρ=1e-3 does not prevent the dip.**
- **AdamW flat, `--lr_actor 1e-5`** (`game24cdA_flat_svg2d_lr1e-5_s0`, base 59.9): 62.3 at it250 → **46.6 at it500 (−13.3)**, buffer success 0.50, token norm 32–35. At the paper's step (1e-6) the same model went 59.9 → 63.0 with no dip. So the AdamW model is not immune: it collapses the Muon way once the actor step is 10× larger. Together with the Muon flat run at a 3× *smaller* step (still −14.4, later), the picture is a **step-size margin**: every model has a critic-exploitation regime, and the AdamW recipe (weights 2.5×, activations 15× larger) sits 10× further from it at a fixed parameter-space step. This is the "scale convention" reading the Muon-RL literature gives for Muon's effects, now shown for critic exploitation. The general fix this implies is a step normalised by the model's output sensitivity (trust region in the clean estimate), not an optimizer choice. Chain continues (links 3–4) to see whether it recovers like the Muon runs.
- **Flat Muon + sharpness-aware actor step, ρ = 1e-3** (`game24cd_flat_svg2d_rsa1e-3_s0`, base 63.9): **52.2 at it250 (−11.7)**, buffer success 0.53 at it330, rsa_gap 1.9 (objective ≈30). Plain flat s0 was 53.3 at it250. At this ρ the flattened objective is within 5–10% of the raw one and the actor exploits the critic just the same. Next: ρ = 1e-2 (gap expected to be of the order of the objective) as soon as a slot frees; if that also fails, the critic's sharpness is not what the actor exploits, and the step-size/trust-region line is the one to pursue.
- ELF-M pretrains are at step 59,900 of 58,600 (finishing their last epoch/save within the 24 h limit after all); the Game24 fine-tunes 13472911/13472913 follow automatically. Total queue 24/24, debug 5/5.

**17:40 (10-27) — the objection: is a smaller actor step the fix?** Raised by the user. Evidence in hand: flat Muon at a 3× smaller step (`lr3e-7`) still collapses (−14.4, later); AdamW at a 10× larger step collapses (−13.3 at it500); hier Muon at 3× smaller gains +2.5, at the standard step +5.2, at 3× larger collapses. Untested: flat Muon at a 10× smaller step (the mirror of the AdamW experiment). If it stays stable and gains, step size alone is a remedy and the hierarchy becomes one way among several to widen the safe window; the fair comparison is then best gain per model at the tuned step. Submitted as top priority: `game24cd_flat_svg2d_lr1e-7_s0` (`--lr_actor 1e-7 --lr_actor_min 1e-8`, debug chain 13508295 → links as slots free). To make room the second link of the ρ=1e-3 sharpness-aware chain (13506453) was cancelled; that run stops at ≈it430 with its it250 result (52.2) on record, which already answers the question at that ρ. If flat at 1e-7 holds, run hier at 1e-7 for the matched pair.

**17:55 (10-27) — AdamW 10× step: 21.9 at it750.** `game24cdA_flat_svg2d_lr1e-5_s0`: 59.9 → 62.3 / 46.6 / **21.9** (−38.0), buffer success 0.34. Deeper than any Muon flat run at the standard step (worst −17.7 at it1500), so at a large enough step the AdamW model is not just exploitable but more so. Flat Muon at 10× smaller step (`game24cd_flat_svg2d_lr1e-7_s0`) started 17:43, healthy at it40 (buffer success 0.84, token norm 17.7); link 2 13509064 queued. The ρ=1e-3 sharpness-aware run ended at its first link (it250 52.2 on record). ELF-M pretraining is at epoch 32/60 (step counter counts micro-batches) and keeps running.
Swapped the AdamW 10× chain's link 3 (13507099, cancelled; the run ends at ≈it860 with its −38.0 at it750 on record) for the base critic-sharpness measurement over all 12 Game24 bases (`scripts/sharpness.slurm 1e-3 …`, job 13509089, debug; outputs/analysis/sharp_<tag>_rho1e-3/). ELF-M pretraining time limit is 40 h (ends 09:17 Wed) at epoch 32/60 after 24.6 h → will be cut near epoch 51; resubmit then (auto-resume).

**18:35 (10-27) — base critic sharpness predicts RL outcome (Spearman +0.76, 10 bases); flat Muon at 10× smaller step holds at it250.**
Sharpness = mean rsa_gap over 6 actor iterations at lr 0 (`scripts/sharpness.slurm 1e-3`, job 13509089): critic-logit drop under the one-step worst-case displacement of the critic input with ‖δ‖ = 10⁻³‖state‖, measured on each base with its own pretrained critic. Logs in `runs/analysis/sharp/`.

| base | gap | gap/|obj| | final Δ (SVG-2D, it1500) |
|---|---|---|---|
| AdamW flat (40 ep) | 102.6 | .119 | +3.1 |
| AdamW flat (60 ep) | 78.0 | .070 | +3.2 |
| Muon flat, early ckpt | 34.6 | .067 | +6.9 |
| AdamW hier | 69.3 | .053 | +1.9 |
| Muon hier, ft seed 43 | 34.0 | .041 | −8.7 |
| Muon hier, seed 42 | 34.5 | .037 | +2.4 (mean of 9) |
| Muon flat, seed 42 | 22.6 | .035 | −11.7 (mean of 3) |
| Muon flat, ft seed 43 | 16.8 | .030 | −13.8 |
| Muon hier, ft seed 44 | 20.7 | .029 | pending |
| Muon flat, ft seed 44 | 27.2 | .029 | pending |
| Γ_random, base-matched | 13.1 | .022 | −7.2 (mean of 3) |
| Γ_random, mature | 16.9 | .021 | −2.1 (mean of 3) |

Spearman(gap, Δ) = **+0.76** over the 10 bases with outcomes (gap/|obj| +0.70; |obj| alone +0.31). Within the 7 Muon bases ≈ +0.61. **Sign is the opposite of the image-diffusion intuition:** bases whose critic is *more* sensitive to its input fare *better*. Tentative reading: a critic with a steep, consistent input gradient gives the actor a real direction to follow, while a flat critic (small gap) has a gradient dominated by directions it was never trained on, which the actor then exploits. This also explains why RSA flattening (ρ=1e-3) did not help and would predict that it hurts at larger ρ. Caveats: n=10 with the 4 AdamW bases at the top (recipe confound); one ρ; gap measured at the base, not along RL. Seed-44 bases give two out-of-sample predictions: flat ft44 (27.2) and hier ft44 (20.7) both in the Muon low-gap band → predicted to lose; hier ft44 at the lowest Muon-hier gap → predicted worse than other hier bases (its tokgrad run did end −2.5).
Flat Muon at `--lr_actor 1e-7` (`game24cd_flat_svg2d_lr1e-7_s0`): it250 **64.5** (+0.5; plain flat 53.3, lr3e-7 65.2 at the same point), buffer success 0.84, token norm 17.9 at it350. Too early to call: lr3e-7 was also up at it250 and fell from it500.

**18:45 (10-27) — queue moves.** Seed-44 plain hier (`game24cd_hier_ft44_svg2d_s0`, the third deciding run) moved from the GPU partition (13485881, cancelled before it ever started) to a debug chain: link 1 13510858, runs alongside the small-step flat run on the second A40. Small-step flat chain extended (link 3 13510857). Seed-44 plain flat 13485880 stays on the GPU partition until a debug slot frees. AdamW 10×-step run archived (ended at its second link, ≈it860; −38.0 at it750). Predictions registered at 18:35 from base sharpness: both seed-44 bases (gap 20.7 hier, 27.2 flat) fall in the low-gap Muon band → both predicted to end below base.

**19:00 (10-27) — flat Muon at 10× smaller step still holding at it500.** `game24cd_flat_svg2d_lr1e-7_s0`: 63.9 → 64.5 → **65.1** at it250/500 (+1.2). Plain flat was 45.9 and lr3e-7 61.4 at it500. Buffer success 0.75 at it580. Not yet decisive: lr3e-7 fell between it500 and it750.

**19:45 (10-27) — flat Muon at 10× smaller step holds through it750; seed-44 hier dips early.**
- `game24cd_flat_svg2d_lr1e-7_s0` (base 63.9): 64.5 / 65.1 / **63.6** at it250/500/750 (−0.3), buffer success 0.75, token norm 17.8 (unchanged from base). lr3e-7 was 55.1 and plain flat 59.7 (after 45.9 at it500) at it750. So far the 10× smaller step neither collapses nor learns. That is the "safe but stalled" outcome: the margin idea holds (a small enough step avoids exploitation), but it buys stability by giving up the gain the hier runs get at the standard step (+5.2/+4.5 seed 42). Decisive comparison = finals at it1500.
- `game24cd_hier_ft44_svg2d_s0` (seed-44 plain hier, base 63.1): **62.4 at it250** (−0.7; tokgrad on the same base was 61.7). Link 1 ended at ≈it430 with no successor queued (the debug cap had been full); resumed with links 2–3 (13511931 → 13511932) at 19:41, ≈20 min of node time lost.

**20:25 (10-27) — small-step flat stays flat; seed-44 hier rises then drops.**
- `game24cd_flat_svg2d_lr1e-7_s0` (base 63.9): 64.5 / 65.1 / 63.6 / 63.4 / **63.8** at it250…1250. Buffer success 0.88, token norm 17.7, unchanged from base. No collapse, no gain: "safe but stalled". Final ≈20:50.
- `game24cd_hier_ft44_svg2d_s0` (seed-44 plain hier, base 63.1): 62.4 / **67.5** / **60.7** at it250/500/750. Up 4.4 at it500, then down 2.4 at it750, with token norm rising to 31.2. That is the same drift signature as hier seed 43 (norm 17.9 → 45, final −8.7), so the registered prediction that it loses is on track.
- Debug chains: flat lr1e-7 link 4 (13513204) queued; hier ft44 link 4 goes in when the flat chain frees a slot.
- Git: the last three pushes had not reached GitHub, because quiet pushes hit the command timeout. A background push is running now.

**21:10 (10-27) — flat Muon at a 10× smaller step: final −1.0. Safe, but no gain.** `game24cd_flat_svg2d_lr1e-7_s0` (base 63.9): 64.5 / 65.1 / 63.6 / 63.4 / 63.8 / **62.9** at it250…1500. Token norm stayed at 17.7–17.9 throughout, and there was no dip. Flat Muon step-size sweep, final Δ:

| actor lr | final Δ |
|---|---|
| 1e-6 (paper) | −9.2 / −14.5 / −13.0 (3 seeds) |
| 3e-7 | −14.4 (collapse delayed to after it500) |
| 1e-7 | **−1.0** (no collapse, no learning) |

Hier Muon on the same base, at the paper step: +5.2 / +4.5 / … (9 runs, mean +2.4); at 3e-7 +2.5.
**Verdict on the user's objection:** a small enough step does prevent the collapse, but no flat step size tried produces a gain. The step that is safe for flat is too small to learn anything in 1500 iterations. The step that learns collapses. Hier at the paper step gains. So on the seed-42 base the hierarchy is not replaceable by step tuning: it moves the cliff so that a step large enough to learn is still safe. Matched check queued: hier at 1e-7 (`game24cd_hier_svg2d_lr1e-7_s0`, chain 13514141 → 13514303). If hier also stalls there, the comparison is cleanly "same step: both safe, neither learns; paper step: hier learns, flat collapses". Caveats: one seed per lr below 1e-6; an intermediate lr (2e-7) might give flat a small gain.
**Seed-44 plain hier** (`game24cd_hier_ft44_svg2d_s0`, base 63.1): 62.4 / 67.5 / 60.7 / **65.3** at it1000 (+2.2). It recovered after the it750 dip. Token norm is 32.5 and still rising. Final ≈22:30.

**21:50 (10-27) — why Muon and not AdamW: the safe window is closed, not shifted; layer-wise relative step queued.** Discussed with the user. From the step sweeps: flat AdamW learns at 1e-6 and collapses at 1e-5, so its safe-and-learning window is ≥10× wide. Flat Muon collapses at 1e-6 and 3e-7 and stalls at 1e-7 (−1.0), so no global step both learns and stays safe. A global scale difference (weights 2.2–2.5×) would only shift the window, so the Muon/AdamW difference is not just parameter scale. Untested remainder: how the step is distributed across layers. Implemented `--rel_step 1` (LARS-style: each matrix lr × rms(W)/mean rms, same mean lr; vectors unchanged). Base weight-RMS spread across the 61 matrices: Muon flat 0.106–5.91× the mean (median 0.88, mean rms 0.136); AdamW flat 0.205–3.06× (median 1.01, mean rms 0.306). Muon has a wider spread of layer scales, so under a uniform lr its smallest layers move up to ≈9× further relative to their size than the mean layer, versus ≈5× for AdamW. Run `game24cd_flat_svg2d_relstep_s0` (flat Muon, paper lr 1e-6) as a debug chain 13515080 → second link, queued ahead of the hier lr1e-7 continuation (whose link 2 was cancelled; that run pauses at ≈it430 and resumes from its state later). Interpretation: if relstep learns without collapsing, a one-line optimizer change is a general fix and becomes the headline; if it collapses, the cause is not where the step lands, and the critic-swap 2×2 (Muon policy with an AdamW-trunk critic and the reverse; both share the frozen T5 latent space) is next.

**22:00 (10-27) — direction set with the user: make the predictor the core and test it out of sample and across tasks.** Plan in order: (1) register a sensitivity prediction for every new base before its RL run (seed 44 already registered; k-means hier, ELF-M, AdamW+wd, AdamW seeds to follow); (2) measure sensitivity on the existing K&K and Countdown bases, which already have RL outcomes (flat collapsed on both), at no new RL cost; (3) mechanism: relstep now, critic swap if relstep fails; (4) seeds for the headline; writing from early January; image-diffusion generalisation only as a February stretch goal. Stopped: more plain-hier seeds, more failed-fix variants. `scripts/sharpness.slurm` now takes `tag:config:task_dir` and deletes its checkpoint copies. The K&K/Countdown measurement (kk_{flat,hier}, countdown_{flat,hier}, countdown_{flat,hier}_c16) is waiting for a debug slot (cap 5 reached).

**22:20 (10-27) — seed-44 plain hier final −0.8.** `game24cd_hier_ft44_svg2d_s0` (base 63.1): 62.4 / 67.5 / 60.7 / 65.3 / 64.4 / **62.3** (−0.8). The registered prediction from base sensitivity (gap 20.7, low Muon band) was "below base": correct in sign, small in size. Hierarchy tally, Δ at it1500 on three independently fine-tuned Muon bases: seed 42 +2.4 (mean of 9), seed 43 −8.7, seed 44 −0.8; frozen-coarse variant +3.4 / −3.5 / −2.5. Its paired flat run (`game24cd_flat_ft44_svg2d_s0`, base gap 27.2, prediction: below base) now on a debug chain (13515686; GPU-partition copy 13485880 cancelled before it started). Cross-task sensitivity job for the K&K and Countdown bases submitted (13515685). Relative-step run started 22:10 (multipliers 0.106–5.91), healthy at it20.

**22:45 (10-27) — cross-task check: the sensitivity predictor does NOT transfer.** Base critic sensitivity (mean rsa_gap, ρ = 1e-3, `scripts/sharpness.slurm`, job 13515685, logs in `runs/analysis/sharp/`) for the existing K&K and Countdown bases, against their RL outcomes (Δ test acc at the final iteration):

| task / base | gap | Δ |
|---|---|---|
| K&K flat | 20.4 | −7.3 / −7.6 / −11.1 (it1000) |
| K&K hier | 7.5 | −5.2 / +0.0 / +0.8 |
| Countdown flat, deployment-sampler critic | 45.3 | −7.1 |
| Countdown hier, deployment-sampler critic | 32.3 | −4.4 |
| Countdown flat, cheap-sampler critic (c16) | 40.5 | +2.6 |
| Countdown hier, cheap-sampler critic (c16) | 18.9 | +1.3 |

"Higher gap → better outcome" is right for the Game24 flat/hier pair, wrong for K&K (hier gap 3× lower, hier much better) and wrong for the Countdown deployment pair. On Countdown, the critic with the better outcome has the lower gap in both architectures (c16 vs deployment). Caveat: the c16 critics were trained on K=16 rollouts and are measured here at K=64. Re-reading the Game24 table, the predictor is cleaner within one architecture than across: within the Game24 flat bases Spearman ≈ +0.6 (n=5), within the Game24 hier/Γ bases ≈ +0.6 (n=6). Flat and hier critics take different inputs (the hier critic sees the coarse channels), so their gaps are probably not comparable. **Revised reading:** sensitivity at ρ = 1e-3 ranks bases of the same architecture on the same task (weakly, n = 5–6) but is not a general cross-architecture or cross-task predictor. The +0.76 headline from 18:35 overstated it, because it pooled architectures, and the AdamW bases sit at the top of both pools. Consequence: the predictor cannot be the paper's core claim as it stands. Options: (a) a within-architecture test with more bases (seed 44 flat pending; k-means, ELF-M, AdamW+wd later), (b) a normalised measure (e.g. gap relative to the critic's own logit spread across rollouts), recomputable from the existing logs plus one cheap job, (c) fall back to the margin + hierarchy framing.

**22:55 (10-27) — relative step: smaller dip, still a dip.** `game24cd_flat_svg2d_relstep_s0` (flat Muon, lr 1e-6, per-matrix lr × rms/mean): 63.9 → **59.8** at it250 (−4.1). Plain flat s0 was 53.3 (−10.6) at the same point; lr3e-7 was 65.2. Buffer success fell to 0.62 at it330, token norm 18.4. Equalising per-layer relative steps halves the early damage, but the collapse signature is still there. it500 (≈23:30) decides whether it recovers or deepens. Seed-44 flat (base 64.4 test, 63.8 val) is at it160. Its link 2 is queued (13516331). Relstep link 3 goes in next tick (debug cap).

**23:30 (10-27) — critic swap ready; seed-44 flat dips.** `game24cd_flat_ft44_svg2d_s0` (base 64.4): **61.3** at it250 (−3.1), buffer success 0.75 at it380. Relstep at it490: buffer success 0.72, token norm 21.0 (rising; plain flat was 25–30 while collapsed). Its it500 evaluation was still computing. Critic swap needs no code change: `pretrain_critic` builds the critic trunk from whatever base checkpoint it is given and saves the trained trunk, and `train_svg` loads that trunk into the critic. New `scripts/critic_swap.slurm <trunk_config> <trunk_tag> <policy_tag> <out_tag>` trains a critic on one policy's stored rollouts (`outputs/critic_data/<policy_tag>`, 64 shards each for game24cd_flat and game24cdA_flat) with the other model's trunk. The 2×2 off-diagonal cells: MA = Muon flat policy + AdamW-trunk critic (`game24cd_swapMA`), AM = AdamW flat policy + Muon-trunk critic (`game24cd_swapAM`). Diagonals exist: MM −9.2 (s0), AA +3.1. Submitting when debug slots free.

**23:45 (10-27) — layer-wise relative step does not prevent the collapse.** `game24cd_flat_svg2d_relstep_s0`: 63.9 → 59.8 → **43.1** at it250/500 (−20.8; plain flat 45.9 at it500), buffer success recovering to 0.84 at it690, token norm 22.1. Equalising each matrix's relative step changes the timing (smaller dip at it250) but not the outcome. Together with the global step sweep, how the step is sized or distributed across layers is not what separates the Muon and AdamW models. The run stops at its link 2 (≈it860; link 3 13516724 cancelled), with it500 as the answer. Critic swap started: critics MA (AdamW-trunk critic trained on the Muon flat policy's rollouts) 13516963 and AM (Muon trunk on the AdamW flat policy's rollouts) 13516964; `outputs/eval/game24cd_swap{MA,AM}_ckpt.txt` point to the respective policies. Seed-44 flat at it490: buffer success 0.66, token norm 20.8.

**00:40 (10-28) — swap critic MA trained.** The AdamW-trunk critic trained on the Muon flat policy's rollouts (`outputs/critic/game24cd_swapMA_2d`, job 13516963) reached a best held-out field score of **0.821** (epoch 6). For comparison, the Muon policy's native critic scored 0.838 and the AdamW policy's native critic 0.761. So the swapped critic is about as accurate as the native one on the same data, and any difference in RL outcome is not about critic quality. The AM critic (Muon trunk on AdamW rollouts) is training now (13516964). RL: MA link1 13517196 (waiting for a GPU on the debug node) → link2 13517362; AM link1 13517361 (after its critic). Seed-44 flat paused after link 2 (resumes from state later). Relative-step logs moved to resolved.

**01:10 (10-28) — swap critic AM trained; MA RL started.** AM critic (Muon trunk trained on the AdamW flat policy's rollouts): best field score **0.804** (epoch 3). The AdamW policy's native critic scored 0.761. Both swapped critics are as accurate as or better than the native ones, so critic quality cannot explain the swap outcomes. MA RL (`game24cd_swapMA_svg2d_s0`, Muon flat policy + AdamW-trunk critic) started 01:02. Its it0 eval is 63.9 (the Muon flat base), buffer success 0.88, and the first iteration is finite. AM RL link1 13517361 is waiting for a GPU (other users now queue on the debug node), with link2 13517578 queued behind it.

**01:50 (10-28) — first swap reading; seed-44 flat collapsing.**
- **MA** (Muon flat policy + AdamW-trunk critic, `game24cd_swapMA_svg2d_s0`): 63.9 → **59.7** at it250 (−4.2), buffer success 0.81, token norm 21.3 at it300. Native MM was 53.3 (−10.6) at it250. The AdamW-trunk critic halves the early dip, but token norm is already rising (MM signature). it500 decides. **AM** (AdamW flat policy + Muon-trunk critic): it190, buffer success 0.75, token norm 17.5 (flat). Its it250 eval comes ≈02:05.
- **Seed-44 flat** (`game24cd_flat_ft44_svg2d_s0`, base 64.4): 61.3 / **56.2** at it250/500 (−8.2). The registered prediction (below base) holds. The paired hier seed-44 run ended −0.8, so the hierarchy is +7.4 better on this base at the matched point. Chain paused after link 2; it resumes after the swap chains.

**02:15 (10-28) — AM swap holds at it250.** **AM** (AdamW flat policy + Muon-trunk critic): 59.9 → **60.1** at it250 (+0.2), token norm 17.8 (flat), buffer success 0.66 at it398. Native AA was 61.8 at it250. So far, the Muon-built critic does not make the AdamW policy collapse. MA (Muon policy + AdamW critic) was 59.7 at it250 with rising token norm. Its it500 is pending (log.jsonl unreadable while held; read from stdout next tick). Tentative reading, if it500 confirms: the fragile part is the Muon policy, not the critic. Chains: MA link2 13517362 → link3 13518039; AM link2 13517578 → link3 13518040.

**02:40 (10-28) — AM swap stable through it500.** AM (AdamW flat policy + Muon-trunk critic): 59.9 → 60.1 → **59.5** at it250/500 (−0.4), token norm 17.8. The Muon-built critic does not make the AdamW policy collapse. MA (Muon policy + AdamW critic) is at it460, buffer success 0.81, token norm 21.1. Its it500 is due in ≈10 min. Both chains have link 3 queued.

**02:55 (10-28) — CRITIC SWAP: the AdamW-trunk critic protects the Muon policy.** MA (Muon flat policy + critic built on the AdamW model's trunk, trained on the Muon policy's own rollouts): 63.9 → 59.7 → **66.0** at it250/500 (**+2.1**), buffer success 0.91, token norm 22.2 at it550. The native pairing MM (same policy, same rollouts, critic on the Muon model's own trunk) was 53.3 → **45.9** at the same points. AM (AdamW policy + Muon-trunk critic): 60.1 → 59.5 (−0.4), stable.

| policy \ critic trunk | Muon | AdamW |
|---|---|---|
| Muon | 45.9 at it500 (final −9.2) | **66.0 at it500 (+2.1)** |
| AdamW | 59.5 at it500 (−0.4) | 61.6 at it500 (final +3.1) |

Reading: the collapse needs both the Muon policy and a critic built on that same Muon model. In CFRL the critic's frozen trunk is a copy of the policy's own base, so the critic shares the policy's representation and its blind spots. The Muon policy can move into regions that its own trunk represents poorly and the critic, reading that same trunk, cannot detect. A critic on a different trunk does not share those blind spots, and the same policy then learns (+2.1 at it500 vs −18.0). The AdamW policy is not exploiting even the Muon critic, so the policy side matters as well: the effect needs a fragile policy paired with its own trunk. Caveats: one seed, it500 only; the AdamW and Muon trunks differ in many ways, so "different trunk" and "AdamW trunk" are confounded. **Decisive next test:** Muon policy + critic built on a *different Muon model* (the seed-43 flat base). Protected → shared-trunk blind spots are the mechanism (any independent trunk fixes it). Collapse → Muon trunks specifically are exploitable. Critic job queued as soon as a debug slot frees: `sbatch scripts/critic_swap.slurm configs/elf_b_game24_cd.yml game24cd_flat_ft43 game24cd_flat game24cd_swapMM43`.
Decisive test submitted: critic on the seed-43 Muon flat base's trunk, trained on the seed-42 Muon flat policy's rollouts (`swap-critic-MM43` 13518381 → outputs/critic/game24cd_swapMM43_2d; eval ckpt file points to the seed-42 Muon flat policy). Its RL run (`game24cd_swapMM43_svg2d_s0`) goes in as soon as a debug slot frees. To free a debug slot permanently, the k-means hier gate0a (debug partition, waiting all day on its SFT dependency) was moved to the GPU partition (13518377, 3 h limit) with its three RL seeds re-chained behind it (13518378–80). The old jobs 13474343/44 and 13476210/11 were cancelled before they ever started.

**03:45 (10-28) — decisive swap running.** MM43 critic (seed-43 Muon flat trunk, trained on the seed-42 Muon flat policy's rollouts): best field score **0.882**, the highest of any flat critic so far (native 0.838, AdamW-trunk 0.821). Its RL run `game24cd_swapMM43_svg2d_s0` started 03:31 (link 2 queued). The held MA/AM third links were released. it250 is due ≈04:05.

**04:25 (10-28) — decisive test: a critic on a *different Muon model* does NOT protect.** `game24cd_swapMM43_svg2d_s0` (seed-42 Muon flat policy + critic on the seed-43 Muon flat trunk, field 0.882): 63.9 → **56.5** at it250 (−7.4), buffer success **0.31** at it320. The collapse signature is there (MM 53.3, MA 59.7 at it250). MA (same policy + AdamW-trunk critic) continues to hold: 59.7 / 66.0 / **66.9** at it250/500/750 (+3.0), token norm 25.2.

| policy \ critic trunk | Muon (own) | Muon (seed 43) | AdamW |
|---|---|---|---|
| Muon | 53.3 → 45.9 | 56.5 at it250, collapsing | 59.7 → 66.0 → 66.9 |
| AdamW | — | — | (native) 61.8 → 61.6 |
| AdamW | 60.1 → 59.5 (Muon-own trunk) | | |

So "shared trunk" in the narrow sense (the critic is a copy of the policy's exact weights) is not the mechanism: an independently fine-tuned Muon trunk is exploited just as well. What protects the Muon policy is a critic on the AdamW-trained trunk. Muon critics are exploitable by the Muon policy, but not by the AdamW policy (AM stable). Caveat on independence: the seed-42 and seed-43 Muon models are both fine-tuned from the same Muon Countdown model (they share most of their weights), whereas the AdamW model went through its own Countdown SFT. "Muon-trained trunk" and "close relative of the policy" are still confounded. Revised hypothesis: a critic is exploitable when its representation is close to the policy's own, and the AdamW recipe produces a representation far enough away. Separating the two needs a critic trunk that is Muon-trained but not a relative (e.g. the k-means or random-code H-ELF Countdown models are separate Muon SFTs but a different architecture; ELF-M is a different size; a fresh Muon Countdown SFT from the OWT init with another seed would be the clean test, ≈10 h on the GPU partition). MM43 runs to it500 for the record (link 2 queued); MA link 4 queued (13518??) to reach it1500.
Clean independence test queued: a fresh Muon flat Countdown SFT from the OWT init with seed 7 (`sft-cd-elf-s7`, outputs/elf_b-countdown-s7; GPU partition, earliest start Wed night). Plan: its Game24 fine-tune (`sbatch -J sft-g24cd-elf-s7 -d afterok:<id> scripts/sft.slurm configs/elf_b_game24_cd.yml --config_override seed=7 --config_override init_from=outputs/elf_b-countdown-s7 --config_override output_dir=outputs/elf_b-game24-cd-s7`), then a critic on that trunk trained on the seed-42 Muon policy's rollouts (`critic_swap.slurm ... game24cd_swapMM7`), then RL. Protected → "relative of the policy" is what makes a critic exploitable (independent trunk = general fix). Collapse → Muon-trained trunks are exploitable by Muon policies.

**05:00 (10-28) — replication of the protective critic queued.** AM (AdamW policy + Muon-trunk critic): 60.1 / 59.5 / **59.4** at it250/500/750, stable. MM43 stopped after link 1 (it≈430; 56.5 at it250, buffer success 0.31; its it500 not reached). The collapse signature was already clear, and the slot goes to the replication: MA with RL seed 1 (`game24cd_swapMA_svg2d_s1`, same AdamW-trunk critic, chain 13519262 → 13519263). AM link 4 13519264 queued; MA seed 0 link 4 13519015 queued. Queue 24/24 total, 5/5 debug. The seed-7 Game24 SFT goes in when a total slot frees.

**05:40 (10-28) — waiting on the shared debug node.** AM (AdamW policy + Muon critic) at it1000: **59.6** (−0.3, stable). MA seed 0 stands at 66.9 at it750; its link 4 and the seed-1 replication are pending on priority behind other users' jobs. Seed-7 Game24 fine-tune queued behind the fresh Muon Countdown SFT.

**07:05 (10-28) — replication started.** `game24cd_swapMA_svg2d_s1` (Muon flat policy + AdamW-trunk critic, RL seed 1) started 06:50. At it60: buffer success 0.69, token norm 18.1. The native flat seed-1 run fell to 45.8 at it250 and 35.9 at it500. Its it250 reading is due ≈07:25. The held MA s0 link 4 and AM link 4 were released.

**07:35 (10-28) — replication: protective at it250 again.** `game24cd_swapMA_svg2d_s1` (Muon flat policy + AdamW-trunk critic, RL seed 1): 64.6 → **62.2** at it250 (−2.4). The native critic with the same seed went 64.7 → 45.8 (−18.9). Buffer success 0.72, token norm 20.1 at it310. Same shape as seed 0 (−4.2 at it250 then +2.1 at it500). Link 3 queued (13520096). The it500 check comes ≈08:10.

**08:15 (10-28) — MA seed 0 at it1000: +4.6.** `game24cd_swapMA_svg2d_s0` (Muon flat policy + AdamW-trunk critic): 59.7 / 66.0 / 66.9 / **68.5** at it250…1000 (+4.6). Buffer success 0.97 at it1110, token norm 28.0. The token norm grows, yet accuracy keeps rising, so norm growth alone is not the failure. The native Muon flat seed-0 run was at 58.6 at it1000. AM (AdamW policy + Muon critic) at it1240 is stable. Both finish ≈08:50. The seed-1 replication (62.2 at it250) is waiting for a GPU behind them, with links 2–4 queued (13519263 → 13520096 → 13520355).

**08:55 (10-28) — AM final: stable. MA seed 0 at it1250: 66.7.** **AM final** (`game24cd_swapAM_svg2d_s0`, AdamW flat policy + Muon-trunk critic): 59.8 → 60.1 / 59.5 / 59.4 / 59.6 / 58.8 / **59.5** (−0.3). A Muon-trunk critic does not make the AdamW policy collapse, though the run also learns nothing (the native AdamW pair gained +3.1). MA s0: 59.7 / 66.0 / 66.9 / 68.5 / **66.7** at it250…1250 (+2.8). Link 4 ended at ≈it1290, and link 5 (13520791) runs it to it1500. Seed-44 flat resumed (link 3 13520792). Expired logs of finished swap and critic jobs moved to outputs/slurm/resolved/ (includes MA s1 link 1, whose it250 = 62.2 is recorded above).

**09:25 (10-28) — the protective critic replicates on a second RL seed.** `game24cd_swapMA_svg2d_s1` (Muon flat policy + AdamW-trunk critic, RL seed 1): 64.6 → 62.2 → **65.7** at it250/500 (+1.1). The native critic with the same seed went 45.8 → **35.9** (−28.8). Seed 0 went 59.7 → 66.0 (+2.1) vs native 53.3 → 45.9. So at it500 the AdamW-trunk critic turns a −18 to −29 collapse into a small gain on both seeds. ELF-M pretraining hit the 40 h limit (TIMEOUT, last checkpoints 78,120 flat / 97,650 hier, in micro-steps); resubmitted with auto-resume (13521888 flat, 13521889 hier → Game24 SFTs 13521890/13521891); the old dependents were cancelled.

**10:05 (10-28) — MA seed 0 final: −0.9; seed-44 flat −9.1 at it750.** `game24cd_swapMA_svg2d_s0` (Muon flat policy + AdamW-trunk critic): 59.7 / 66.0 / 66.9 / 68.5 / 66.7 / **63.0** at it250…1500 (−0.9 at the final, peak +4.6 at it1000). The native pairing ended at 55.0 (−9.2) and was below 60 from it250 on. So the AdamW-trunk critic prevents the collapse (+8.0 vs native at the final, +20.1 at it500), but over the full 1500 iterations it does not deliver a lasting gain; accuracy drifts back after it1000. Honest summary for the paper: the swap removes the collapse and gives a mid-run gain, the final is roughly neutral. The best-validation checkpoint would be reported alongside the final. Seed-44 flat (`game24cd_flat_ft44_svg2d_s0`, base 64.4): 61.3 / 56.2 / **55.3** at it250/500/750 (−9.1). The registered prediction (below base) holds, and the paired hier seed-44 run ended −0.8. Started: protective-critic test on the seed-43 base (AdamW-trunk critic trained on the seed-43 Muon flat policy's rollouts, `swap-critic-MA43` 13522191 → RL `game24cd_swapMA43_svg2d_s0` next; native seed-43 flat ended −13.8). Seed-44 flat link 4 queued (13522192).

**10:35 (10-28) — replication at it1000: +4.1. Seed-44 flat −16.0.** MA s1 (Muon flat policy + AdamW-trunk critic, RL seed 1): 62.2 / 65.7 / … / **68.7** at it1000 (+4.1). Native seed 1 was 48.1 at it1000. This matches seed 0 (68.5 at it1000, then 66.7 / 63.0), so the drift after it1000 is the thing to watch in the seed-1 final. Seed-44 flat: 61.3 / 56.2 / 55.3 / **48.4** at it1000 (−16.0). The MA43 critic is waiting for a GPU (debug 5/5); its RL link goes in as soon as the cap allows.

**11:00 (10-28).** MA s1: 68.7 → **66.4** at it1250 (+1.8). It drifts back after it1000, just like seed 0 (68.5 → 66.7 → 63.0). The MA43 critic is training (13522191), and its RL is queued (13523491 → 13523492). Seed-44 flat link 4 is running toward the final.

**11:40 (10-28) — finals: replication +1.9 (native −14.5); seed-44 flat −17.4.**
- **MA s1 final** (`game24cd_swapMA_svg2d_s1`, Muon flat policy + AdamW-trunk critic, RL seed 1): 64.6 → 62.2 / 65.7 / … / 68.7 / 66.4 / **66.5** (**+1.9**). The native seed-1 pairing finished at 50.2 (−14.5).

| RL seed | native critic (Muon trunk) final | AdamW-trunk critic final | peak with AdamW trunk |
|---|---|---|---|
| 0 | 55.0 (−9.2) | 63.0 (−0.9) | 68.5 at it1000 |
| 1 | 50.2 (−14.5) | **66.5 (+1.9)** | 68.7 at it1000 |

Mean final: native −11.9, swapped +0.5 (difference +12.4 points). Both swapped runs peak at about +4.5 around it1000, then drift down.
- **Seed-44 flat final** (`game24cd_flat_ft44_svg2d_s0`, base 64.4): 61.3 / 56.2 / 55.3 / 48.4 / 48.7 / **47.0** (**−17.4**). The registered prediction (below base) holds. The paired seed-44 hier run ended −0.8, so the hierarchy is 16.6 points better on this base. Hier vs flat finals on the three Muon bases: seed 42 +2.4 vs −11.7 (means), seed 43 −8.7 vs −13.8, seed 44 −0.8 vs −17.4.
- Seed-43 protective test: MA43 critic best field score 0.737 (native seed-43 critic 0.819, lower than the other swaps). RL started 11:32 (base 68.8).

**11:50 (10-28) — the late drift in the swapped runs is a generalisation effect, not exploitation.** Training statistics in 125-iteration windows (`runs/rl/game24cd_swapMA_svg2d_s{0,1}/log.jsonl`): training-buffer success stays at **0.83–0.86 the whole run**, and critic AUC stays at 0.94–0.98. The objective decays smoothly (1067 → 41 for s0) and the learning rate is already down to 1–3e-7 in the drift window. In the native collapse, buffer success fell to **0.55** at it375–499 (s0), i.e. the policy was producing fewer correct answers on its own training prompts while the critic scored them highly. That never happens with the AdamW-trunk critic. The held-out drop after it1000 (s0 68.6 → 63.0, s1 68.7 → 66.5) comes with flat training success and continued token-norm growth (s0 27.9 → 32.9, s1 25.4 → 27.1; the larger growth matches the larger drop). The likely reading is slow over-fitting to the training number sets, not critic exploitation. Validation tracks it: the best-validation checkpoints are it1000 (s0, test **68.6**, +4.7) and it1250 (s1, test **66.5**, +1.9). With model selection on validation, as is standard and as the native runs get too (their best-val checkpoint is it0, i.e. no improvement at all), the swapped runs give **+4.7 / +1.9** against **0 / 0**. Both are reported in the paper: final iterate and best-val checkpoint. Third RL seed for the swap started (`game24cd_swapMA_svg2d_s2`, 13524399); seed-43 swap chain extended to 4 links (13524397/98).

**12:25 (10-28) — the protective critic works on a second base (seed 43) at it250.** `game24cd_swapMA43_svg2d_s0` (seed-43 Muon flat policy + AdamW-trunk critic trained on its own rollouts): 68.8 → **70.9** at it250 (+2.1). Buffer success 1.0 and token norm 20.6 at it400. The native seed-43 pairing was 68.9 → 60.5 (−8.4) at it250 and ended −13.8. Third RL seed on the seed-42 base (`game24cd_swapMA_svg2d_s2`, base 65.0) is at it70. The debug cap is full, so its link 2 goes in next tick.

**12:50 (10-28) — seed-43 base protected through it500; third seed dips at it250.**
- `game24cd_swapMA43_svg2d_s0` (seed-43 Muon flat policy + AdamW-trunk critic): 68.8 → 70.9 → **72.3** at it250/500 (**+3.5**). The native seed-43 pairing was 60.5 / 60.2 (−8.7) at the same points. Buffer success 0.88, token norm 22.6 at it550.
- `game24cd_swapMA_svg2d_s2` (seed-42 base, AdamW-trunk critic, RL seed 2): 65.0 → **55.1** at it250 (−9.9), buffer success 0.72 at it290. Native seed 2 was 48.3 (−16.8) at it250. This dip is much deeper than seeds 0/1 (−4.2/−2.4). The protection is partial on this seed so far. Seeds 0/1 both recovered by it500, and it500 decides. Link 2 queued (13525527).

**13:25 (10-28) — third seed recovers; seed-43 base still climbing.**

| run (AdamW-trunk critic) | base | it250 | it500 | it750 | native at it500 |
|---|---|---|---|---|---|
| seed-42 base, RL seed 0 | 63.9 | 59.7 | 66.0 | 66.9 | 45.9 |
| seed-42 base, RL seed 1 | 64.6 | 62.2 | 65.7 | 65.0 | 35.9 |
| seed-42 base, RL seed 2 | 65.0 | 55.1 | **66.6** | | 36.2 |
| seed-43 base, RL seed 0 | 68.8 | 70.9 | 72.3 | **72.8** | 60.2 |

At it500 all four swapped runs are above their base (+1.0 to +3.5). Their native counterparts are 8.7 to 28.8 points below base. The third seed dipped deeper at it250 (−9.9) and recovered fully by it500. Seed-43 base: +4.0 at it750, buffer success 0.81. Third-seed link 3 queued (13526188).

**13:40 (10-28) — is the collapse just mode collapse? No (from the existing failure-mode analysis).** The user raised it. `runs/analysis/fm_game24cd_{flat,hier}.json` (`hcfrl/analysis/failure_modes.py`, 500 val prompts per checkpoint of the seed-0 runs): in the flat Muon collapse, output diversity does **not** fall. The distinct-output fraction is 0.84 at it0, 0.85 at it250 and **0.88 at it500** (the bottom of the collapse), and token repetition (0.019–0.020) and length (22.7–23.4 chars) are unchanged. What changes is correctness *within* well-formed outputs. Arithmetic errors double (13.8% → **28.4%** at it500), invalid-operand errors double (3.0% → 6.8%), and step-count errors spike at it250 (0.4% → 10.4%). The policy keeps producing varied, well-formatted solution chains, but more of them contain wrong arithmetic that the critic scores well. That is critic exploitation (Goodhart), not loss of diversity. Supporting it: in the native runs training-buffer success drops (0.85 → 0.55) while the actor is ascending the critic, and swapping only the critic's trunk (same policy, same data, same RL settings) removes the collapse. Mode collapse would not depend on which trunk the critic uses. Hier for comparison: diversity 0.82–0.85, arithmetic errors fall 19.6% → 13.4%. TODO: run the same failure-mode analysis on the swapped runs for the paper (`failure_modes.py --run outputs/rl/game24cd_swapMA_svg2d_s0`).

**14:10 (10-28) — AdamW flat replicates (no collapse); seed-43 swap holds at it1000.**
- **AdamW flat seed 1, final** (`game24cdA_flat_svg2d_s1`, ran on the GPU partition 11:00–12:42): 59.8 → 60.0 / 60.7 / 60.9 / 61.7 / 62.4 / **62.0** (+2.2). Monotone, no dip. **AdamW flat seed 2** (running): 59.4 → 60.7 / 61.4 / 62.6 / **62.6** at it1000 (+3.2). With seed 0 (+3.1), that makes three seeds where the AdamW flat model with its own AdamW critic never collapses. AdamW hier s2 is also running (its eval lines were unreadable this tick).
- Seed-43 swap: 70.9 / 72.3 / 72.8 / **71.3** at it250…1000 (+2.5; native 56.2 → 55.1). Third seed: 55.1 / 66.6 / **66.6** at it250/500/750 (+1.6; native 42.8 at it750).
- k-means hier Game24 fine-tune finished (helfK, 09:58); its gate0a 13518377 is pending on the GPU partition.
- Queue priorities: set `nice=500` on the lower-urgency GPU-partition jobs (refit8, AdamW hier s1, AdamW+wd chain head, ELF-M pretrains) so the fresh independent Muon Countdown SFT (13519017, the relatedness-vs-Muon test) starts first. `scontrol top` is not permitted for users.

**14:30 (10-28) — deferred analysis plan (after all experiments finish; agreed with the user).** No new RL needed. (1) Exploit direction: before/after-collapse latent difference on fixed prompts, decoded, to show which tokens change (prediction: arithmetic and operands, format intact). (2) Linear probes on each critic's features (native Muon, sibling Muon, AdamW-trunk) for step-level arithmetic correctness. Prediction: the AdamW-trunk critic separates correct from incorrect steps along the exploited directions and the Muon critics do not. (3) Shared blind spot: policy vs critic representation overlap on wrong-vs-right steps. (4) Activation patching native ↔ AdamW critic by layer. (5) Optional SAE on critic features. Also deferred: weight-spectrum and subspace-overlap measurements for the theory, a failure-mode analysis of the swapped runs (`failure_modes.py`), and the subspace-restricted actor step (proposed 14:20). Requirement: keep the policy checkpoints and critics of these runs: game24cd_flat_svg2d_s{0,1,2}, game24cd_swapMA_svg2d_s{0,1,2}, game24cd_swapAM_svg2d_s0, game24cd_swapMM43_svg2d_s0, game24cd_swapMA43_svg2d_s0, game24cdA_flat_svg2d_s0, and critics outputs/critic/game24cd_{flat,swapMA,swapAM,swapMM43,swapMA43}_2d, game24cdA_flat_2d. Copied to ~/hcfrl_backup/interp/ (home, persistent), it0 / it500 / it1500 / best only.

**14:55 (10-28) — AdamW hier s1 final; swap runs late; third base started.**
- **AdamW hier seed 1, final** (`game24cdA_hier_svg2d_s1`): 69.2 → 69.0 / 70.0 / 68.8 / 70.1 / 70.1 / **68.9** (−0.3). Stable, no gain (seed 0 was +1.9). AdamW hier seed 2: 69.6 / 70.9 / 70.4 at it750/1000/1250. AdamW flat seed 2: 62.6 / 62.6 / 62.3 at it750/1000/1250 (+2.9 at it1250).
- Seed-43 swap: 70.9 / 72.3 / 72.8 / 71.3 / **69.0** at it250…1250 (+0.2). Third seed: 55.1 / 66.6 / 66.6 / **65.9** at it1000 (+0.9). Both show the same late drift as seeds 0/1. Finals ≈15:30 / 16:30.
- Swap on the seed-44 base started: critic `swap-critic-MA44` 13528031 (AdamW trunk on the seed-44 Muon flat policy's rollouts) → RL `game24cd_swapMA44_svg2d_s0` 13528032. Native seed-44 flat ended −17.4.
- Interpretability backup: 13 GB copied to ~/hcfrl_backup/interp (re-run after the remaining swap finals).

**15:35 (10-28) — finals: seed-43 swap, AdamW seed 2 (flat and hier).** Final iterate / best-validation checkpoint (Δ test vs base):

| run | base | final | best-val (it) | native counterpart final / best-val |
|---|---|---|---|---|
| swap, seed-43 base (`game24cd_swapMA43_svg2d_s0`) | 68.9 | 69.2 (**+0.3**) | 72.9 (it750, **+4.0**) | 55.1 (−13.8) / 68.9 (it0, 0) |
| AdamW flat s1 | 59.9 | 62.0 (+2.1) | 61.8 (it1000, +1.9) | |
| AdamW flat s2 | 59.5 | 62.3 (+2.8) | 62.6 (it1000, +3.1) | |
| AdamW hier s1 | 69.3 | 69.0 (−0.3) | 70.2 (it1000, +0.9) | |
| AdamW hier s2 | 68.2 | 70.2 (+2.0) | 70.9 (it1000, +2.7) | |

Swap summary, finals (best-val): seed-42 base RL s0 −0.9 (+4.7), s1 +1.9 (+1.9); seed-43 base +0.3 (+4.0); s2 pending (65.9 at it1000). Native: −9.2 / −14.5 / −13.8 with best-val = base. AdamW recipe, 3 seeds each: flat +3.1 / +2.1 / +2.8 (mean +2.7), hier +1.9 / −0.3 / +2.0 (mean +1.2); no AdamW run collapses. The seed-44 swap critic is training (13528031), with RL links 13528032 → 13529213 → 13529214. `runs/` archived (83 RL runs).

**16:15 (10-28) — third swap seed final +2.1; swap table complete for two bases.** `game24cd_swapMA_svg2d_s2`: 65.1 → 55.2 / 66.7 / 66.7 / 66.0 / 68.4 / **67.2** (+2.1; best-val = final). The native seed 2 ended 52.1 (−13.0). Seed-44 swap critic: best field score 0.801. Its RL (13528032) is waiting for a GPU. The paused hier lr1e-7 run resumes alongside it (13531026).

**Critic swap, all completed runs (Muon flat policy; Δ test vs base, final / best-val):**

| base, RL seed | own (Muon) critic | AdamW-trunk critic |
|---|---|---|
| seed-42 base, s0 | −9.2 / 0 | −0.9 / +4.7 |
| seed-42 base, s1 | −14.5 / 0 | +1.9 / +1.9 |
| seed-42 base, s2 | −13.0 / 0 | +2.1 / +2.1 |
| seed-43 base, s0 | −13.8 / 0 | +0.3 / +4.0 |
| **mean** | **−12.6 / 0** | **+0.9 / +3.2** |

Final-iterate improvement from the swap: +8.3, +16.4, +15.1, +14.1 (mean **+13.5**, all four positive). Min at it500 across the swapped runs: +1.1 vs native max −8.7. Seed-44 base pending.

**18:20 (10-28).** Seed-44 swap (`game24cd_swapMA44_svg2d_s0`): 64.4 → 61.5 at it250 (native 61.3); it470 buffer success 0.78; link 2 running. k-means hier gate0a (13518377) started on the GPU partition (critic pretraining → 3 RL seeds). hier lr1e-7 link 3 queued (13535785). Paper draft rewritten on request (paper/main.tex, critic-exploitation framing; old draft kept as main_2026-10-26.tex).

**18:45 (10-28) — the protective critic works on the third base.** `game24cd_swapMA44_svg2d_s0` (seed-44 Muon flat policy + AdamW-trunk critic): 64.4 → 61.5 → **65.3** at it250/500 (**+0.9**). The native seed-44 pairing was 61.3 → **56.2** (−8.2) at the same points and ended −17.4. Buffer success 0.78 at it600. Swapped runs at it500 are now above base on all three Muon bases and all three RL seeds (5/5): +2.1, +1.1, +1.6 (seed-42 base), +3.5 (seed 43), +0.9 (seed 44). Natives at it500: −18.3, −28.8, −28.9, −8.7, −8.2.

**19:25 (10-28) — combined-remedy test started; k-means critic ready.**
- Seed-44 swap: 61.5 / 65.3 / **64.9** at it250/500/750 (+0.5; native 55.3 at it750). hier lr1e-7: 65.7 / **66.0** at it250/500 (+1.2 from 64.8; flat lr1e-7 was 65.1 at it500 and ended −1.0).
- k-means hier gate0a finished (critic pretrained; base checkpoint helfK checkpoint_2800). Its three RL seeds (13518378–80) are pending on priority on the GPU partition, and its base accuracy will be read from their it0 evals.
- **Combined remedy, discussed with the user:** the Muon hier policy (seed-42 base, task map) with a critic built on the AdamW *hier* model (`critic_swap.slurm configs/helfA_b_game24_cd.yml game24cdA_hier game24cd_hier game24cd_swapHMA`, job 13536932). Then RL `game24cd_swapHMA_svg2d_s0` (args configs/helf_b_game24_cd.yml outputs/helf_b-game24-cd data/game24 game24cd_swapHMA 0 64 56 1500). Native hier on this base: +5.2 / +3.9 / +5.4. Question: does the swap add to the hierarchy, or is the hierarchy what protects when the critic cannot be swapped?

**19:55 (10-28).** Seed-44 swap: 61.5 / 65.3 / 64.9 / **69.2** at it250…1000 (**+4.8**). The native run was 48.4 (−16.0) at it1000. hier lr1e-7: 65.7 / 66.0 / 64.9 at it250/500/750 (+0.1). It stays near base like flat lr1e-7 (63.6 at it750). The hierarchy-swap critic (13536932) is waiting for a GPU. hier lr1e-7 link 3 (13535785) is held so the critic goes first; release it once the critic starts.

**20:55 (10-28).** Hierarchy-swap critic (AdamW-hier trunk on the Muon hier seed-42 rollouts): best field score **0.823**. Its RL (13537549 → 13538443) is waiting for a GPU. Seed-44 swap: … / 69.2 / **66.5** at it1000/1250 (+2.1; native 48.7). Final ≈21:15. hier lr1e-7: 65.7 / 66.0 / 64.9 / **65.3** at it250…1000 (+0.5). Link 4 queued.

**21:20 (10-28) — seed-44 swap final; the swap table is complete on three bases.** `game24cd_swapMA44_svg2d_s0`: 64.4 → 61.6 / 65.3 / 65.0 / 69.2 / 66.5 / **63.8** (final −0.6; best-val it1000 **69.2, +4.8**). Native: 47.0 (−17.4), best-val = base.

| base, RL seed | native final / best-val | AdamW-trunk final / best-val | swap gain (final) |
|---|---|---|---|
| seed-42, RL 0 | −9.2 / 0 | −0.9 / +4.7 | +8.3 |
| seed-42, RL 1 | −14.5 / 0 | +1.9 / +1.9 | +16.4 |
| seed-42, RL 2 | −13.0 / 0 | +2.1 / +2.1 | +15.1 |
| seed-43, RL 0 | −13.8 / 0 | +0.3 / +4.0 | +14.1 |
| seed-44, RL 0 | −17.4 / 0 | −0.6 / +4.8 | +16.8 |
| **mean (5 runs, 3 bases)** | **−13.6 / 0** | **+0.6 / +3.5** | **+14.1** |

All five swapped runs improve on their native counterpart (min +8.3), and all five have a best-validation checkpoint above base (+1.9 to +4.8). The hierarchy-swap RL (`game24cd_swapHMA_svg2d_s0`) started 21:04 (it0 64.8). hier lr1e-7: 65.3 / 65.5 at it1000/1250 (+0.7). Archived (85 RL runs).

**22:00 (10-28) — hier at 10× smaller step: final +0.6.** `game24cd_hier_svg2d_lr1e-7_s0` (base 64.8): 65.7 / 66.0 / 64.9 / 65.3 / 65.5 / **65.4** (+0.6). The matched flat run ended −1.0. Step-size table, seed-42 base, final Δ:

| actor lr | flat | hier |
|---|---|---|
| 1e-7 | −1.0 | +0.6 |
| 3e-7 | −14.4 | +2.5 |
| 1e-6 (published) | −11.7 (3 seeds) | +4.8 (3 seeds) |
| 3e-6 | — | −9.1 |

At the smallest step both models are safe and neither learns. The hierarchy opens a window (3e-7 to 1e-6) where the flat model has none, and it also closes by 3e-6. Hierarchy-swap (`game24cd_swapHMA_svg2d_s0`): 64.8 → **63.8** at it250 (−1.0; native hier on this base 66.1 at it250), buffer success 0.84 at it430. Checkpoint backup re-run done.

**22:20 (10-28) — K&K critic swap started (the second-task test; required for the paper).** New `configs/elfA_b_kk.yml` (AdamW, abs lr 1e-3 at batch 256, otherwise identical to the Muon K&K SFT). Chain: AdamW K&K SFT `sft-kk-elfA` 13539889 (GPU partition, ≈1.7 h) → critic `swap-critic-kkMA` 13539890 (AdamW K&K trunk, epoch-4 checkpoint_4684 like the Muon base, trained on the Muon K&K flat policy's rollouts) → RL seeds 0/1/2 `kk_swapMA_svg2d_s{0,1,2}` on debug chains, 1,000 iterations (args configs/elf_b_kk.yml outputs/elf_b-kk data/kk kk_swapMA <s> 64 56 1000). Native K&K flat finals at it1000: −7.3 / −7.6 / −11.1 (mean −8.7).

**22:45 (10-28).** Hierarchy-swap (`game24cd_swapHMA_svg2d_s0`, Muon hier policy + AdamW-hier-trunk critic): 64.8 → 63.8 → **66.1** at it250/500 (+1.3). Native hier on the same base: 66.1 / **70.8** (+6.1) at it250/500. So far the AdamW-trunk critic does not add to the hierarchy and gains less (token norm 29.7 at it730, higher than native hier). Final ≈02:00. The AdamW K&K SFT was moved from the GPU partition (pending on priority) to two resumable 1-h debug links (13540620 → 13540621, auto-resume from outputs/elfA_b-kk). Its swap critic gets resubmitted after link 2 (old 13539889/13539890 cancelled).

**23:45 (10-28) — the hierarchy-swap run collapses late: the AdamW-trunk critic is NOT protective for the hierarchical policy.** `game24cd_swapHMA_svg2d_s0` (Muon hier policy, seed-42 base, task map; critic on the AdamW *hier* model's trunk, trained on the Muon hier policy's rollouts, field 0.823): 64.8 → 63.8 / 66.1 / 65.4 / **51.1** at it250…1000 (**−13.7** at it1000). Native hier on the same base: 66.1 / 70.8 / 70.3 / 70.8 (+6.1 at it1000), final +5.2. Training signals, 125-iteration windows: buffer success 0.84–0.87 through it750, then **0.69 / 0.64** at it875–1000. Token norm 18 → 32 and the objective decays 708 → 42: the critic-exploitation signature, arriving later than in the flat native runs. Consequences: (i) "any differently trained critic protects" is false as a general statement. On the flat Muon policy the AdamW-trunk critic protects on 5/5 runs, on the hierarchical Muon policy it is exploited (one run). (ii) The hierarchy's protection under its own critic does not transfer to a foreign critic. The native hier critic and policy share the jointly trained denoiser that keeps code and tokens consistent (cf. the consistency trace); a foreign trunk breaks that pairing. (iii) The paper's claim has to be stated as an interaction between the policy and the critic's representation, not as "independent critics are safe". Caveat: one seed, and native hier also fell on the seed-43 base (−8.7). Process note: a temporary analysis script was briefly written to /tmp on the login node and deleted immediately; analysis now runs locally.

**00:05 (10-29).** AdamW K&K SFT reached epoch 4 (checkpoint_4684, 23:53; ≈15 min/epoch). The remaining SFT links were cancelled (epochs 5–8 not needed), and the K&K swap critic (13542191, AdamW K&K trunk on Muon K&K flat rollouts) is running. K&K swap RL seeds 0/1 queued (13543131 / 13543132, afterok critic); seed 2 follows when a slot frees. Hierarchy-swap continues to its final (link 3 running, 51.1 at it1000). A second RL seed of the hierarchy-swap is queued next, to check whether the late collapse replicates.

**00:50 (10-29) — hierarchy-swap final −8.8; K&K swap critic too weak (redoing).**
- `game24cd_swapHMA_svg2d_s0` final: 64.8 → 63.8 / 66.1 / 65.4 / 51.1 / 56.1 / **56.0** (**−8.8**). Native hier on the same base: +5.2. Second RL seed (`game24cd_swapHMA_svg2d_s1`, 13544738) started 00:44 (native hier s1 final +3.9).
- K&K swap critic (AdamW K&K trunk at epoch 4, checkpoint_4684, on Muon K&K flat rollouts): best field score **0.576**, against **0.965** for the native K&K critic. The Game24 swaps were matched (0.74–0.88 vs 0.84). A K&K comparison with this critic would confound "foreign trunk" with "weak critic". Cause: the AdamW K&K model was stopped at epoch 4 of 8. Fix: AdamW K&K SFT resumed to completion (debug links 13544739 → 13544740, auto-resume from checkpoint_4684), then a second swap critic from the final checkpoint (`kk_swapMA2`); only a field-score-matched critic is used for the K&K claim. RL seed 0 with the weak critic (`kk_swapMA_svg2d_s0`, it0 79.2) continues as a documented weak-critic data point. Its seed-1 copy was cancelled.
- Process: interp backup re-run started.

**01:30 (10-29).** Weak-critic K&K swap (`kk_swapMA_svg2d_s0`, critic field 0.576): 79.2 → **69.3** at it250 (−9.9). Stopped after link 1 (≈it430) so the AdamW K&K SFT resume gets the GPU. It is recorded only as a weak-critic data point, not as evidence about trunk swapping. Second hierarchy-swap seed (`game24cd_swapHMA_svg2d_s1`): 64.2 → **60.6** at it250 (−3.6; native hier s1 65.2 at it250). Link 2 queued. GPU partition: the AdamW + weight-decay Countdown SFT (13503070, the weight-scale control) started 01:07.

**02:10 (10-29).** Fresh independent Muon Countdown SFT (`sft-cd-elf-s7` 13519017, the relatedness-vs-Muon test) started 01:33 on the GPU partition (≈15.5 h → ≈17:00, then the Game24 fine-tune, critic and RL). The weight-decay control SFT is also running (since 01:07). AdamW K&K SFT at epoch 7 (checkpoint_8197); epoch 8 ≈02:20. Second hierarchy-swap seed: 64.2 → 60.6 → **66.9** at it250/500 (+2.7; native hier s1 67.1 at it500). Link 3 queued.

**02:30 (10-29).** AdamW K&K SFT finished epoch 8 (checkpoint_9368, 02:13). Swap critic v2 on that trunk (`swap-critic-kkMA2` 13548278, Muon K&K flat rollouts) → RL seed 0 `kk_swapMA2_svg2d_s0` 13548279 (afterok). It is used for the K&K claim only if the critic's field score is close to the native 0.965. Hierarchy-swap s1 link 4 queued.

**03:00 (10-29) — the hierarchy-swap late collapse replicates.** `game24cd_swapHMA_svg2d_s1` (Muon hier policy + AdamW-hier-trunk critic, RL seed 1): 64.2 → 60.6 / 66.9 / 65.1 / **40.9** at it250…1000 (**−23.3**). Native hier seed 1: 65.2 / 67.1 / 70.5 / 69.0 (+4.8 at it1000). Seed 0 had the same shape (66.1 at it500 → 51.1 at it1000, final −8.8). Two of two: for the hierarchical Muon policy, the AdamW-hier-trunk critic is exploited after ≈750 iterations, while the native (own-trunk) critic is not. For the flat Muon policy the AdamW-trunk critic protected 5/5. The swap result is therefore architecture-dependent. The paper's claim has to be the interaction (which critic a given policy can exploit), not "use an independent critic".

**03:20 (10-29) — K&K swap critic still weak; diagnosing.** Critic v2 (AdamW K&K epoch-8 trunk): field score 0.637 / 0.662 / 0.652 / **0.676** / 0.675 / 0.671 over 6 epochs. The native K&K critic (Muon trunk, same rollouts, same recipe) reached 0.929 / 0.952 / 0.951 / 0.961 / **0.965** / 0.956. The gap is not about training length of the base (epoch 4 → 8 moved 0.58 → 0.68). Its RL run (13548279) was cancelled because the comparison would be unfair. Diagnostics launched: (a) AdamW K&K base test accuracy (`scripts/eval_base.slurm`, 13549006; Muon K&K base 79.2), to see whether the AdamW model simply solves K&K worse; (b) critic v3 with a longer, faster-adapting trunk fit (`--epochs 12 --lr_trunk 3e-5`, 13549007; `scripts/critic_swap.slurm` now passes extra args to pretrain_critic). Until a quality-matched critic exists, the K&K swap is inconclusive.

**04:00 (10-29) — K&K AdamW-trunk swap abandoned; independent-Muon K&K test started; hierarchy-swap s1 final.**
- AdamW K&K base test accuracy **71.3** (1,000 rollouts; Muon K&K base 79.2). Critic v3 (`--epochs 12 --lr_trunk 3e-5`): field 0.70 / 0.74 / 0.66 / 0.70 / 0.70 / 0.71 / 0.71 / 0.69 / 0.64 over epochs 2–10, peak **0.74** vs native 0.965. Conclusion: on K&K the AdamW model's features do not support a quality-matched critic (≤0.74 under three fits), so the K&K AdamW-trunk swap cannot be run fairly. It is reported as a limitation. Data points kept: the weak-critic run (field 0.58) fell −9.9 by it250.
- Replacement K&K test, quality-matched and on-mechanism: a critic on an **independent Muon K&K model** (fresh SFT from the OWT init, seed 7; same recipe and epoch-4 checkpoint as the policy's base), trained on the seed-42 Muon K&K policy's rollouts. Muon critics reach ≈0.96 on K&K. Protected → an unrelated trunk suffices (relatedness is the variable); collapse → Muon-trained trunks are exploitable by Muon policies. Chain: SFT debug links 13549441 → 13549442 → critic `swap-critic-kkMM7` 13549450 → RL (`configs/elf_b_kk.yml outputs/elf_b-kk data/kk kk_swapMM7 <s> 64 56 1000`).
- **Hierarchy-swap seed 1 final:** 64.2 → 60.6 / 66.9 / 65.1 / 40.9 / 58.8 / **62.5** (−1.7; native hier s1 +3.9). Both hierarchy-swap seeds dip deeply at it1000 (51.1, 40.9) and end below native hier (−8.8, −1.7 vs +5.2, +3.9).
- GPU partition: k-means hier RL seeds 0/1/2 (13518378–80) started ≈03:00; AdamW+wd and fresh-Muon Game24 SFTs running.

**04:40 (10-29) — k-means hierarchy, first evaluations.** `game24cdK_hier_svg2d_s{0,1,2}` (H-ELF with the k-means coarse map Γ_kmeans, fixed before SFT; native critic; GPU partition since ≈03:00). Bases 63.0 / 62.9 / 62.1 (task-map hier 64.7 / 64.3 / 64.9). s0: 61.7 / **65.2** at it250/500 (+2.2). s1: 63.0 / 63.4 / **62.4** at it250/500/750 (−0.5; buffer success 0.63 at it750, watch). s2: 62.9 / **67.0** (+4.9). Task-map hier at it500: 70.8 / 67.1 / … Independent Muon K&K SFT at epoch 2. K&K RL seeds 0/1 on its critic queued (13549xxx, afterok critic 13549450).

**05:20 (10-29).** Independent Muon K&K SFT (seed 7) reached epoch 5. The remaining training was cancelled (epoch 4 = checkpoint_4684 is the trunk, matching the policy's base epoch). Its swap critic `swap-critic-kkMM7` 13549450 is starting, then K&K RL seeds 0/1. k-means hier at it1000: s0 **64.7** (+1.7), s1 **62.4** (−0.5), s2 **67.3** (+5.2). Task-map hier at it1000: +6.1 / +4.7 / +5.9.

**05:55 (10-29) — quality-matched K&K critic on an independent Muon model.** `swap-critic-kkMM7` (trunk: Muon K&K flat model fine-tuned from the OWT init with seed 7, epoch 4; trained on the seed-42 Muon K&K policy's rollouts): field 0.927 / 0.946 / 0.942 / **0.952** / 0.939 / 0.943. Native 0.965. Matched. RL `kk_swapMM7_svg2d_s{0,1,2}` running/queued (s0 13550111 → 13552031, s1 13550112 → 13552032, s2 started). Native K&K flat −7.3 / −7.6 / −11.1 at it1000. This is also the cleanest relatedness test so far: the trunk is Muon-trained but not a relative of the policy (fresh SFT from OWT). k-means hier: s0 64.7 / **63.3** at it1000/1250, s1 62.4 / **61.2**, s2 **67.3** at it1000.

**06:30 (10-29) — k-means hierarchy finals; K&K relatedness test, first readings.**

| coarse map (H-ELF, Game24, seed-42 SFT recipe, native critic) | bases | finals Δ | mean | best-val Δ |
|---|---|---|---|---|
| task map Γ_task | 64.7 / 64.3 / 64.8 | +5.2 / +4.5 / +5.5 | **+5.1** | +6.1 / +6.2 / +7.3 |
| k-means map Γ_kmeans (`game24cdK_hier_svg2d_s*`) | 63.1 / 62.9 / 62.2 | −1.8 / −2.2 / +3.5 | **−0.2** | +2.1 / +0.6 / +4.8 |
| random map Γ_rand (base 69.9) | — | −1.5 / −4.3 / −0.6 | −2.1 | — |
| flat (reference) | 64.2 / 64.7 / 65.1 | −9.2 / −14.5 / −13.0 | −12.2 | 0 |

The k-means map protects nearly as much as the random one (no collapse; mean −0.2 vs flat −12.2) but gains much less than the task map: the k-means runs rise by it500 (+2.1 / +0.6 / +4.8) and then drift down. The coarse-stream result therefore splits cleanly into *protection* (any jointly denoised code: task, k-means, random) and *gain* (needs a task-informative code). Archived (94 RL runs).

K&K relatedness test (critic on an independent Muon K&K model, field 0.952): s0 79.2 → **76.1** at it250 (−3.1; native s0 72.8, −6.4), s1 78.6 → **73.4** (−5.2; native s1 72.2, −6.4). The early dip is milder than native but present. The native K&K runs dip at it250 then wander (−7.3 / −7.6 / −11.1 at it1000), so it500–1000 decide. s2 is waiting for a GPU.

**07:10 (10-29).** K&K relatedness s0: 79.2 → 76.1 / **75.7** at it250/500 (−3.5; native s0 72.8 / 75.2, −4.0 at it500). So far the independent-Muon critic tracks the native K&K curve within noise. s1 waits for link 2 (73.4 at it250). s2 at it190. Link 3 queued for s0/s1.

**07:55 (10-29) — K&K relatedness test: no protection so far.** Independent-Muon-trunk critic (field 0.952): s0 79.2 → 76.1 / 75.7 / **69.5** at it250/500/750 (−9.7; native s0 74.0 at it750, −5.2). s1 73.4 at it250 (link 2 running). s2 **72.6** at it250 (native s2 72.9). Unlike the AdamW-trunk critic on Game24, a critic on an unrelated *Muon* model does not prevent the K&K collapse, at least on seed 0 through it750. If this holds over three seeds, the operative property is how the critic's trunk was trained (Muon vs AdamW), not relatedness to the policy. The Game24 fresh-Muon test (s7 SFT, GPU partition, ≈17:00) is the same test on the main task. s2 links 2–3 queued.

**08:40 (10-29) — K&K relatedness: mixed so far.** Independent-Muon-trunk critic: s0 final **69.6** (−9.6; native s0 −7.3), trajectory 76.1 / 75.7 / 69.5 / 69.6. s1 73.4 / … / **77.8** at it250/750 (−0.8; native s1 65.3 at it750). s2 72.6 / **71.1** at it250/500 (native 72.9 / 70.1). One seed worse than native, one much better, one similar. K&K's native drop is small and noisy (−7 to −11 with large swings), so three finals and best-val are needed before any K&K conclusion. s1/s2 finals ≈09:30–10:30.

**09:30 (10-29) — K&K: a critic on an unrelated Muon model gives no protection.**

| K&K, RL seed | native critic (policy's own Muon trunk) final Δ | critic on an independent Muon K&K model (field 0.952) final Δ |
|---|---|---|
| 0 | −7.3 | −9.6 |
| 1 | −7.6 | −2.7 |
| 2 | −11.1 | −14.5 |
| mean | **−8.7** | **−8.9** |

Best-val checkpoints: native = base on all three; independent-Muon critic = base on two, +... (it250 76.1 < base 79.2) on the third, so no gain anywhere. With critic quality matched (0.952 vs 0.965), replacing the policy's own trunk by an *unrelated Muon-trained* trunk changes nothing on K&K. Combined with Game24 (the sibling Muon critic was exploited, the AdamW critic protected 5/5), the evidence now favours **"Muon-trained critic trunks are exploitable by Muon policies"** over "relatedness to the policy". Caveats: different task from the Game24 swap (K&K's AdamW trunk could not make a matched critic, so the K&K AdamW cell is missing), and K&K's native drop is noisier. The Game24 fresh-Muon test (s7 Countdown SFT running on the GPU partition, ≈17:00) is the same-task check. Archived (95 RL runs).

**10:30 10-29 — refit8 control (flat Muon s0, 4× critic refit: `--mc 8 --mc_extra 16`; `game24cd_flat_svg2d_refit8_s0`, job 13485886, finished 10-28 11:49, logged late).**

| it | 0 | 250 | 500 | 750 | 1000 | 1250 | 1500 |
|---|---|---|---|---|---|---|---|
| test | 63.9 | 51.4 | 59.5 | 57.7 | 60.4 | 60.6 | 60.2 |

Final **−3.8** (native flat s0 −9.2; AdamW-trunk-critic swap s0 −0.9). Best val is the it0 value (63.2), so it never improves. Critic AUC stays 0.97–0.98 and buffer success is around 0.80 at the end. Reading: 4× more critic refitting softens the collapse. It still shows the same early crash at it250 (51.4, vs native 53.3) and only partly recovers, with no gain. Refit budget is a sixth partial fix, not a cure. The independent AdamW critic does better at the native refit budget.
- 10:35: submitted refit8 seeds 1/2 as debug chains (13561568→13561569, 13561570→13561571; more links added as they run) for a 3-seed refit-budget control. Native flat s1/s2 finals −14.5/−13.0.
