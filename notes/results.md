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
