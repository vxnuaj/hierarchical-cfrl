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
