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
