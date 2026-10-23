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
