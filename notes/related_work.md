# Related work scan (2026-10-27)

Question: has anyone characterised critic/reward exploitation in pathwise (reward-gradient) RL on diffusion/flow models, tied it to the base model, or proposed a general fix? Searched on the user's request after the seed-43 and AdamW results.

## Directly relevant

- **Reward Sharpness-Aware Fine-Tuning for Diffusion Models** (Kim & Sim, CVPR 2026, arXiv 2603.21175). Reward-gradient methods (DRaFT-LV, AlignProp, ReFL, DRTune) hack the reward model; they attribute it to the *sharpness of the reward w.r.t. the input* and define sharpness S1 = r(x) − r(x − ρ∇r/‖∇r‖) (one-step worst case in a ball, ρ = 0.01). S1 correlates with hacking (r = −0.80 with PickScore on a DRaFT run). Fix (RSA-FT): maximise the reward at the worst-case perturbed sample x₀ + δ, δ = −ρ∇ₓr/‖∇ₓr‖ (and/or SAM on the policy parameters). Big gains on held-out preference metrics (AlignProp on SD1.5: ImageReward −0.03 → 0.27). This is the nearest competitor fix and the nearest predictor: in our setting r is the critic logit l_φ at step k and x is the latent x_k. Our one-shot "survival" displacement test is the same construction but measured on the *policy's* ability to heal, not on the critic's sharpness.
- **KL regularisation only changes the timescale of hacking** (arXiv 2606.28955, "Modification-Considering Value Learning"): runs with and without KL hack equally at equal reward; early stopping remains the practical safeguard; separate value heads per reward for multi-reward. Matches our GRR/KL-free observations and the "dip then partial recovery" shape.
- **Explicit Critic Guidance for Aligning Diffusion Models** (arXiv 2605.27736): diffusion model as its own timestep-conditioned value function on noisy latents + PPO; reports reward-hacking concerns, mitigated by joint training with complementary rewards. Closest architecture to CFRL's per-step critic, image domain.
- **Exploring the Design Space of Reward Backpropagation for Flow Matching** (arXiv 2606.11075): direct reward backprop pathologies = memory and *chained Jacobian products inflating the reward gradient toward early steps*; they cap chaining at one Jacobian factor (CFRL already does: one denoiser call). No learned critic.
- Surveys (arXiv 2505.17352, 2409.07253): DRaFT/AlignProp over-optimise the reward model; DRaFT's own mitigations were LoRA scaling (most effective), early stopping, KL. TDPO-R: per-step reward supervision + critic neuron reset.

## Muon vs AdamW

- **When Does Muon Help Agentic RL?** (arXiv 2607.16169): Muon's RL gains disappear under an RMS-matched control (Muon applies ≈3.5× AdamW's update RMS); the effect is Muon's *scale convention*, not its spectral structure. Directly supports testing our Muon/AdamW difference as an effective-step-size (safety-margin) effect → the AdamW flat 10×-actor-step run.
- **When Muon Optimizer Meets Adversarial Training** (arXiv 2605.26929) and **Muon Learns More Robust and Transferable Features than Adam** (arXiv 2606.09658): Muon-trained networks are *more* robust to input corruptions/attacks, with larger logit margins and higher effective rank. This cuts against a "Muon features are fragile" reading of our result; the consistent reading is scale: the actor's normalised step is relatively larger on the Muon model (weights 2.5×, activations 15× smaller), so the same lr explores much further in output space. Our lr-3e-7 flat run (still collapses, later) says the margin is not the whole story, but it fixes what the AdamW+weight-decay and 10×-step controls must show.
- Nothing links pretraining optimizer to reward-hacking susceptibility; that claim would be new.

## Not found

No work on learned per-step critics for diffusion/flow *language* models, on base-model-dependent exploitability, or on a pre-RL predictor of collapse. Those are open.

## Consequences for us

1. Cite the reward-gradient hacking literature; our contribution is the learned-critic + verifiable-reward + base-dependence version, plus the consistency/early-warning diagnostics.
2. Implement the RSA input-space step in `train_svg.py` (`--rsa_rho`: actor objective evaluated at x_k + δ, δ = −ρ∇ₓl_φ/‖∇ₓl_φ‖, one extra critic backward per step) and test on the flat Muon model: the general fix the user asked about, and the competitor the reviewers will name.
3. Measure critic sharpness S1 at the base (critic logit vs one-step worst-case latent displacement at k ∈ {24, 40}) across all bases and correlate with final Δ — the predictor analysis; the probe/survival code already builds the displacement.
