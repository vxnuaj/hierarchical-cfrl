# Literature survey (2026-10-23) — condensed

## Novelty verdict
No paper combines hierarchical/multi-scale representations with pathwise (reparameterized) RL for a language flow/diffusion model.
- CFRL (2610.19080) is the only reward-backprop RL for any language diffusion/flow model; zero citations so far.
- H-CDLM (2610.08738), HC-DLM (2610.02193, Schwing group; latent as "a plan"; Sudoku/Countdown already set up), HMDLM (2610.29255), HDLM (2510.08632), NCM (2610.14162), Cola DLM (2605.06548): hierarchical text generators, none with RL.
- All dLLM RL is score-function (d1, wd1, DiffuCoder, SPG, ESPO, EGSPO, MDPO, d2, ...). Critics on intermediate states exist only in TraceRL (2509.06949, value model w/ token GAE) and CAPR (2606.04396, block value head).
- Latent-reasoning RL (LaDi-RL 2602.01705, LTC 2608.01593, Latent-GRPO, LEPO) is GRPO on single-scale latents; SofT-GRPO (2511.06411) is the only reparameterized one, no hierarchy.

## Closest threats
1. LaDi-RL (2602.01705): Flow-GRPO on a latent diffusion policy (LaDiR VAE, 64 latent tokens, one block), two-level latent/text ROLLOUT scheme, Qwen3-8B code 60.6→77.1. Score-function, no critic. A v4 with critic or multi-level latent is the direct competitor.
2. HC-DLM / H-CDLM authors adding RL with the CFRL recipe.
3. Image community: VAR RL Done Right (2601.02256, scale-aware credit, code NextFlow), Stepwise-Flow-GRPO (2603.28718), VARD (2505.15791: value fn on noisy latents + pathwise update through it, last 10 steps; closest image analogue of CFRL).
Also: ELF-REG (2609.29102, code lizeyu090312/scaling_dLM): ELF-style flow + REPA + a jointly-denoised GLOBAL representation from a frozen Qwen3-1.7B teacher; ELF-REG-L 652M GSM8K 56.0. This is a competing notion of "hierarchy" (one global vector) AND a possible stronger base with a GSM8K harness.

## Theory anchors for the mechanism
- Mohamed et al. 1906.10652: pathwise variance bounded by squared Lipschitz constant of the cost; score-function variance sums over coordinates the parameter does not affect; pathwise can be worse for high-frequency costs.
- MAGE 2004.14309 Prop 3.1: deterministic-policy-gradient bias is bounded by the ACTION-GRADIENT error of the critic, not its value error. → the quantity to measure is ||∇_s V_φ − ∇_s V*||, not BCE.
- Rosas 2606.06123 (rate-distortion abstractions): value error ≤ learning term (Bellman residual within abstraction) + abstraction term (bisimulation); Abel et al. 1701.04113. → coarse level makes the critic easier to fit at bounded bias.
- Metz 2111.05803, Suh 2202.00817: unrolled pathwise gradients explode with horizon (chaos); truncation to one step (CFRL) is standard (ReFL 1, DRaFT ≤10, VARD 10, FlowBP ≤1 Jacobian).
- Empirical evidence token-level critics are poor: VinePPO 2410.01679 (PPO value net barely beats random at ranking steps), EVPO 2604.19485, Value Flattening/SP3O 2609.18708 (critic predictions flat across intermediate states).
- Value-Gradient Hypothesis 2605.21654 (pathwise-like updates for LLM RL under differentiable rollouts).

## Positioning advice
Lead with the estimator argument: a critic on the coarse level has lower action-gradient error → lower pathwise bias/variance (MAGE + Mohamed). Use VinePPO/EVPO/value-flattening as motivation. Baselines: flat CFRL, score-function (CFRL-GRR / Flow-GRPO on embeddings), and discuss LaDi-RL.

## Continuous flow LMs with reasoning numbers (for context)
Lafa 7B GSM8K 73.8 (code+weights); Sigma-8B 89.1 (NVIDIA, no code); ELF-REG-L 652M 56.0; MLFM 31.2; S-FLM 170M 18.0 (TinyGSM); TESS 2 7B 36.5. Sub-1B with public training code: FLM 179M, CoBit 130M/462M, ELF 105-652M, ELF-REG, S-FLM, LangFlow 130M, Lafa-0.5B (GSM8K 36-38).

## Hierarchy-for-credit claims (text)
Only in explicit-token RL: HICRA 2509.03646 (advantage on planning tokens), HiPER 2602.16165 (ICML 2026, hierarchical plan-execute, HAE), LTC (thought/answer two-level credit), HarA 2610.04344. In latent models: HiLP 2608.05806 (higher-level latent = longer-horizon belief state), HC-DLM ("latent plan"). None in diffusion + RL.

## Reusable code
- ELF pytorch_elf (vendored). ELF-REG: github.com/lizeyu090312/scaling_dLM (GSM8K/MATH/HumanEval harness).
- Verifiers/reward plumbing: d1 (dllm-reasoning.github.io) has GSM8K/MATH/Countdown/Sudoku.
- Critic template: TraceRL github.com/Gen-Verse/dLLM-RL.
- Score-function on continuous flows: Flow-GRPO github.com/yifan123/flow_grpo.
- Truncated reward backprop reference: VADER github.com/mihirp1998/VADER. DiffusionRVGE github.com/jy-wang-tsinghua/DIFFUSIONRVGE (regression value-gradient, no backprop).
- Multi-scale RL reference: NextFlow github.com/ByteVisionLab/NextFlow.
- Clustering: HDLM github.com/zhouc20/HDLM (compute_cluster.py). LangFlow embeddings HF Continuous-Rivals-Discrete/langflow-*.
