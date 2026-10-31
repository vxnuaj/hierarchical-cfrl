# Theory: why Muon-trained critics are exploitable by Muon-trained policies

Working notes for the mechanism section (started 2026-10-30). Status of each claim is marked
**[derived]**, **[checked numerically]**, **[measured on ELF]** or **[open]**.

## 1. First-order exploitation rule

Notation. A policy state s_k = (z~_k, x_{k-1}) lives in R^{L x D}. Let T be the data manifold's tangent space of
clean latents (per position: the top principal directions of the SFT targets' latents, `offspan.py`), N its
orthogonal complement. The critic logit is l_phi(s); write its input gradient g = grad_s l_phi = g_T + g_N.
One actor step moves the clean estimate by dx = -eta J P J^T (-g) where J = d x_hat / d theta and P is the
optimizer's preconditioner (AdamW). With K = J P J^T (the policy's output-space movement kernel):

    critic gain      dl  = eta (g_T^T K_TT g_T + 2 g_T^T K_TN g_N + g_N^T K_NN g_N)
    useful movement      = eta K_TT g_T + eta K_TN g_N          (the reward depends on the tangent component)

The head of the critic is fit only on states the policy produces, which vary (almost) only along T. Along N the
logit's slope g_N is set by the trunk's input Jacobian and is not constrained by any label, so its correlation with
the reward is zero on average. **[derived]** The share of the critic gain that is not backed by reward is

    E = g_N^T K_NN g_N / (g_T^T K_TT g_T)         ("exploitation index").

E factorises approximately into a critic term |g_N|^2/|g_T|^2 (how much the critic responds off the manifold) and a
policy term k_N/k_T (how cheaply the policy moves off the manifold, the mean eigenvalues of K on N and T). The
collapse needs both factors to be large, which is the 2x2 of Table relate: Muon policy + Muon critic collapses;
either an AdamW critic (small critic term) or an AdamW policy (small policy term) is safe. **[derived]**

## 2. Muon removes the spectral bias that keeps networks insensitive off the data manifold

Linear case, W in R^{d x n}, data covariance diag(lam_1..lam_n), targets Y = W* X + noise (std s), N samples.
The empirical least-squares solution W_ols has column i equal to W* e_i + O(s / sqrt(N lam_i)).

* Gradient flow from 0: W(t) e_i = (1 - exp(-lam_i t)) W_ols e_i. For an off-manifold direction (W* e_i = 0,
  lam_i -> 0) the sensitivity |W(t) e_i| ~ t s sqrt(lam_i / N) -> 0. The network stays insensitive to directions
  the data barely varies along (spectral bias). AdamW behaves like GD in this respect (elementwise scaling does not
  equalise the singular values of the gradient). **[derived]**
* **Proposition (polar flow fits every direction at unit speed).** Let L(W) = 1/2 tr((W - Wb) S (W - Wb)^T) with
  S = diag(lam_1..lam_n) the (empirical) input covariance and Wb the least-squares solution, whose columns Wb e_i
  are mutually orthogonal with norms c_i (d >= n). Then from W(0) = 0
      gradient flow  dW/dt = -(W - Wb) S :      W(t) e_i = (1 - exp(-lam_i t)) Wb e_i
      polar flow     dW/dt = -polar((W - Wb) S): W(t) e_i = min(t, c_i) Wb e_i / c_i
  Proof: under both flows W - Wb = -Wb diag(a_i(t)) stays column-orthogonal; (W - Wb) S = -Wb diag(a_i lam_i), whose
  polar factor is -Wb diag(1/c_i) on the columns with a_i > 0, so each column moves toward its target at unit speed
  until it arrives (a_i = 1 - t/c_i), independently of lam_i; for GD da_i/dt = -lam_i a_i. **[derived]**
  For a direction the targets do not depend on (W* e_i = 0), c_i = |noise_i| ~ s / sqrt(N lam_i): polar flow reaches
  sensitivity min(t, s/sqrt(N lam_i)), which grows without bound as lam_i -> 0; GD reaches t s sqrt(lam_i/N) -> 0.
  Discrete Muon with step eta for T steps is the case t = eta T (constant-norm steps).
* **AdamW is not a whitening method in this sense.** Adam normalises each parameter coordinate, i.e. it equalises
  rates only in the axis basis of the weights. Polar descent is rotation-equivariant (polar(R G Q) = R polar(G) Q)
  and equalises rates in the data's own eigenbasis, whatever its orientation. The data manifold of a network's
  inputs (T5 latents) is not axis-aligned, so AdamW keeps GD's spectral bias with respect to it, and Muon removes
  it. In the numerical check below the manifold is a random rotation, and AdamW behaves like SGD. **[derived; checked]**
* Muon / polar descent dW/dt = -polar(grad L) moves every singular direction of the gradient at unit rate, so the
  fit of column i is approached at a rate independent of lam_i: |W(t) e_i| ~ min(t, s / sqrt(N lam_i)). Off the
  manifold the sensitivity grows like lam_i^{-1/2} -- the network fits the label noise along low-variance
  directions (the whitening effect; cf. Wadia et al. 2021, "Whitening and second order optimization both make
  information in the dataset unusable during training"). **[derived, sketch; needs a clean statement]**

Numerical check (`hcfrl/theory/whitening.py`, runs/theory/whitening_v1.json; 2-layer residual ReLU MLP, n=64,
manifold m=8 with variance 1, off-manifold variance lam_off, 32 outputs, 1024 samples, label noise 0.1, 3000 steps,
2 seeds): ratio of input-Jacobian gains G_off/G_on

| lam_off | SGD | AdamW | Muon | Muon, no orthogonalisation (blend 0) |
|---|---|---|---|---|
| 1e-2 | 0.21 | 0.29 | 0.72 | 0.76 |
| 1e-3 | 0.26 | 0.32 | 1.67 | 1.02 |
| 1e-4 | 0.27 | 0.31 | **3.97** | 0.82 |

and the fraction of a random readout's input gradient lying off the manifold (chance 0.875): SGD 0.23-0.35,
AdamW 0.37-0.46, Muon 0.74 -> 0.94 -> 0.99. Muon's off-manifold gain grows ~ lam_off^{-1/2} as predicted, GD/AdamW
stay flat, and removing the orthogonalisation removes most of the effect at small lam_off. **[checked numerically]**

Weight-level fingerprint on ELF-B (`spectra.py`): Muon fine-tuning updates have stable rank / n 0.18-0.24 (AdamW 0.03)
and leave flatter weight spectra than the init (0.143 vs init 0.117 vs AdamW 0.034). **[measured on ELF]**

## 3. Predictions to test on ELF (`offspan.py`, `cross_critic.py`, `attack_transfer.py`)

1. Critic input gradients put more mass on N for Muon trunks (own, sibling, independent) than for the AdamW trunk.
2. The Muon policy's Jacobian gain off/on is larger than the AdamW policy's.
3. Along collapsing RL runs the clean estimate drifts off the manifold; along protected runs it does not.
4. States reached by exploiting one Muon critic are over-scored by the other Muon critics, not by the AdamW critic.
5. E (measured at RL start) orders the final outcomes of all policy x critic pairings.
6. H-ELF: the coarse channels add tangent directions with large data variance (one-hot codes), which raises g_T and
   k_T, lowering E for the native hierarchical critic; a foreign (AdamW-hier) trunk breaks the code-token coupling. **[open]**

## 4. What the ELF measurements say (2026-10-30 evening) — revised picture

Measured (details and numbers in notes/results.md, 19:20–21:00 entries):

* **Optimizer fingerprint in the weights.** Muon updates and weights have flat, high-rank spectra; AdamW's are
  concentrated (stable rank/n 0.14 vs 0.03). Independent Muon models share the statistics. [measured]
* **Critic gradient geometry.** Critics on AdamW trunks have nearly the same input gradient whatever data their
  head saw (0.81–0.85 in input space, 0.94 in parameter space), along a direction that is steep (4–5× more logit per
  unit displacement than any Muon critic) and answer-preserving. Critics on Muon trunks have idiosyncratic,
  shallower gradients (0.09–0.38 agreement; the SAME Muon trunk with different head data 0.20). [measured]
* **First-order usefulness is the opposite of the naive "Muon gradients are junk" story.** One actor step at Λ/4 along
  a Muon critic raises accuracy by +4.7…+5.3 (3 of 4 critics), along an AdamW critic by −1.6…−0.8. [measured]
* **Exploitation of a frozen critic needs self-similarity.** Ascending the frozen native critic (trunk = the
  policy's own model) collapses accuracy 82 → 3% in 600 steps while the critic's AUC falls to 0.51; ascending the
  frozen independent-Muon critic does not (80–86%). The exploit does not transfer to any other critic. [measured]
* **Off-manifold drift is not the collapse.** The clean estimate's off-manifold energy does not rise in collapsing
  runs. [measured — falsifies prediction 3 of §3]

Revised mechanism (to be confirmed by the gradient-coherence and refit-churn runs):

1. *Self-similarity route.* A critic whose trunk is a copy of the policy can be satisfied by the policy changing its
   own computation: the actor moves the policy's denoiser, and the critic is the same denoiser one step later. The
   frozen own-trunk critic is exploited to 3% accuracy; no other trunk is fooled by those states. This is the
   original CFRL configuration (native critic).
2. *Refit route (any Muon trunk).* A non-self Muon critic is not exploitable while frozen, but collapses in RL where
   its head is refit on the policy's new states every iteration. Muon trunks give heads whose input gradients are
   set by the idiosyncratic noise fit of the trunk (whitening: §2), so each refit can point the actor in a new
   answer-changing direction (toy: Muon head-to-head gradient cosine 0.62–0.82 across refits vs 0.95–0.98).
3. *Why AdamW trunks protect.* Their critics share one steep, answer-preserving direction set by the trunk's
   dominant singular directions (spectral bias kept, §2). The actor's ascent is absorbed along it (large logit gains,
   answers almost unchanged: MA run changes 33% of answers, balanced), and refits do not redirect it.
4. *Policy side.* An AdamW-trained policy is safe even with a Muon critic: its output moves an order of magnitude
   less per actor step (|J^T u|^2 35–41 vs 614–629) and preferentially on-manifold.

## 5. Persistence and removal (toy, 2026-10-30)

* **GD cannot unlearn what Muon wrote.** The GD/AdamW gradient along an input direction scales with its variance
  lam_i, so the weight component Muon wrote along a low-variance direction decays at rate ~lam_i under any later
  GD-type fine-tuning: AdamW refinement for 3,000 steps leaves the off/on gain at 1.67. [checked numerically]
* **Weight decay removes it.** Decoupled decay shrinks all components at rate lr*wd; the loss gradient restores only
  the components the data support. Off/on gain 1.67 -> 1.55 (wd 0.1) -> 0.40 (wd 1) -> 0.13 (wd 1, 3x lr), and the
  pathwise loop with a refit head recovers (peak true reward +0.32 / +1.00 vs never improving). [checked numerically]
* **Refit loop toy.** With refit heads a critic on a Muon trunk never improves a free policy, even though the head is
  refit to the TRUE reward at every step; critics on SGD/AdamW trunks improve it, then over-optimise slowly; with
  frozen heads every trunk is exploited. Head-side invariance augmentation does not help. [checked numerically]

## 6. Synthesis of the mechanism analysis (draft 2026-10-31 02:30; final numbers in notes/results.md 19:20 10-30 → )

**What is established on ELF (Game24, flat model, seed-42 Muon policy):**

1. *Optimizer fingerprint.* Muon writes flat, high-rank updates (stable rank/n 0.18–0.24 vs AdamW 0.03) and leaves
   flat weight spectra (0.143 vs 0.034); every Muon model shares this, related or not.
2. *Two routes to collapse.*
   - **Self-similarity.** A critic whose trunk is the policy's own model is exploited even when frozen:
     600 steps of pure ascent take accuracy 82 → 3% while the critic's AUC falls to 0.51. The exploit is specific:
     no other critic (sibling, independent Muon, AdamW) is fooled by those states. Frozen exploitation is slower for a
     sibling Muon or an AdamW critic (82 → 66 / 45 at 600 steps) and absent for an independent Muon critic (80–86).
   - **The refit loop.** In RL the critic head is refit every iteration. For Muon trunks a refit rotates the
     critic's input gradient (cos 0.59–0.60 with the original after 200 refit steps) about twice as far as for
     AdamW trunks (0.84–0.85). With a frozen head the independent-Muon critic barely moves the policy (63.9 → 62.8 at
     it250) whereas with refits it collapses (56.0 at it250, final −11.4). The refit loop is what turns a non-self
     Muon critic into a collapse.
3. *Why AdamW trunks protect in the loop.* Their critics share one steep direction (gradient agreement 0.81–0.85
   across critics trained on different data; 0.94 in parameter space; logit gain 4–5× larger per unit move), which is
   answer-preserving (a single displacement along it changes ~1% of tokens; a frozen attack along it inflates the
   critic +5,900 with accuracy unchanged for 400 steps). Under refits the direction is kept, so the actor's
   normalised steps are coherent (gcos 0.93–0.95 vs native 0.46 early) and are spent along it. First-order, an
   AdamW critic step does NOT improve accuracy (−1.6/−0.8) while Muon critic steps do (+4.7…+5.3): protection is not
   a better signal but a stable, benign sink for the actor's step plus slow genuine improvement (best-val +3.5).
4. *Policy side.* An AdamW-trained policy moves its output ~15× less per actor step (|J^T u|^2 35–41 vs 614–629) and
   is safe even with a Muon critic.
5. *Negative results.* The collapsing policy does not drift off the data manifold (off-manifold energy flat);
   critic AUC on shifted states drops alike for all critics; a radial (latent-scale) explanation is ruled out
   (radial share ≤1%); no attack transfers between trunks.

**Theory (toys, `hcfrl/theory/`):** Muon's polar step is a rotation-equivariant whitening: it learns every input
direction at a rate independent of its variance, so it fits noise along low-variance directions (off/on gain
∝ lam_off^-1/2; AdamW/SGD flat; AdamW whitens too when the manifold is axis-aligned). A head on a Muon trunk can
fit the reward yet its input gradient is mostly off-manifold, weakly aligned with the true gradient (0.26–0.32 vs
0.84) and unstable across refits (0.62–0.82 vs 0.95–0.98); in a pathwise loop with refit heads a Muon trunk never
improves the policy while SGD/AdamW trunks do. GD-type fine-tuning cannot remove the Muon component (gain unchanged
after 3,000 AdamW steps); decoupled weight decay can (1.67 → 0.13) and restores the loop.

**Practical rule.** For pathwise RL, (i) never build the critic on the policy's own weights (self-similarity
route); (ii) build it on a trunk whose optimizer kept the spectral bias (AdamW/SGD, or with weight decay strong
enough to erase low-variance components), because the refit loop needs a gradient direction that refits do not
rotate. Head-side remedies (more data, more refits, sharpness-aware objectives, invariance augmentation) do not fix a
Muon trunk.

**Hierarchy.** The H-ELF critic built on the policy's own model is NOT exploited by 600 steps of frozen ascent
(82.8 → 85.5; flat: 82 → 3): the jointly denoised coarse code blocks the self-similarity route. Its failure with a
foreign AdamW-hier critic (late collapse 2/2) is therefore a refit-route or coupling effect, not self-similarity.

**Open:** (a) the ELF trunk-refinement test: AdamW-refined (no wd) 59.2 / 57.4 at it250/500 — partial, as predicted;
weight-decay-refined 64.2 / 63.2 — no collapse so far but confounded (weaker critic, trunk less self-similar, spectrum
only 15% shrunk); needs it1500 and more seeds. (b) a closed-form account of the self-similarity route. (c) why the
frozen independent-Muon critic is not exploited at all within 600 steps (its first-order step also helps least, +1.4).
(d) ELF-M replication.
