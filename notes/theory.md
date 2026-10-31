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
