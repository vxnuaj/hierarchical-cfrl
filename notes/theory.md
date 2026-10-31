# Theory: why pathwise RL collapses when the critic is built on a Muon-trained model

Status: written 2026-10-31 (replaces the working notes of 2026-10-30, kept as `notes/theory_2026-10-31_draft.md`).
Every result is tagged **[proved]** (complete proof below, in the stated model), **[proved, linear model]**,
**[verified numerically]** (closed form checked by `hcfrl/theory/verify_theorems.py`, log `runs/theory/verify_theorems.log`),
**[measured]** (on ELF; numbers and job ids in `notes/results.md`), **[conjecture]** or **[falsified]**.
Section 9 maps every statement to its evidence; section 10 lists what the theory does not explain.

---

## 0. Setting and notation

**Policy and sampler.** The flow policy is a denoiser $\mathrm d_\theta$. A rollout is a sequence of states
$s_k = (\tilde z_k, \tilde t_k, \hat x_{k-1})$; the clean estimate produced at step $k-1$ is
$$\hat x_{k-1} = \mathrm d_\theta(s_{k-1}) \in \mathbb R^{L\times D},$$
and the next state is an affine function of it (advance and churn):
$$s_k = \Phi_k(\hat x_{k-1}) ,\qquad \frac{\partial \tilde z_k}{\partial \hat x_{k-1}} = \alpha\, a_k\, I,\qquad \frac{\partial \hat x^{\rm sc}_k}{\partial \hat x_{k-1}} = I ,$$
with $a_k = (t_k-\tilde t_{k-1})/\max(1-\tilde t_{k-1},t_\epsilon)$ and $\hat x^{\rm sc}_k$ the self-conditioning input of state $k$.

**Critic.** $\ell_\phi(s) = f\big(h(s)\big)$ with trunk $h$ (a copy of some denoiser's hidden layers) and head $f$.
Its input gradient with respect to the clean estimate, through $\Phi_k$, is the *critic pull*
$$G_k = \frac{\partial\, \ell_\phi(\Phi_k(\hat x))}{\partial \hat x}\Big|_{\hat x=\hat x_{k-1}} \in \mathbb R^{L\times D}.$$

**Actor step.** CFRL ascends the critic through one denoiser call. With $J_\theta = \partial \hat x_{k-1}/\partial\theta$,
$$\nabla_\theta\, \ell_\phi(s_k) = J_\theta^\top G_k , \qquad \delta\theta = \eta\, \mathcal P\big(J_\theta^\top G_k\big),$$
where $\mathcal P$ is the actor optimizer's direction map (AdamW with gradient clipping in all our runs).

**Data manifold.** $T\subset\mathbb R^{D}$ is the span of the principal directions of the clean latents of training
solutions holding a fraction $1-\epsilon$ of their variance, $N = T^\perp$, and $P_T, P_N$ the orthogonal projections
(applied per position). On ELF, $\dim T = 117$ of $D=512$ at $1-\epsilon = 0.99$ **[measured]**.

**Optimizers as steepest descent.** For a matrix parameter $W$ with gradient $G$:
- gradient descent: $\Delta W = -\eta G$ (steepest descent in $\|\cdot\|_F$);
- sign descent (AdamW with $\beta_1=\beta_2=0$): $\Delta W = -\eta\,\mathrm{sign}(G)$ (steepest descent in the entrywise $\ell_\infty$ norm) [Balles & Hennig 2018; Bernstein & Newhouse 2024];
- Muon without momentum: $\Delta W = -\eta\,\mathrm{polar}(G)$, where $\mathrm{polar}(G) = UV^\top$ for the compact SVD $G = U\Sigma V^\top$ (steepest descent in the spectral norm) [Bernstein & Newhouse 2024; Jordan et al. 2024].

---

## 1. First-order analysis of one actor step

**Lemma 1.1 (first-order effect of an actor step).** Let $u = \mathcal P(J_\theta^\top G)$ and $\delta\theta = \eta u$. Then the
clean estimate, the critic and any differentiable reward functional $R(\hat x)$ change by
$$\delta\hat x = \eta J_\theta u + O(\eta^2),\qquad \delta\ell = \eta \langle G, J_\theta u\rangle + O(\eta^2), \qquad \delta R = \eta\langle \nabla R, J_\theta u\rangle + O(\eta^2).$$
*Proof.* Taylor expansion of $\hat x(\theta+\eta u)$ and of $\ell\circ\Phi$ and $R$ around $\hat x$. $\square$ **[proved]**

For gradient descent ($\mathcal P = \mathrm{id}$) define the *output kernel* $K = J_\theta J_\theta^\top$, a PSD operator on
$\mathbb R^{L\times D}$ (the empirical NTK of one denoiser call). Then $\delta\ell = \eta\, G^\top K G$ and
$\delta R = \eta\,\nabla R^\top K G$.

**Proposition 1.2 (exploitation decomposition).** Write $G = G_T + G_N$ with $G_T = P_T G$, and assume, as an
idealisation of a verifier on well-formed outputs, that the reward responds only to on-manifold movement to first
order, $\nabla R = P_T\nabla R$. Then, to first order and under gradient descent,
$$\delta\ell = \eta\big(G_T^\top K_{TT}G_T + 2G_T^\top K_{TN}G_N + G_N^\top K_{NN}G_N\big),\qquad \delta R = \eta\big(\nabla R^\top K_{TT} G_T + \nabla R^\top K_{TN}G_N\big),$$
with $K_{AB} = P_A K P_B$. The term $G_N^\top K_{NN} G_N \ge 0$ raises the critic and contributes nothing to $\delta R$.
*Proof.* Substitute $G = G_T+G_N$ and $\nabla R = P_T\nabla R$ into Lemma 1.1. $\square$ **[proved]**

**Corollary 1.3 (exploitation index and its factorisation).** Define
$$\mathcal E = \frac{G_N^\top K_{NN} G_N}{G_T^\top K_{TT}G_T}.$$
With $k_N^{\max} = \lambda_{\max}(K_{NN})$ and $k_T^{\min}$ the smallest eigenvalue of $K_{TT}$ restricted to $T$ (assumed $>0$),
$$\mathcal E \;\le\; \frac{\|G_N\|^2}{\|G_T\|^2}\cdot\frac{k^{\max}_N}{k^{\min}_T}.$$
*Proof.* Rayleigh-quotient bounds on numerator and denominator. $\square$ **[proved]**

The bound is a product of a *critic term* (the off-manifold share of the pull) and a *policy term* (how cheaply the
policy moves off the manifold). The policy × critic interaction of the swap experiments — only a Muon policy with a
Muon critic collapses — is the statement that both factors must be large. On ELF the critic term is modestly larger
for Muon trunks (off-manifold share of $\|G\|^2$: 0.39–0.42 Muon, 0.33 AdamW, 0.24 AdamW+wd), and the Muon
policy's output moves an order of magnitude more per unit parameter step ($\|J^\top u\|^2$ on-manifold: 614–629 vs
35–41) **[measured]**. The first-order picture alone does not predict the collapse: one actor step along a Muon
critic *raises* accuracy by +4.7…+5.3 **[measured]**. The collapse is a property of the iterated loop (sections 4–5).

---

## 2. Geometry of the three update rules

**Lemma 2.1 (rotation equivariance of the polar update).** For orthogonal $R\in O(d)$, $Q\in O(n)$ and any $G$,
$$\mathrm{polar}(RGQ) = R\,\mathrm{polar}(G)\,Q.$$
*Proof.* If $G = U\Sigma V^\top$ then $RGQ = (RU)\Sigma(Q^\top V)^\top$ is a compact SVD, so
$\mathrm{polar}(RGQ) = RUV^\top Q$. $\square$ **[proved]**

**Lemma 2.2 (sign descent is equivariant only to signed permutations).** If $\mathrm{sign}(RGQ) = R\,\mathrm{sign}(G)\,Q$ for all
$G$, then $R$ and $Q$ are signed permutation matrices.
*Proof.* Take $G = e_ie_j^\top$. Then $\mathrm{sign}(G)=G$ and the condition reads $\mathrm{sign}(r_i q_j^\top) = r_iq_j^\top$, where $r_i$ is the
$i$-th column of $R$ and $q_j^\top$ the $j$-th row of $Q$. Every entry of the rank-one matrix $r_iq_j^\top$ must then lie in
$\{-1,0,1\}$. Since $\|r_i\| = \|q_j\| = 1$, the nonzero entries of $r_iq_j^\top$ have modulus $|r_{li}||q_{jm}|\le 1$ with equality
only if both factors have modulus one, so $r_i$ and $q_j$ each have exactly one nonzero entry, of modulus one. As this
holds for all $i,j$, $R$ and $Q$ are signed permutations. $\square$ **[proved]**

**Lemma 2.3 (descent rates).** Along the respective flows,
$$\dot L = -\|\nabla L\|_F^2\ \ \text{(GD)},\qquad \dot L = -\textstyle\sum_{ij}|(\nabla L)_{ij}|\ \ \text{(sign flow)},\qquad \dot L = -\|\nabla L\|_*\ \ \text{(polar flow)} .$$
*Proof.* $\dot L = \langle \nabla L,\dot W\rangle$; $\langle G,\mathrm{polar}(G)\rangle = \operatorname{tr}(V\Sigma U^\top UV^\top) = \sum_i\sigma_i = \|G\|_*$ and
$\langle G,\mathrm{sign}(G)\rangle = \sum_{ij}|G_{ij}|$. $\square$ **[proved]**

Interpretation. The polar step gives every singular direction of the gradient the same step length, *whatever its
singular value*, in a basis that rotates with the data (Lemma 2.1). Sign descent does the same only in the fixed
coordinate basis of the weights (Lemma 2.2). This is the sense in which Muon whitens: second-order optimisation of
a linear model equals first-order optimisation on whitened data [Wadia et al. 2021], and polar gradient descent has
been shown to act as an implicit, iterate-dependent whitening preconditioner [*A Quadratic Lens on Muon*, ICML 2026].
The implicit bias of spectral descent and Muon on separable data is the spectral-norm max-margin solution
[Fan, Schmidt & Thrampoulidis 2025]: again the (rotation-invariant) geometry, not the data covariance, decides which
directions are used.

---

## 3. Muon removes the spectral bias that keeps networks insensitive off the data manifold

**Model.** A linear layer $W\in\mathbb R^{d\times n}$ with $d\ge n$; inputs with (empirical) covariance
$\Sigma = \operatorname{diag}(\lambda_1,\dots,\lambda_n)\succ 0$ in a fixed orthonormal basis $\{e_i\}$; loss
$$L(W) = \tfrac12\operatorname{tr}\big((W-\bar W)\Sigma(W-\bar W)^\top\big),\qquad \nabla L(W) = (W-\bar W)\Sigma ,$$
with $\bar W$ the least-squares solution. The *sensitivity of the trained layer to input direction $e_i$* is $\|We_i\|$.

**Theorem 3.1 (gradient flow vs. polar descent).**
(a) Gradient flow $\dot W = -\nabla L$ from $W(0)=0$ gives, for every $\bar W$,
$$W(t)e_i = \big(1-e^{-\lambda_i t}\big)\,\bar We_i .$$
Discrete GD with step $\eta<1/\lambda_{\max}$ gives $W_k e_i = \big(1-(1-\eta\lambda_i)^k\big)\bar We_i$.

(b) Assume the columns of $\bar W$ are mutually orthogonal: $\bar We_i = c_iq_i$ with $c_i>0$ and $\{q_i\}$ orthonormal.
Then the polar flow $\dot W = -\mathrm{polar}(\nabla L)$ from $W(0)=0$ has the solution
$$W(t)e_i = \min\Big(\frac{t}{c_i},1\Big)\,\bar We_i ,$$
independent of $\lambda_i$, and discrete polar descent with step $\eta$ satisfies
$$\big\|W_ke_i - \bar We_i\big\| \le \max\big(c_i - k\eta,\ \eta\big)\qquad\text{for all } i,k.$$

*Proof.* (a) $D = W-\bar W$ obeys $\dot D = -D\Sigma$, so $D(t) = D(0)e^{-\Sigma t} = -\bar W e^{-\Sigma t}$; read off column $i$.
The discrete case is identical with $e^{-\lambda_i t}$ replaced by $(1-\eta\lambda_i)^k$.

(b) Make the ansatz $D(t) = -\sum_i c_i a_i(t)\,q_ie_i^\top$ with $a_i(0)=1$. Then
$$\nabla L = D\Sigma = -\sum_i c_i a_i\lambda_i\, q_ie_i^\top .$$
Because $\{q_i\}$ and $\{e_i\}$ are orthonormal families, this is a singular value decomposition with singular values
$c_i|a_i|\lambda_i$, left vectors $-\mathrm{sign}(a_i)q_i$ and right vectors $e_i$; therefore
$$\mathrm{polar}(\nabla L) = -\sum_{i:\,a_i\neq 0}\mathrm{sign}(a_i)\,q_ie_i^\top ,\qquad \dot D = -\mathrm{polar}(D\Sigma) = \sum_{i:\,a_i\ne0}\mathrm{sign}(a_i)\, q_ie_i^\top ,$$
i.e. $c_i\dot a_i = -\mathrm{sign}(a_i)$ while $a_i\neq0$. Hence $a_i(t) = \max(0,1-t/c_i)$: the ansatz is preserved, and since the
right-hand side is discontinuous only at $a_i=0$, where the component stops, this is the Filippov solution. Then
$We_i = \bar We_i + De_i = c_i(1-a_i)q_i$. In discrete time $a_i^{k+1} = a_i^k - (\eta/c_i)\,\mathrm{sign}(a_i^k)$: $a_i$ decreases
linearly until it changes sign and then oscillates in $[-\eta/c_i,\eta/c_i]$, which gives the bound. $\square$
**[proved; verified numerically: GD columns match the formula to 4 decimals; polar residuals equal $c_i-k\eta$ or are $\le\eta$]**

*Remark (numerical).* In floating point, a column that lands exactly on its target produces a zero singular value;
round-off in it is then promoted by $\mathrm{polar}$ to a unit-length step and couples the columns (observed in the
verification when $c_i$ is a multiple of $\eta$). This instability is itself the whitening property: directions with
arbitrarily small gradient receive unit steps.

**Corollary 3.2 (label noise is fitted along low-variance directions).** Take a fixed design with $X^\top X/m = \Sigma$,
targets $Y = XW^{*\top} + E$ with $E$ i.i.d. $\mathcal N(0,s^2)$, and a direction $i$ that the target does not use
($W^*e_i = 0$). Then $\bar We_i \sim \mathcal N\big(0,\ \tfrac{s^2}{m\lambda_i}I_d\big)$ and, under the orthogonality assumption of
3.1(b) (which holds approximately when $d\gg n$, the noise columns being independent Gaussians),
$$\mathbb E\,\|W^{\rm GD}(t)e_i\|^2 = \big(1-e^{-\lambda_it}\big)^2\frac{d\, s^2}{m\lambda_i} \;\le\; \frac{t^2\,\lambda_i\, d\, s^2}{m}\;\xrightarrow[\lambda_i\to0]{}\;0,$$
$$\mathbb E\,\|W^{\rm polar}(t)e_i\|^2 = \mathbb E\,\min(t,c_i)^2,\qquad c_i^2\sim \frac{s^2}{m\lambda_i}\chi^2_d ,\qquad \xrightarrow[\lambda_i\to0]{}\; t^2 .$$
At any fixed training time the GD sensitivity along a direction of variance $\lambda$ vanishes like $\sqrt\lambda$, while the
polar sensitivity grows like $\min\big(t,\ s\sqrt{d/(m\lambda)}\big)$, i.e. like $\lambda^{-1/2}$ until it saturates at the
training horizon $t$.
*Proof.* $\bar W = Y^\top X(X^\top X)^{-1} = W^* + E^\top X\Sigma^{-1}/m$. Its column $i$ is $E^\top X e_i/(m\lambda_i)$ with
$\|Xe_i\|^2 = m\lambda_i$, a Gaussian vector with covariance $\tfrac{s^2}{m\lambda_i}I_d$. Insert into Theorem 3.1 and use
$1-e^{-x}\le x$. $\square$ **[proved, linear model]**

This is the classical spectral bias of gradient descent — directions are learned at rates proportional to their
variance, cf. the mode-by-mode dynamics of [Saxe, McClelland & Ganguli 2014] and the spectral bias of
[Rahaman et al. 2019] — and its removal by whitening [Wadia et al. 2021]. It differs from [Amari et al. 2021], who show
that preconditioning can *help* in-distribution generalisation under label noise: our quantity is sensitivity to
*off-distribution* directions, which preconditioning increases.

**Numerical check in a nonlinear network** (2-layer residual ReLU MLP, input dimension 64, manifold of dimension 8 and
variance 1, off-manifold variance $\lambda_{\rm off}$, 32 outputs, 3000 steps; `hcfrl/theory/whitening.py`). Input-Jacobian
gain off/on the manifold:

| $\lambda_{\rm off}$ | SGD | AdamW | Muon | Muon, no orthogonalisation |
|---|---|---|---|---|
| $10^{-2}$ | 0.21 | 0.29 | 0.72 | 0.76 |
| $10^{-3}$ | 0.26 | 0.32 | 1.67 | 1.02 |
| $10^{-4}$ | 0.27 | 0.31 | **3.97** | 0.82 |

The Muon column grows by about $2.3\times$ per decade of $\lambda_{\rm off}$, between the $\lambda^{-1/2}$ law ($3.2\times$) and saturation, as
Corollary 3.2 predicts; SGD and AdamW are flat. **[verified numerically]**

**Proposition 3.3 (AdamW whitens only in the weight basis).** In the model of Theorem 3.1 with $\Sigma$ diagonal in the
*parameter* basis, sign flow $\dot W = -\mathrm{sign}(\nabla L)$ from $W(0)=0$ gives
$$W_{ji}(t) = \mathrm{sign}(\bar W_{ji})\,\min\big(t,|\bar W_{ji}|\big),$$
independent of $\lambda_i$.
*Proof.* $(\nabla L)_{ji} = (W-\bar W)_{ji}\lambda_i$ has the sign of $(W-\bar W)_{ji}$, so every entry moves at unit speed to its
target and stops there. $\square$ **[proved]** For a rotated covariance $R\Lambda R^\top$ no such decoupling exists (Lemma 2.2)
and AdamW keeps the spectral bias. Prediction checked: with the manifold axis-aligned, AdamW's off/on gain rises
0.32 → 0.82 while SGD (0.26 → 0.27) and Muon (1.67 → 1.71) are unchanged (`hcfrl/theory/axis_check.py`)
**[verified numerically]**. The ELF latents (normalised T5 embeddings) are not axis-aligned.

**Theorem 3.4 (persistence under GD fine-tuning; removal by weight decay).** Start from any $W_0$ (e.g. a Muon solution)
and fine-tune on a loss with the same input covariance and least-squares target $\bar W'$.
(a) Gradient flow:
$$W(t)e_i = e^{-\lambda_it}\,W_0e_i + \big(1-e^{-\lambda_it}\big)\bar W'e_i .$$
The component written by Muon along direction $i$ decays at rate $\lambda_i$, i.e. hardly at all along low-variance directions.
(b) Gradient flow with decoupled weight decay $\mu>0$, $\dot W = -(W-\bar W')\Sigma - \mu W$:
$$W(t)e_i = e^{-(\lambda_i+\mu)t}\,W_0e_i + \big(1-e^{-(\lambda_i+\mu)t}\big)\frac{\lambda_i}{\lambda_i+\mu}\,\bar W'e_i ,$$
so the initial component decays at rate at least $\mu$ in every direction, and the limit is the ridge solution, whose
sensitivity along direction $i$ is shrunk by $\lambda_i/(\lambda_i+\mu)\to0$ as $\lambda_i\to0$.
*Proof.* The ODE $\dot W = -W(\Sigma+\mu I) + \bar W'\Sigma$ is linear and diagonal in $\{e_i\}$; solve column by column ($\mu=0$
gives (a)). $\square$ **[proved; verified numerically: residual fractions match $e^{-\lambda t}$ and $e^{-(\lambda+\mu)t}$ to the printed precision]**

Consequences. (i) Plain AdamW or GD fine-tuning of a Muon-trained critic trunk cannot remove its off-manifold
sensitivity (toy: off/on gain 1.67 before and after 3,000 AdamW steps). (ii) Weight decay can (toy: 1.67 → 1.55 /
0.40 / 0.13 for $\mu\eta k\approx 0.3,\ 3,\ 9$; the pathwise loop recovers, section 5). (iii) On ELF the weight-decayed
critic refinement reached only $\mu\eta k\approx 0.16$ (weights shrunk by ≈15%; stable rank/n 0.139 vs Muon 0.143 and
AdamW 0.034), far below the toy's effective range, so that experiment does not test the theorem.

---

## 4. Refitting a critic head on a whitened trunk: noisy, unstable, off-manifold gradients

The RL loop refits the critic head every iteration on fresh rollouts. Early-stopped gradient descent on least squares
closely tracks ridge regression [Ali, Kolter & Tibshirani 2019], so we analyse a ridge head on a (locally) linear trunk.

**Model.** Trunk features $\varphi(x) = Ax$ with $A\in\mathbb R^{p\times n}$; $m$ states with $X^\top X/m = \hat\Sigma$; rewards
$y = X\beta + \varepsilon$ with $\mathbb E\varepsilon = 0$ and $\mathrm{Cov}\,\varepsilon = s^2 I_m$ (Bernoulli rewards: $s^2 = \pi(1-\pi)$). The ridge head is
$$\hat w = \arg\min_w\ \tfrac1m\|XA^\top w-y\|^2 + \kappa\|w\|^2 = \big(A\hat\Sigma A^\top + \kappa I\big)^{-1}A\,X^\top y/m ,$$
the critic is $\ell(x) = \hat w^\top A x$ and its input gradient is $g = A^\top\hat w$. Define the PSD matrix
$$M = A^\top \big(A\hat\Sigma A^\top+\kappa I\big)^{-1}A \in\mathbb R^{n\times n}.$$

**Theorem 4.1 (law of the critic's input gradient under refits).** Conditionally on the states,
$$\mathbb E[g] = \bar g := M\hat\Sigma\beta,\qquad \mathrm{Cov}[g] = C := \frac{s^2}{m}\,M\hat\Sigma M .$$
For two refits on independent reward noise, $g$ and $g'$, the *second-moment stability* is
$$\mathcal S := \frac{\mathbb E\langle g,g'\rangle}{\mathbb E\|g\|^2} = \frac{\|\bar g\|^2}{\|\bar g\|^2 + \operatorname{tr} C},$$
and the alignment with the reward direction satisfies $\mathbb E\langle g,\beta\rangle = \langle\bar g,\beta\rangle$.
*Proof.* $g = A^\top(A\hat\Sigma A^\top+\kappa I)^{-1}AX^\top(X\beta+\varepsilon)/m = M\hat\Sigma\beta + MX^\top\varepsilon/m$. The second term has mean 0
and covariance $M X^\top \mathrm{Cov}(\varepsilon) X M/m^2 = \tfrac{s^2}{m}M\hat\Sigma M$. For independent $\varepsilon,\varepsilon'$,
$\mathbb E\langle g,g'\rangle = \|\bar g\|^2$ and $\mathbb E\|g\|^2 = \|\bar g\|^2+\operatorname{tr}C$. $\square$
**[proved; verified by Monte Carlo (4,000 refits): mean error ≤3% of $\|\bar g\|$, covariance relative error 3–8%, stability exact to 3 decimals]**

**Proposition 4.2 (spiked trunk).** Let $\hat\Sigma = \lambda_TP_T + \lambda_NP_N$ with $\dim T = n_T$, $\dim N = n_N$, let $\beta\in T$, and let the
trunk have gain $a_T$ on $T$ and $a_N$ on $N$ into orthogonal feature subspaces: $A = a_TU_T + a_NU_N$ with
$U_T^\top U_N = 0$, $U_T^\top U_T = P_T$, $U_N^\top U_N = P_N$. With
$$\mu_T = \frac{a_T^2}{a_T^2\lambda_T+\kappa},\qquad \mu_N = \frac{a_N^2}{a_N^2\lambda_N+\kappa},$$
we have $M = \mu_TP_T + \mu_NP_N$, $\bar g = \mu_T\lambda_T\,\beta \in T$, and
$$\operatorname{tr}C = \frac{s^2}{m}\Big(n_T\,\mu_T^2\lambda_T + n_N\,\nu(a_N)\Big),\qquad \nu(a_N) := \mu_N^2\lambda_N = \frac{a_N^4\,\lambda_N}{(a_N^2\lambda_N+\kappa)^2}.$$
The mean gradient has no off-manifold part, and the off-manifold share of the gradient energy is
$$F_N = \frac{\tfrac{s^2}{m}\,n_N\,\nu(a_N)}{\|\bar g\|^2+\operatorname{tr}C}.$$
*Proof.* $A\hat\Sigma A^\top + \kappa I$ acts as $a_T^2\lambda_T+\kappa$ on $\mathrm{range}\,U_T$, as $a_N^2\lambda_N+\kappa$ on $\mathrm{range}\,U_N$, and as $\kappa$ on the
orthogonal complement; conjugating by $A$ gives $M$. Substitute into Theorem 4.1. $\square$ **[proved]**

**Corollary 4.3 (monotonicity in the trunk's off-manifold gain).** $\nu$ is strictly increasing in $a_N>0$:
$$\frac{d\nu}{d(a_N^2)} = \frac{2\kappa\,\lambda_N\,a_N^2}{(a_N^2\lambda_N+\kappa)^3} > 0,\qquad \sup_{a_N}\nu = \frac1{\lambda_N}.$$
Hence the refit stability $\mathcal S$ and the normalised alignment $\|\bar g\|/\sqrt{\mathbb E\|g\|^2} = \sqrt{\mathcal S}$ decrease, and the
off-manifold share $F_N$ increases, monotonically in $a_N$ — with every on-manifold quantity ($a_T,\lambda_T,\beta$, hence the
critic's predictions on the data) held fixed.
*Proof.* For $\nu(u) = \lambda u^2/(\lambda u+\kappa)^2$ with $u = a_N^2$,
$\nu'(u) = \frac{2\lambda u(\lambda u+\kappa) - 2\lambda^2u^2}{(\lambda u+\kappa)^3} = \frac{2\kappa\lambda u}{(\lambda u+\kappa)^3}$. $\square$
**[proved; Monte Carlo: $a_N = 0.1, 1, 10$ gives $\mathcal S$ = 0.999, 0.993, 0.684 and $F_N$ = 0.000, 0.006, 0.315, matching the formulas to 3 decimals]**

**Combining sections 3 and 4.** Theorem 3.1 and Corollary 3.2 say that a Muon-trained trunk has a large off-manifold
gain $a_N$ (growing like $\lambda_N^{-1/2}$) and a GD/AdamW-trained trunk a small one (shrinking like $\lambda_N^{1/2}$).
Corollary 4.3 then says that a critic head refit on the Muon trunk has an input gradient that is (i) just as accurate on
the data (same $\bar g$), but (ii) dominated by off-manifold noise and (iii) different after every refit. Two critics of
equal held-out accuracy can therefore have very different gradients — the regime of the swap experiments, where
critic accuracy was matched (field score 0.82–0.88) but outcomes differed by +14 points.

Empirical counterparts **[measured]**: refitting the head on new states rotates the input gradient of Muon-trunk
critics about twice as far as that of AdamW-trunk critics (cosine to the original 0.60/0.59 vs 0.84/0.85); critics with
the same Muon trunk but different head data agree at 0.20, while AdamW-trunk critics trained on three different
datasets agree at 0.81–0.85. Toy ridge heads on trained MLP trunks (`head_churn.py`): head-to-head cosine 0.62 /
0.82 / 0.94 (Muon, $m$ = 128 / 512 / 2048) vs 0.95 / 0.98 / 0.99 (AdamW); on-manifold share 0.08–0.11 vs 0.73;
cosine with the true gradient 0.26–0.32 vs 0.84.

---

## 5. Dynamics of the actor–critic loop

**Model.** Output state $x_t\in\mathbb R^n$ (the policy's clean estimate, moved directly — the "free policy" limit $K\propto I$),
reward $R(x) = \langle\beta,x\rangle - \tfrac\gamma2\|P_Nx\|^2$ with $\beta\in T$ (off-manifold movement produces malformed answers),
normalised actor steps $x_{t+1} = x_t + \eta\,g_t/\|g_t\|$, and $x_0\in T$.

**Proposition 5.1 (a frozen critic is eventually exploited).** If the critic is frozen, $g_t\equiv g$ with unit direction $u$, then
$$R(x_t)-R(x_0) = \eta t\,\langle\beta,u\rangle - \tfrac\gamma2\,\eta^2t^2\,\|P_Nu\|^2 ,$$
which is maximal at $t^\star = \langle\beta,u\rangle\big/\big(\gamma\eta\|P_Nu\|^2\big)$ and negative for $t > 2t^\star$ whenever $P_Nu\ne0$. The time to
exploitation grows with the critic's alignment and shrinks with its off-manifold share.
*Proof.* $x_t = x_0+\eta tu$ and $P_Nx_0 = 0$. $\square$ **[proved]**

Counterparts: every frozen critic is exploited in the toy (reward −22…−36 by 300 steps), and on ELF the frozen self,
sibling and AdamW critics are exploited by 600 ascent steps (accuracy 3, 66, 45) **[measured]**. The frozen
independent-Muon critic is not (86) **[measured]**. Proposition 5.1 accommodates this only if its ratio
$\langle\beta,u\rangle/\|P_Nu\|^2$ is large; its first-order step is also the least useful (+1.4), so this case is unexplained (section 10).

**Proposition 5.2 (refits with independent noise are self-averaging).** Let $g_t = \bar g + \xi_t$ with $\bar g\in T$ and $\xi_t$ i.i.d. with
mean zero and $\mathbb E\|P_N\xi_t\|^2 = \sigma_N^2$, and suppose $\|g_t\|\approx\bar\gamma$ is approximately constant (small noise relative to the
mean, or a clipped step). Then
$$\mathbb E\,R(x_t)-R(x_0) \approx \frac{\eta t}{\bar\gamma}\,\langle\beta,\bar g\rangle - \frac\gamma2\,\frac{\eta^2}{\bar\gamma^2}\,t\,\sigma_N^2 ,$$
so the off-manifold excursion grows diffusively ($\propto t$ in squared norm), not ballistically ($\propto t^2$) as for a frozen critic.
*Proof.* With constant normalisation, $x_t - x_0 = \tfrac{\eta}{\bar\gamma}\sum_{s<t}(\bar g+\xi_s)$; then
$\mathbb E\|P_N(x_t-x_0)\|^2 = (\eta/\bar\gamma)^2\sum_{s<t}\mathbb E\|P_N\xi_s\|^2$ by independence and $P_N\bar g=0$, and
$\mathbb E\langle\beta,x_t-x_0\rangle = \tfrac{\eta t}{\bar\gamma}\langle\beta,\bar g\rangle$ because $\beta\in T$. $\square$ **[proved, under the stated approximation]**

**This model is falsified for Muon critics.** Proposition 5.2 predicts that refits make a noisy critic *safer* than a frozen
one. On ELF the opposite holds for the independent-Muon critic: frozen head, accuracy 63.9 → 64.2 at it750; refit head,
63.9 → 50.7 (final −11.4) **[measured]**. Hence the refit noise is *not* independent of the policy's state.
**[falsified: i.i.d.-refit model]**

**Proposition 5.3 (state-dependent refits can make the loop unstable).** Suppose instead that refitting on the states the
policy currently produces gives
$$g_t = \bar g + \xi_t + \Gamma\,\delta_t,\qquad \delta_t := P_N(x_t-x_0),$$
with $\Gamma$ symmetric PSD on $N$ (the refit head assigns higher value further along the direction the policy has already
moved). With constant normalisation $\bar\gamma$ the displacement obeys
$$\delta_{t+1} = \Big(I+\tfrac{\eta}{\bar\gamma}\Gamma\Big)\delta_t + \tfrac{\eta}{\bar\gamma}P_N\xi_t ,$$
so $\mathbb E\|\delta_t\|^2 \ge \big(\tfrac{\eta}{\bar\gamma}\big)^2\sigma_N^2\sum_{s<t}\big(1+\tfrac\eta{\bar\gamma}\lambda_{\min}(\Gamma)\big)^{2s}$ grows geometrically when $\lambda_{\min}(\Gamma)>0$, and the
reward loss $\tfrac\gamma2\|\delta_t\|^2$ eventually dominates any linear on-manifold gain.
*Proof.* Unroll the linear recursion; the noise terms are independent with zero mean, so cross terms vanish, and
$\|(I+c\Gamma)^s v\|\ge(1+c\lambda_{\min}(\Gamma))^s\|v\|$ for $c\ge0$. $\square$ **[proved, given the feedback assumption]**

Whether refits on Muon trunks produce $\Gamma\succ0$ is a **[conjecture]**. Supporting facts: refits on the states of a collapsing
policy rotate the Muon critic's gradient (cosine 0.60) **[measured]**; the iteration-to-iteration coherence of the actor
gradient falls as the native collapse unfolds (0.80 → 0.44) **[measured]**; and in the toy loop a whitened trunk never
improves the policy although its head is refit to the *true* reward at every step, while GD/AdamW trunks do (reward
peak +0.58 vs never above the start −0.19) **[verified numerically, `loop_toy.py`]**. Against a literal reading: the
off-manifold energy of the clean estimate does not grow in collapsing ELF runs **[measured]**, so if the feedback exists it
acts on answer-changing directions inside $T$ rather than on the PCA complement $N$.

**Proposition 5.4 (a benign sink stabilises the loop).** Let the pathwise gradient decompose as $J^\top G = J^\top b + J^\top G_{\rm rest}$, where $b$
is *answer-preserving* — $\langle\nabla R, J\,v\rangle = 0$ for every $v$ in the span of the updates it induces — and suppose the
actor step is normalised, $\delta\theta = \eta\,J^\top G/\|J^\top G\|$. Then the reward-relevant movement per step satisfies
$$|\delta R| \le \eta\,\|J^\top\nabla R\|\,\frac{\|J^\top G_{\rm rest}\|}{\|J^\top G\|} + O(\eta^2),$$
so a steep sink ($\|J^\top b\|\gg\|J^\top G_{\rm rest}\|$) reduces both the useful and the harmful movement per iteration by the factor
$\|J^\top G_{\rm rest}\|/\|J^\top G\|$.
*Proof.* $\delta R = \eta\langle J^\top\nabla R, J^\top G\rangle/\|J^\top G\| + O(\eta^2)$, the $b$ part contributes zero, and Cauchy–Schwarz bounds the rest. $\square$ **[proved]**

Counterparts **[measured]**: AdamW-trunk critics share one steep direction (agreement 0.81–0.85 in input space, 0.94 in
parameter space; 4–5× the logit gain of Muon critics per unit displacement; a displacement along it changes ≈1% of
decoded tokens); the pathwise gradient under the AdamW critic is ≈12× larger and 0.93–0.95 coherent between
iterations; a frozen attack along it inflates the critic by +5,900 while accuracy stays at 80–82 for 400 steps. This is
the "pressure valve": the AdamW critic does not give better first-order advice (−1.6 vs +4.7…+5.3 per step) but absorbs
most of each normalised step in an answer-preserving direction, and refits keep that direction (Theorem 4.1 with small
$a_N$, i.e. high $\mathcal S$).

---

## 6. Self-similarity

**Proposition 6.1 (alignment gain of a self-similar critic, linear model).** Let the critic gradient be $g = A_c^\top w$ with an
isotropic random head $w\sim\mathcal N(0,I_p/p)$, let the policy's output kernel be $K$ (section 1), and write $B = A_c^\top A_c$
(PSD, $n\times n$). Then $\mathbb E_w\, g^\top Kg = \tfrac1p\operatorname{tr}(BK)$. If the critic is built on the policy's own representation, $K = B$
("self"); if it is built on an unrelated representation with the same spectrum, $K = RBR^\top$ with $R$ Haar-distributed
on $O(n)$ ("independent"). Then
$$\frac{\mathbb E_w\, g^\top Kg\,\big|_{\rm self}}{\mathbb E_R\,\mathbb E_w\, g^\top Kg\,\big|_{\rm indep}} = \frac{n\operatorname{tr}(B^2)}{(\operatorname{tr}B)^2} = \frac{n}{\mathrm{PR}(B)}\ \ge 1,$$
where $\mathrm{PR}(B) = (\operatorname{tr}B)^2/\operatorname{tr}(B^2)\in[1,n]$ is the participation ratio; equality holds iff the spectrum of $B$ is flat.
*Proof.* $\mathbb E_w\,ww^\top = I/p$ gives the first identity. By Haar invariance $\mathbb E_R\,RBR^\top = \tfrac{\operatorname{tr}B}{n}I$, so
$\mathbb E_R\operatorname{tr}(BRBR^\top) = (\operatorname{tr}B)^2/n$, while the self case gives $\operatorname{tr}(B^2)$. Finally $\operatorname{tr}(B^2) = \sum\sigma_i^2 \ge (\sum\sigma_i)^2/n$ by
Cauchy–Schwarz on the eigenvalues, with equality iff they are all equal. $\square$
**[proved; verified: flat spectrum ratio 1.00; spiky spectrum (3 eigenvalues 100, 27 of 0.09) ratio 9.84 = $n/\mathrm{PR}$]**

Reading. In a *linear* model, self-similarity amplifies the critic gain the policy can obtain by $n/\mathrm{PR}$, close to 1 for
Muon's flat spectra and large for AdamW's concentrated ones — and for AdamW the amplified directions are the
dominant data directions, i.e. the benign sink of Proposition 5.4. This does **not** explain the strongest
self-similarity effect we measured: the frozen critic on the Muon policy's own model is exploited to 3% accuracy in
600 steps, whereas critics on other Muon models are exploited slowly (sibling, 66%) or not at all (independent, 86%)
**[measured]**. The self-similarity route is therefore nonlinear. Working hypothesis **[conjecture]**: the native
critic's trunk *is* the policy's next denoiser call, so its features encode the policy's self-consistency between
$\tilde z_k$ and $\hat x_{k-1}$; success correlates with self-consistency on the base policy, and the policy can raise its
self-consistency directly — a gradient-based analogue of a model's preference for its own generations
[Panickssery et al. 2024]. A test (not yet run): regress the native critic's logit on the denoiser's own residual
$\|\mathrm d_{\theta_0}(s_k)-\hat x_{k-1}\|$ before and after the frozen attack.

**Hierarchy.** The H-ELF critic built on its own policy is *not* exploited by 600 frozen ascent steps (82.8 → 85.5;
flat: 82 → 3) **[measured]**: the jointly denoised coarse code blocks the self-similarity route. A mechanism
compatible with the conjecture above: in H-ELF, self-consistency includes code–token agreement, which the policy
cannot raise without matching tokens to their classes, i.e. without moving towards well-formed answers **[conjecture]**.

---

## 7. Summary of the mechanism

1. *Optimizer fingerprint.* Polar (Muon) updates give every singular direction the same step in a basis that rotates
   with the data (Lemmas 2.1–2.3), so Muon-trained layers become sensitive to low-variance input directions
   (Theorem 3.1, Corollary 3.2: sensitivity $\propto\lambda^{-1/2}$ vs $\propto\lambda^{1/2}$ for GD). AdamW does not (Lemma 2.2,
   Proposition 3.3) unless the data are axis-aligned. On ELF, Muon updates have stable rank/n 0.18–0.24 vs 0.03.
2. *Critic gradients.* A head refit on a trunk with large off-manifold gain has an input gradient whose noise is
   off-manifold and whose direction changes with every refit, at unchanged accuracy (Theorem 4.1, Corollary 4.3).
3. *Loop.* Frozen critics are eventually exploited (Proposition 5.1). With refits, a stable, steep, benign direction
   (AdamW trunks) absorbs the actor's normalised steps (Proposition 5.4); a refit gradient that rotates and depends on
   the policy's state can destabilise the loop (Proposition 5.3, conditional). Independent refit noise alone would be
   harmless (Proposition 5.2), which the data falsify for Muon critics.
4. *Self-similarity* is a second, nonlinear route (frozen own-trunk critic → 3%), blocked by the hierarchy.
5. *Persistence and remedy.* GD/AdamW fine-tuning cannot remove the Muon component; weight decay can (Theorem 3.4).

**Practical rule.** For pathwise RL: never use a critic trunk that is the policy's own model; use a trunk whose optimizer
kept the spectral bias (AdamW/SGD), or remove the low-variance component with weight decay strong enough that
$\mu\eta k\gg1$. Head-side remedies (more data, more refits, sharpness-aware objectives, invariance augmentation) cannot fix
a whitened trunk: Corollary 4.3 holds at fixed on-manifold accuracy, and the toy augmentation and every ELF head-side
change failed.

---

## 8. Connections to prior theory

- **Steepest descent in operator norms.** Muon and Shampoo without accumulation are steepest descent in the spectral
  norm, Adam in the $\ell_\infty$ norm [Bernstein & Newhouse 2024]. Lemmas 2.1–2.3 are the elementary consequences used here.
- **Implicit bias of spectral descent.** Normalised steepest descent and Muon converge to the spectral-norm max-margin
  classifier on separable data, Adam to the $\ell_\infty$ max-margin one [Fan, Schmidt & Thrampoulidis 2025]. The geometry,
  not the data covariance, selects which directions are used — consistent with Theorem 3.1.
- **Whitening makes data information unusable.** Second-order optimisation of a linear model equals first-order
  optimisation of whitened data, and whitening can prevent generalisation [Wadia et al. 2021]. Polar GD acts as an
  implicit iterate-dependent whitening preconditioner [*A Quadratic Lens on Muon*, ICML 2026]. Theorem 3.1 is the
  matrix-polar version of this statement for the quantity that matters for pathwise RL: off-manifold sensitivity.
- **Preconditioning and generalisation.** [Amari et al. 2021] give exact bias–variance results for preconditioned vs.
  plain GD in ridgeless regression; preconditioning helps under label noise or misaligned signal *in distribution*. Our
  results concern sensitivity *off* the data distribution, where preconditioning hurts; the two are compatible.
- **Spectral bias and mode-by-mode learning.** GD learns input modes at rates set by their variance
  [Saxe et al. 2014; Rahaman et al. 2019]; Theorem 3.1(a) is the linear instance.
- **Early stopping ≈ ridge.** Early-stopped GD on least squares tracks ridge regression [Ali, Kolter & Tibshirani 2019],
  justifying the ridge-head model of section 4 for a head refit with a few gradient steps.
- **Non-robust features and transfer.** Adversarial directions as predictive but brittle features [Ilyas et al. 2019].
  Unlike adversarial examples in vision, our exploits do not transfer between critics (section 9), consistent with
  Theorem 4.1: the exploitable part of a Muon critic's gradient is its model-specific noise term, not a shared feature.
- **Reward over-optimisation and ensembles.** Over-optimisation scales predictably [Gao et al. 2023]; ensembles help more
  when members differ in pretraining [Eisenstein et al. 2023]. Corollary 4.3 suggests a sharper criterion: what matters
  is the trunk's off-manifold gain, which a different pretraining seed alone does not change if the optimizer is the same.
- **Self-preference.** Models evaluating their own outputs favour them [Panickssery et al. 2024]; the self-similarity
  route of section 6 is a gradient-based analogue.

---

## 9. Evidence map

| statement | evidence | status |
|---|---|---|
| Thm 3.1, Cor 3.2 (GD vs polar sensitivity) | closed forms exact (`verify_theorems`); MLP: off/on gain 0.72 / 1.67 / 3.97 Muon vs 0.21–0.32 SGD/AdamW | proved (linear) + verified |
| Prop 3.3 (AdamW whitens only axis-aligned) | AdamW 0.32 → 0.82 when axis-aligned; SGD, Muon unchanged | proved + verified |
| Thm 3.4 (persistence; weight decay) | residuals match $e^{-\lambda t}$, $e^{-(\lambda+\mu)t}$; toy: gain 1.67 unchanged by AdamW, 0.13 with weight decay | proved + verified |
| Muon fingerprint on ELF | update stable rank/n 0.18–0.24 vs 0.03; weights 0.143 vs 0.034 | measured |
| Thm 4.1, Cor 4.3 (refit gradients) | Monte Carlo exact; ELF refit rotation 0.60 vs 0.84; agreement 0.20 (same Muon trunk) vs 0.81–0.85 | proved + measured |
| Cor 1.3 factors | ELF critic off-share 0.39–0.42 vs 0.33 vs 0.24; policy Jacobian 15× | measured (critic factor modest) |
| Prop 5.1 (frozen critics exploited) | toy, all trunks; ELF self / sibling / AdamW exploited by 600 steps | proved + measured; independent Muon not exploited (unexplained) |
| Prop 5.2 (i.i.d. refits harmless) | frozen-head MM7 64.2 vs refit 50.7 at it750 | **falsified** for Muon critics |
| Prop 5.3 (feedback instability) | refit rotation; coherence decay; toy loop | conjecture (premise unproven) |
| Prop 5.4 (benign sink) | AdamW critics: steep shared answer-preserving direction; coherence 0.95; frozen attack +5,900 at flat accuracy | proved + measured |
| Prop 6.1 (linear self-alignment) | verified $n/\mathrm{PR}$ | proved; does not explain the measured self-similarity route |
| Self-similarity route | frozen own critic 82 → 3%, no transfer; H-ELF own critic 85.5 | measured; mechanism conjectural |
| Off-manifold drift during collapse | not observed (off-manifold energy flat) | **falsified** |
| Attack transfer between critics | none (other critics score exploited states lower) | measured; consistent with Thm 4.1 |

---

## 10. What the theory does not (yet) explain

1. **The nonlinear self-similarity route** (section 6): why the policy's own frozen critic is exploited to 3% while sibling
   and independent Muon critics are not. Proposition 6.1 shows linear self-alignment cannot be the cause.
2. **The feedback term $\Gamma$** of Proposition 5.3: a derivation of state-dependent refit bias for whitened trunks, and why
   the feedback acts on answer-changing directions inside $T$ rather than on the PCA complement $N$.
3. **The frozen independent-Muon critic** is not exploited within 600 steps; Proposition 5.1 requires a favourable
   alignment/off-share ratio that its weak first-order step does not suggest.
4. **Depth and attention.** All proofs are for linear layers (sections 3–4) or a free policy (section 5); the transfer to
   transformers rests on the measured spectra and gradient statistics, not on a theorem.
5. **ELF test of the weight-decay remedy** is underpowered ($\mu\eta k\approx0.16$ vs the toy's 3–9). The weight-decayed critic
   has not collapsed by it500 (64.2 / 63.2) but is confounded (weaker critic, less self-similar trunk).

---

## References

- Ali, Kolter & Tibshirani (2019). A continuous-time view of early stopping for least squares regression. AISTATS.
- Amari, Ba, Grosse, Li, Nitanda, Suzuki, Wu & Xu (2021). When does preconditioning help or hurt generalization? ICLR.
- Balles & Hennig (2018). Dissecting Adam: the sign, magnitude and variance of stochastic gradients. ICML.
- Bernstein & Newhouse (2024). Old Optimizer, New Norm: An Anthology. arXiv:2409.20325 (NeurIPS 2024 OPT workshop).
- Eisenstein et al. (2023). Helping or herding? Reward model ensembles mitigate but do not eliminate reward hacking.
- Fan, Schmidt & Thrampoulidis (2025). Implicit bias of spectral descent and Muon on multiclass separable data. NeurIPS (arXiv:2502.04664).
- Gao, Schulman & Hilton (2023). Scaling laws for reward model overoptimization. ICML.
- Ilyas, Santurkar, Tsipras, Engstrom, Tran & Madry (2019). Adversarial examples are not bugs, they are features. NeurIPS.
- Jordan et al. (2024). Muon: an optimizer for hidden layers in neural networks. Technical report.
- Panickssery, Bowman & Feng (2024). LLM evaluators recognize and favor their own generations. NeurIPS.
- Rahaman et al. (2019). On the spectral bias of neural networks. ICML.
- Saxe, McClelland & Ganguli (2014). Exact solutions to the nonlinear dynamics of learning in deep linear neural networks. ICLR (arXiv:1312.6120).
- Wadia, Duckworth, Schoenholz, Dyer & Sohl-Dickstein (2021). Whitening and second order optimization both make information in the dataset unusable during training, and can reduce or prevent generalization. ICML.
- *A Quadratic Lens on Muon: Orthogonalization, Invariance, and Implicit Preconditioning* (2026). ICML.

---

## Appendix: code

Theory checks in `hcfrl/theory/`: `verify_theorems.py` (closed forms of Thm 3.1, Thm 3.4, Thm 4.1 / Cor 4.3,
Prop 6.1; log `runs/theory/verify_theorems.log`), `whitening.py` (Cor 3.2 in an MLP), `dose.py`, `axis_check.py`
(Prop 3.3), `readout_agreement.py`, `head_churn.py` (Thm 4.1 in an MLP), `loop_toy.py`, `loop_fix.py`,
`loop_refine.py` (section 5, Thm 3.4). ELF measurements: `hcfrl/analysis/{spectra,cross_critic,offspan,attack_transfer,
multi_onestep,pull_effects,refit_churn}.py`; offline summaries in `hcfrl/analysis/offline/`.

---

## 11. Remaining work (status 2026-10-31)

The theory is complete for the data currently available: every result that the existing measurements can support is
stated, proved in its model, and checked. The items below need new data or new mathematics.

**Needs a new (cheap) experiment**
- [ ] **T1. Self-consistency test of the self-similarity route (§6).** Regress the native critic's pooled logit on the
  denoiser's own residual $\|\mathrm d_{\theta_0}(s_k)-\hat x_{k-1}\|$ on base-policy states and on states after the frozen
  attack (it 0/200/400/600). Prediction: strong negative correlation that persists while accuracy collapses. ≈1 debug-GPU hour.
- [ ] **T2. Measure the feedback term $\Gamma$ (Prop. 5.3).** Refit the head on states of policies displaced along a fixed
  direction $v$ by increasing amounts $\epsilon$ and measure $\langle g_\epsilon - g_0, v\rangle/\epsilon$ for Muon vs AdamW trunks (sign and size
  of $\Gamma$ along answer-changing directions). Extends `refit_churn.py`. ≈1–2 debug-GPU hours.
- [ ] **T3. Frozen independent-Muon critic (§10.3).** Measure $\langle\beta,u\rangle$ and $\|P_Nu\|$ (Prop. 5.1) for each frozen critic
  using the one-step and offspan tools, to test whether the alignment/off-share ratio orders the frozen exploitation
  times (self < sibling ≈ AdamW < independent).
- [ ] **T4. Powered test of Theorem 3.4 on ELF.** Weight-decayed critic trunk with $\mu\eta k\gtrsim3$ (e.g. wd 1.0 for ~10× more
  steps, or wd 10 at lr 1e-4 with more critic data), then off-manifold gain (offspan) and RL, 3 seeds.

**Needs new mathematics**
- [ ] **M1. Remove the column-orthogonality assumption of Theorem 3.1(b)** (general $\bar W$): a lower bound on the polar
  flow's progress along low-variance directions, e.g. via the nuclear-norm descent identity (Lemma 2.3).
- [ ] **M2. Derive $\Gamma\succ0$ for whitened trunks** (state-dependent refit bias) from Theorem 4.1 with a state-dependent design $\hat\Sigma_t$.
- [ ] **M3. Nonlinear self-similarity model** (critic = next denoiser call): a fixed-point/self-consistency argument.
- [ ] **M4. Beyond linear layers:** random-matrix derivation of the off-manifold gain of deep Muon-trained networks, or a
  wide-network (NTK / mean-field) extension of Theorems 3.1 and 4.1.

**Depends on runs in progress**
- [ ] **R1.** Refined-trunk critics to it1500 (`game24cd_flatAT_svg2d_s0`, `game24cd_flatATwd_svg2d_s0`): update §3 consequence (iii) and §10.5.
- [ ] **R2.** ELF-M (3×) replication of the fingerprint (spectra) and of the gradient-agreement / refit-churn measurements, once
  its critics exist (GPU partition queue).
