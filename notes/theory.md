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
*Assumptions.* $\mathrm d_\theta$, $\ell\circ\Phi$ and $R$ are $C^2$ (or have Lipschitz gradients) near the current point, and the input state
$s_{k-1}$ is held fixed with respect to $\theta$ (stop-gradient: one differentiable denoiser call, as in CFRL).
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

Caveats. Proposition 1.2 and Corollary 1.3 are statements for plain gradient steps ($\mathcal P=\mathrm{id}$); all our runs use AdamW,
for which $\delta\ell = \eta\langle J^\top G,\mathcal P(J^\top G)\rangle$ has no kernel form, so the measured quantities below are proxies for the two
factors, not instances of them. $\mathcal E$ requires $G_T\neq0$, and its denominator is the critic gain on $T$, not reward progress.

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
holds for all $i,j$, $R$ and $Q$ are signed permutations. Conversely, signed permutations commute with the entrywise sign,
so the characterisation is exact. $\square$ **[proved]**

**Lemma 2.3 (descent rates).** Along the respective flows,
$$\dot L = -\|\nabla L\|_F^2\ \ \text{(GD)},\qquad \dot L = -\textstyle\sum_{ij}|(\nabla L)_{ij}|\ \ \text{(sign flow)},\qquad \dot L = -\|\nabla L\|_*\ \ \text{(polar flow)} .$$
(the sign and polar flows understood in the Filippov sense, the identities holding for almost every $t$).
*Proof.* Where the velocity is given by the formulas, $\dot L = \langle \nabla L,\dot W\rangle$, $\langle G,\mathrm{polar}(G)\rangle = \operatorname{tr}(V\Sigma U^\top UV^\top) = \sum_i\sigma_i = \|G\|_*$
and $\langle G,\mathrm{sign}(G)\rangle = \sum_{ij}|G_{ij}|$. In general (rank-deficient $\nabla L$, or zero entries for sign flow) the right-hand
sides are discontinuous. They are bounded and measurable, so Filippov solutions exist. Every element of the Filippov set
of $W\mapsto-\mathrm{polar}(\nabla L(W))$ lies in $-\partial\|\cdot\|_*(\nabla L(W))$, because subdifferentials of a convex function are convex and
outer-semicontinuous, and $\langle G,Y\rangle = \|G\|_*$ for every $Y\in\partial\|G\|_*$ (positive homogeneity). Since $L\circ W$ is absolutely continuous,
$\dot L = -\|\nabla L\|_*$ for almost every $t$. The same argument with $\partial\|\cdot\|_1$ covers sign flow. $\square$ **[proved]**

Interpretation. The polar step gives every singular direction of the gradient the same step length, *whatever its
singular value*, in a basis that rotates with the data (Lemma 2.1). Sign descent does the same only in the fixed
coordinate basis of the weights (Lemma 2.2). This is the sense in which Muon whitens: second-order optimisation of
a linear model equals first-order optimisation on whitened data [Wadia et al. 2021]. The implicit bias of spectral descent
and Muon on separable data is the spectral-norm max-margin solution [Fan, Schmidt & Thrampoulidis 2025]: the norm defining
the descent geometry, rather than the plain Euclidean geometry of GD, determines which max-margin solution is selected.

---

## 3. Muon removes the spectral bias that keeps networks insensitive off the data manifold

**Model.** A linear layer $W\in\mathbb R^{d\times n}$ with $d\ge n$; inputs with (empirical) covariance
$\Sigma = \operatorname{diag}(\lambda_1,\dots,\lambda_n)\succ 0$ in a fixed orthonormal basis $\{e_i\}$; loss
$$L(W) = \tfrac12\operatorname{tr}\big((W-\bar W)\Sigma(W-\bar W)^\top\big),\qquad \nabla L(W) = (W-\bar W)\Sigma ,$$
with $\bar W$ the least-squares solution. The *sensitivity of the trained layer to input direction $e_i$* is $\|We_i\|$.

**Theorem 3.1 (gradient flow vs. polar descent).**
(a) Gradient flow $\dot W = -\nabla L$ from $W(0)=0$ gives, for every $\bar W$,
$$W(t)e_i = \big(1-e^{-\lambda_i t}\big)\,\bar We_i .$$
Discrete GD with any step $\eta$ gives $W_k e_i = \big(1-(1-\eta\lambda_i)^k\big)\bar We_i$ (convergent iff $\eta<2/\lambda_{\max}$).

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
right-hand side is discontinuous only at $a_i=0$, where the component stops, this is a Filippov solution. It is the
unique one: for two Filippov solutions $W_1,W_2$ with velocities $-Y_1,-Y_2$, $Y_j\in\partial\|\cdot\|_*(\nabla L(W_j))$, the inner product
$\langle A,B\rangle_\Sigma = \operatorname{tr}(A\Sigma B^\top)$ gives $\tfrac{d}{dt}\tfrac12\|W_1-W_2\|_\Sigma^2 = -\langle Y_1-Y_2,\nabla L(W_1)-\nabla L(W_2)\rangle\le0$ by monotonicity of
the subdifferential, since $\nabla L(W_1)-\nabla L(W_2) = (W_1-W_2)\Sigma$. Then
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
$$\text{(conditional on the orthogonality of 3.1(b))}\qquad \mathbb E\,\|W^{\rm polar}(t)e_i\|^2 = \mathbb E\,\min(t,c_i)^2,\qquad c_i^2\sim \frac{s^2}{m\lambda_i}\chi^2_d ,\qquad \xrightarrow[\lambda_i\to0]{}\; t^2 .$$
At any fixed training time the GD sensitivity along a direction of variance $\lambda$ vanishes like $\sqrt\lambda$, while the
polar sensitivity grows like $\min\big(t,\ s\sqrt{d/(m\lambda)}\big)$, i.e. like $\lambda^{-1/2}$ until it saturates at the
training horizon $t$.
*Proof.* $\bar W = Y^\top X(X^\top X)^{-1} = W^* + E^\top X\Sigma^{-1}/m$. Its column $i$ is $E^\top X e_i/(m\lambda_i)$ with
$\|Xe_i\|^2 = m\lambda_i$, a Gaussian vector with covariance $\tfrac{s^2}{m\lambda_i}I_d$. Insert into Theorem 3.1 and use
$1-e^{-x}\le x$. $\square$ **[GD half: proved, linear model. Polar half: proved conditional on the orthogonality assumption of 3.1(b),
which under this Gaussian noise model holds only approximately (with probability 0 exactly); a fully fitted solution, by any
interpolating method, has noise sensitivity $\|\bar We_i\|\sim s\sqrt{d/(m\lambda_i)}$ without that assumption.]** Note also that the
polar/GD ratio at fixed $t$ scales as $\lambda^{-1/2}$ only in the saturated regime $c_i>t$; once direction $i$ is fitted it scales as $1/(\lambda_it)$.

This is the classical spectral bias of gradient descent — directions are learned at rates proportional to their
variance, the classical eigenvalue-spread picture of linear least-squares learning [LeCun, Kanter & Solla 1991; Advani & Saxe
2020] — and its removal by whitening [Wadia et al. 2021]. It differs from [Amari et al. 2021], who show
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
target and stops there. $\square$ **[proved]** For a rotated covariance $R\Lambda R^\top$ this decoupling is lost (Lemma 2.2), but that
alone does not imply that AdamW keeps the spectral bias: as Theorem 3.5 shows, idealised sign *flow* also fits every direction in
finite time. Why AdamW keeps it in practice is an empirical statement here, with two candidate mechanisms that the theory does
not model: (i) in a rotated basis every weight coordinate mixes all input directions, so Adam's per-coordinate normaliser
$\sqrt{v}+\epsilon$ is set by the dominant (high-variance) gradient components and is nearly uniform across coordinates, making the
update approximately a rescaled GD step, whose per-direction rate is $\propto\lambda_i$; (ii) along directions whose gradient falls
below $\epsilon$, Adam reduces to GD with step $\eta/\epsilon$. Prediction checked: with the manifold axis-aligned, AdamW's off/on gain rises
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
gives (a)). $\square$ **[proved; verified numerically for $\bar W'=0$: residual fractions match the discrete-time factors
$(1-\eta\lambda)^k$ and $(1-\eta(\lambda+\mu))^k$ (the continuous $e^{-\lambda t}$, $e^{-(\lambda+\mu)t}$ differ from them by ≈4% at the tested step);
the ridge-shrinkage term with $\bar W'\neq0$ is not separately tested]** This is a gradient-flow result; the claim that *AdamW* fine-tuning
cannot remove the Muon component rests on the toy (below), not on the theorem.

**Theorem 3.5 (normalised flows fit every direction in finite time — this does *not* separate Muon from GD).** Let
$\bar W\in\mathbb R^{d\times n}$ be arbitrary and $\Sigma = \operatorname{diag}(\lambda_i)\succ0$. Let $\|\cdot\|$ be any norm on matrices whose dual norm
dominates the Frobenius norm, $\|G\|_\circ\ge\|G\|_F$ (nuclear norm: polar flow; entrywise $\ell_1$: sign flow; Frobenius: normalised GD
$\dot W = -\nabla L/\|\nabla L\|_F$). Along any Filippov solution of the corresponding normalised steepest-descent flow from $W(0)=0$,
$$\frac{d}{dt}\sqrt{L(t)} \;\le\; -\sqrt{\lambda_{\min}/2}\qquad\text{for almost every } t \text{ with } L(t)>0,$$
so $W(t) = \bar W$ for all $t\ge T := \sqrt{2L(0)/\lambda_{\min}}$.
*Proof.* As in Lemma 2.3, $\dot L = -\|\nabla L\|_\circ$ a.e. With $D = W-\bar W$, $\|\nabla L\|_\circ\ge\|D\Sigma\|_F \ge \sqrt{\lambda_{\min}}\,\|D\Sigma^{1/2}\|_F = \sqrt{2\lambda_{\min}L}$,
so $\tfrac{d}{dt}\sqrt L\le-\sqrt{\lambda_{\min}/2}$ wherever $L>0$. Integrating, $L$ reaches 0 by $T$; because $L\circ W$ is absolutely continuous and
non-increasing, it stays 0, and $L=0$ with $\Sigma\succ0$ forces $W=\bar W$. $\square$ **[proved; verified numerically: with generic non-orthogonal
$\bar W$, discrete polar descent fits every column by $1.2\,T$; an independent check (proof review, `scratchpad/review/check1.py`) found normalised GD
does the same and sign descent approaches it as $\eta\to0$]**

*Why this is not a separation.* The proof uses only that the dual norm dominates $\|\cdot\|_F$, so it holds for normalised GD, which
traces the same path as GD at a different speed. Comparing polar flow with *unnormalised* gradient flow at the same $t$ compares time
units, and is not invariant to rescaling the data ($\Sigma\to c\Sigma$ speeds GD by $c$ and leaves $T$ unchanged). An earlier version of this
theorem drew a Muon-vs-GD ratio $\ge 1/(\lambda_iT)$ from it; that conclusion is withdrawn.

**Corollary 3.5′ (the separation is per step under a stable step size).** With a step budget of $k$ steps:
(a) discrete GD needs $\eta<2/\lambda_{\max}$ to converge on the top direction, and then by Theorem 3.1(a) has fitted at most a fraction
$$1-(1-\eta\lambda_i)^k \;\le\; k\eta\lambda_i \;<\; 2k\,\frac{\lambda_i}{\lambda_{\max}}$$
of direction $i$ — a bound that depends only on the condition ratio $\lambda_i/\lambda_{\max}$ and is invariant to rescaling the data;
(b) discrete polar descent with step $\eta$ fits direction $i$ to within $\max(c_i-k\eta,\eta)$ (Theorem 3.1(b)), independently of $\lambda_i$, under the
orthogonality assumption of 3.1(b).
So with $k\ll\lambda_{\max}/\lambda_i$ steps, GD has fitted almost nothing along direction $i$ while polar descent with $k\eta\ge c_i$ has fitted it
completely. *Proof.* (a) $1-(1-x)^k\le kx$ for $x\in[0,1]$; (b) Theorem 3.1(b). $\square$ **[proved; (b) conditional on orthogonality]**

*Open.* Part (b) for general (non-orthogonal) $\bar W$ in discrete time. The descent-lemma bound
$L_{k+1}\le L_k - \eta\|G_k\|_* + \tfrac{\eta^2}{2}\,n\,\lambda_{\max}$ only guarantees a residual of order $\eta n\lambda_{\max}/\lambda_{\min}$ along the weakest direction,
which is not scale-free; we have no general per-step separation without orthogonality. The numerical evidence for the
general case is the MLP table above (real Muon with momentum and Newton–Schulz, generic targets).

**Theorem 3.6 (any fixed feature map: the vulnerable input directions are those with large Mahalanobis feature norm).**
Let $f(x) = W\varphi(x)$ with an arbitrary fixed, differentiable feature map $\varphi:\mathbb R^{n_{\rm in}}\to\mathbb R^p$ (random features, or a frozen
trunk with a trained linear readout — e.g. a critic head on frozen features); $m\ge p$ training inputs with feature matrix $\Phi$ and $\hat\Sigma_\varphi = \Phi^\top\Phi/m\succ0$; targets $Y = \Phi W^{*\top} + E$,
$E$ i.i.d. $\mathcal N(0,s^2)$; $W\in\mathbb R^{d\times p}$ trained from 0 on the squared loss. For an input direction $u$ at a point $x$, let
$J_u := J_\varphi(x)\,u$ be the induced feature perturbation. The input sensitivity of the *noise-fitted part* of the trained model,
$\Delta(t) := (W(t) - W_{\rm clean}(t))J_u$ with $W_{\rm clean}$ the run on noiseless targets, satisfies:
$$\text{fully fitted } (W=\bar W\text{, reached by any interpolating method, e.g. the normalised flows of Theorem 3.5}):\qquad \mathbb E\|\Delta\|^2 = \frac{d\,s^2}{m}\;J_u^\top\hat\Sigma_\varphi^{-1}J_u ,$$
$$\text{gradient flow at time } t:\qquad \mathbb E\|\Delta(t)\|^2 = \frac{d\,s^2}{m}\;J_u^\top\big(I-e^{-\hat\Sigma_\varphi t}\big)\hat\Sigma_\varphi^{-1}\big(I-e^{-\hat\Sigma_\varphi t}\big)J_u \;\le\; \frac{d\,s^2}{m}\,t^2\;J_u^\top\hat\Sigma_\varphi J_u .$$
So a fully fitted model's noise sensitivity along $u$ is the *inverse*-covariance (Mahalanobis) norm of the feature perturbation,
large exactly when moving the input along $u$ moves the features into directions the training features barely vary in; the GD
model's is bounded by the *covariance* norm, small in exactly those directions.
*Proof.* The least-squares solution is $\bar W = W^* + E^\top\Phi\hat\Sigma_\varphi^{-1}/m$ and gradient flow gives $W(t) = \bar W\big(I-e^{-\hat\Sigma_\varphi t}\big)$
(Theorem 3.1(a) in the eigenbasis of $\hat\Sigma_\varphi$); the noise part along $J_u$ is $E^\top\Phi\hat\Sigma_\varphi^{-1}P_tJ_u/m$ with $P_t = I-e^{-\hat\Sigma_\varphi t}$ ($P_\infty = I$). Each of
its $d$ coordinates is Gaussian with variance $\tfrac{s^2}{m^2}\|\Phi\hat\Sigma_\varphi^{-1}P_tJ_u\|^2 = \tfrac{s^2}{m}J_u^\top P_t\hat\Sigma_\varphi^{-1}P_tJ_u$. For the bound use
$(1-e^{-\lambda t})^2/\lambda\le t^2\lambda$ on each eigenvalue. $\square$ **[proved; verified numerically with random ReLU features (3-dim input
manifold, off-manifold input variance $10^{-4}$): fitted sensitivity 1.18 (on) vs 10.11 (off) predicted, 1.16 / 9.65 by Monte Carlo; GD at
$t=50$: 0.0095 / 0.0105 predicted, 0.010 / 0.010 Monte Carlo]**

This extends the noise-sensitivity part of Theorem 3.1 / Corollary 3.2 from linear layers to any model that is linear in its trained
parameters given fixed features with $m\ge p$, and identifies the vulnerability measure: the Mahalanobis norm $J_u^\top\hat\Sigma_\varphi^{-1}J_u$ of a
feature perturbation. Off-manifold input directions are dangerous for a fully fitted model precisely to the extent that the features
respond to them in low-variance directions. *Scope.* It does not cover the lazy/NTK regime, where the tangent features have $p\gg m$
(one would need $\hat\Sigma_\varphi^+$ with $J_u$ projected onto $\mathrm{range}(\Phi^\top)$), and polar flow on a tangent-feature matrix is not Muon (which
orthogonalises each layer's matrix separately). Which optimizer reaches the fully fitted solution within a training budget is the
per-step question of Corollary 3.5′. An earlier version claimed lazy-regime coverage; that claim is withdrawn.

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
$U_T^\top U_N = 0$, $U_T^\top U_T = P_T$, $U_N^\top U_N = P_N$ (so $p\ge n$: the two feature subspaces are orthogonal and of total rank $n$). With
$$\mu_T = \frac{a_T^2}{a_T^2\lambda_T+\kappa},\qquad \mu_N = \frac{a_N^2}{a_N^2\lambda_N+\kappa},$$
we have $M = \mu_TP_T + \mu_NP_N$, $\bar g = \mu_T\lambda_T\,\beta \in T$, and
$$\operatorname{tr}C = \frac{s^2}{m}\Big(n_T\,\mu_T^2\lambda_T + n_N\,\nu(a_N)\Big),\qquad \nu(a_N) := \mu_N^2\lambda_N = \frac{a_N^4\,\lambda_N}{(a_N^2\lambda_N+\kappa)^2}.$$
The mean gradient has no off-manifold part, and the off-manifold share of the gradient energy is
$$F_N = \frac{\tfrac{s^2}{m}\,n_N\,\nu(a_N)}{\|\bar g\|^2+\operatorname{tr}C}.$$
*Proof.* $A\hat\Sigma A^\top + \kappa I$ acts as $a_T^2\lambda_T+\kappa$ on $\mathrm{range}\,U_T$, as $a_N^2\lambda_N+\kappa$ on $\mathrm{range}\,U_N$, and as $\kappa$ on the
orthogonal complement; conjugating by $A$ gives $M$. Substitute into Theorem 4.1. $\square$ **[proved]**

**Corollary 4.3 (monotonicity in the trunk's off-manifold gain).** For a regularised head, $\kappa>0$, $\nu$ is strictly increasing in $a_N>0$
(at $\kappa=0$, $\nu\equiv1/\lambda_N$ and the effect vanishes: it is a property of regularised, e.g. early-stopped, heads):
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
which, if $\langle\beta,u\rangle>0$ and $P_Nu\ne0$, is maximal at $t^\star = \langle\beta,u\rangle\big/\big(\gamma\eta\|P_Nu\|^2\big)$ and negative for $t > 2t^\star$; if
$\langle\beta,u\rangle\le0$ the reward is non-increasing from $t=0$. The time to
exploitation grows with the critic's alignment and shrinks with its off-manifold share.
*Proof.* $x_t = x_0+\eta tu$ and $P_Nx_0 = 0$. $\square$ **[proved]**

Counterparts: every frozen critic is exploited in the toy (reward −22…−36 by 300 steps), and on ELF the frozen self,
sibling and AdamW critics are exploited by 600 ascent steps (accuracy 3, 66, 45) **[measured]**. The frozen
independent-Muon critic is not (86) **[measured]**. Proposition 5.1 accommodates this only if its ratio
$\langle\beta,u\rangle/\|P_Nu\|^2$ is large; its first-order step is also the least useful (+1.4), so this case is unexplained (section 10).

**Test of Proposition 5.1 against the frozen attacks (resolves item T3 with existing data).** Proposition 5.1 orders frozen
critics by $t^\star\propto\langle\beta,u\rangle/\|P_Nu\|^2$. Using the available proxies — first-order usefulness = accuracy change of one actor step
at $\Lambda/4$ (native +4.7, sibling +5.3, independent Muon +1.4, AdamW −1.6) and off-manifold share of the critic gradient
(0.39, 0.40, 0.41, 0.34) — the predicted order of exploitation times is AdamW (immediate) < independent Muon (3.4) < native (12.1)
≈ sibling (13.3). The observed order at 500–600 frozen steps is native (3%) < AdamW (45%) ≈ sibling (66%) < independent
Muon (86%, not exploited). The proxies do not reproduce the order **[falsified with the available proxies]**. Two caveats:
the usefulness proxy is a parameter-space Adam step, not the output-space direction $u$ of the model, and Proposition 5.1 has
no term for self-similarity, which dominates the native case. The frozen-attack ordering is therefore not explained by
alignment and off-manifold share alone.

**Proposition 5.2 (refits with independent noise are self-averaging).** Let $g_t = \bar g + \xi_t$ with $\bar g\in T$ and $\xi_t$ i.i.d. with
mean zero and $\mathbb E\|P_N\xi_t\|^2 = \sigma_N^2$, and suppose $\|g_t\|\approx\bar\gamma$ is approximately constant. (With exact normalisation, the same
diffusive scaling holds without this approximation if the law of $\xi_t$ is invariant under $P_N\xi\mapsto-P_N\xi$ — true for the Gaussian
noise of Theorem 4.1 when its covariance is block-diagonal in $(T,N)$ — because then $\mathbb E[P_N\xi_s/\|g_s\|]=0$. The "small noise"
regime is *not* the Muon regime, which is noise-dominated.) Then
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
so $\mathbb E\|\delta_t\|^2 \ge \big(\tfrac{\eta}{\bar\gamma}\big)^2\sigma_N^2\sum_{s<t}\big(1+\tfrac\eta{\bar\gamma}\lambda_{\min}(\Gamma)\big)^{2s}$ grows geometrically when $\lambda_{\min}(\Gamma)>0$ — **as long as
$\|\Gamma\delta_t\|\ll\|\bar g\|$**, i.e. while the constant-normalisation approximation holds. With the exact normalised step, $\|\delta_t\|\le\eta t$
always, so once the feedback term dominates the growth is at most ballistic. The conclusion survives: a ballistic drift costs
$\tfrac\gamma2\|\delta_t\|^2\propto t^2$, which eventually dominates any linear on-manifold gain.
*Proof.* Unroll the linear recursion; the noise terms are independent with zero mean, so cross terms vanish, and
$\|(I+c\Gamma)^s v\|\ge(1+c\lambda_{\min}(\Gamma))^s\|v\|$ for $c\ge0$ (for symmetric PSD $\Gamma$). $\square$ **[proved, given the feedback assumption and within the
constant-normalisation regime; corrected after review: an earlier version claimed geometric growth without that qualification]**

Whether refits on Muon trunks produce $\Gamma\succ0$ is a **[conjecture]**. Supporting facts: refits on the states of a collapsing
policy rotate the Muon critic's gradient (cosine 0.60) **[measured]**; the iteration-to-iteration coherence of the actor
gradient falls as the native collapse unfolds (0.80 → 0.44) **[measured]**; and in the toy loop a whitened trunk never
improves the policy although its head is refit to the *true* reward at every step, while GD/AdamW trunks do (reward
peak +0.58 vs never above the start −0.19) **[verified numerically, `loop_toy.py`]**. Against a literal reading: the
off-manifold energy of the clean estimate does not grow in collapsing ELF runs **[measured]**, so if the feedback exists it
acts on answer-changing directions inside $T$ rather than on the PCA complement $N$.

**Proposition 5.5 (refit feedback in the linear–Gaussian model).** Model the states produced by the current
policy as $x = x_c + \zeta$ with $\zeta\sim\mathcal N(0,\Sigma_\zeta)$ and centre $x_c = x_0+\delta$, $x_0\in T$, $\delta\in N$, and the reward as
$R(x) = \langle\beta,x\rangle - \tfrac\gamma2\|P_Nx\|^2 + \varepsilon$. Refit a ridge head with intercept on the linear trunk of section 4 using
infinitely many such states. Then the critic's input gradient is
$$g(\delta) = M_\zeta\,\Sigma_\zeta\,(\beta - \gamma\,\delta),\qquad M_\zeta = A^\top(A\Sigma_\zeta A^\top+\kappa I)^{-1}A,$$
so in Proposition 5.3's notation $\Gamma = -\gamma\,P_NM_\zeta\Sigma_\zeta P_N$ (in general **not symmetric**). (a) If $\Sigma_\zeta$ is block-diagonal in $(T,N)$,
$P_NM_\zeta\Sigma_\zeta P_N = (P_NM_\zeta P_N)(P_N\Sigma_\zeta P_N)$ is a product of PSD matrices, whose eigenvalues are real and $\ge0$, so $-\Gamma$ has no
eigenvalue with negative real part: no linear instability (its symmetric part can still be indefinite, allowing transient growth).
In the spiked trunk of Proposition 4.2 with $\Sigma_\zeta = \lambda_TP_T+\lambda_NP_N$, $\Gamma = -\gamma\,\mu_N\lambda_N\,P_N\preceq0$: refitting **pulls the policy back**,
more strongly the larger $a_N$. (b) If $\Sigma_\zeta$ couples $T$ and $N$, $P_NM_\zeta\Sigma_\zeta P_N$ can have eigenvalues with negative real part, so $\Gamma$
can be **destabilising even for a linear trunk and a smooth reward** (independent check by the proof review, `scratchpad/review/check2.py`:
eigenvalues down to −0.89 with $p=6\ge n=4$).
*Proof.* With an intercept, the population ridge slope is $w = (A\Sigma_\zeta A^\top+\kappa I)^{-1}A\,\mathrm{Cov}(x,R)$. For Gaussian $\zeta$,
Stein's lemma gives $\mathrm{Cov}(x,R) = \Sigma_\zeta\,\mathbb E[\nabla R(x)] = \Sigma_\zeta(\beta - \gamma P_N x_c) = \Sigma_\zeta(\beta-\gamma\delta)$, since $\nabla R$ is affine. Then
$g = A^\top w = M_\zeta\Sigma_\zeta(\beta-\gamma\delta)$, and in the spiked model $M_\zeta\Sigma_\zeta = \mu_T\lambda_TP_T+\mu_N\lambda_NP_N$. $\square$
**[proved; verified numerically: slope of the refit gradient along a displacement −0.018 / −0.935 measured vs −0.018 / −0.947
predicted for $a_N$ = 0.3 / 3]**

**Consequence (M2, corrected after review).** In the isotropic/block-diagonal model linear refits are restoring, and a whitened
trunk is *more* restoring, so the destabilising feedback cannot come from that idealisation. But it *can* arise in a linear model as
soon as the policy's state noise couples on- and off-manifold directions (case (b)), and it can also come from the trunk's
nonlinearity or the verifier's non-smoothness (a reward that does not fall until a displacement crosses a decision boundary). An
earlier version of this note concluded that the feedback "must be nonlinear"; that conclusion is withdrawn. Which mechanism
operates on ELF is an empirical question (item T2).

**Measurement (T2) [measured; one drift, one refit seed, 64 probe states].** Drift: the native ELF-B Game24 run, seed 0,
$\theta_0$ (it0) → $\theta_1$ (it250). Interpolate $\theta(\epsilon)$. At each $\epsilon$, roll out 1536 training prompts, then refit each critic's head
from its pretrained weights with the RL recipe (AdamW $10^{-4}$, 200 steps, batch 64) on the $k=40$ states with verified rewards.
On fixed $\theta_0$ probe states, take the direction-normalised feedback along the drift of the clean estimate, $v$:
$\Gamma_v(\epsilon) = \langle g_\epsilon-g_0, v\rangle/(\epsilon\|g_0\|\|v\|)$. Success was 0.818 at $\epsilon=0$ and 0.811 at $\epsilon=0.5$.

| critic (trunk) | $\cos(g_0,v)$ | $\Gamma_v(0.5)$ mean (frac $>0$) | $\Gamma_v(1)$ mean (frac $>0$) |
|---|---|---|---|
| native (Muon, own) | +0.14 | −0.21 (5%) | −0.12 (5%) |
| MM7 (Muon, independent) | −0.10 | −0.11 (8%) | **+0.08 (92%)** |
| MA (AdamW) | +0.06 | −0.08 (11%) | −0.05 (14%) |
| Anat (AdamW-trained base's own critic) | +0.08 | −0.09 (16%) | −0.03 (14%) |

Readings.
- For small displacement every critic's refit is restoring, as Proposition 5.5(a) predicts for the linear regime.
- The native critic stays strongly restoring at full drift. Its collapse is therefore not driven by refit feedback, consistent with the
  self-similarity route dominating it (frozen native critic → 3%, §6).
- The independent Muon critic (MM7) flips sign between $\epsilon=0.5$ and $\epsilon=1$: at full drift its refit rewards *further* drift, at
  92% of probe states. This is a displacement-dependent, i.e. nonlinear, destabilising feedback. It matches the refit route documented for MM7:
  freezing its head improves RL (64.2 vs 50.7 at it750), and its refits rotate its gradient (cos 0.60 vs AdamW 0.84).
- The AdamW critics stay weakly restoring.
- Caveat: the drift is the native run's, not MM7's own trajectory, and there is one seed.
- Conclusion: on ELF, refit feedback is restoring near the base. For the independent Muon trunk it turns destabilising at a finite
  displacement, so the mechanism is the nonlinear one, not the linear $T$–$N$ coupling case (b), which would act at small $\epsilon$ as well.

**Proposition 5.4 (a benign sink stabilises the loop).** Let the pathwise gradient decompose as $J^\top G = J^\top b + J^\top G_{\rm rest}$, where $b$
is *answer-preserving* — $\langle\nabla R, J\,v\rangle = 0$ for every $v$ in the span of the updates it induces — and suppose the
actor step is normalised, $\delta\theta = \eta\,J^\top G/\|J^\top G\|$. Then the reward-relevant movement per step satisfies
$$|\delta R| \le \eta\,\|J^\top\nabla R\|\,\frac{\|J^\top G_{\rm rest}\|}{\|J^\top G\|} + O(\eta^2),$$
so a steep sink ($\|J^\top b\|\gg\|J^\top G_{\rm rest}\|$) reduces both the useful and the harmful movement per iteration by the factor
$\|J^\top G_{\rm rest}\|/\|J^\top G\|$.
*Proof.* $\delta R = \eta\langle J^\top\nabla R, J^\top G\rangle/\|J^\top G\| + O(\eta^2)$, the $b$ part contributes zero, and Cauchy–Schwarz bounds the rest. $\square$ **[proved]**

*Scope.* "Answer-preserving" is a first-order notion; second-order and cumulative effects of moving along $b$ are not covered, and
the proposition assumes a normalised plain-gradient actor step, not AdamW. It bounds the reward movement $|\delta R|$, not off-manifold
movement.

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
$\mathbb E_R\operatorname{tr}(BRBR^\top) = (\operatorname{tr}B)^2/n$, while the self case gives $\operatorname{tr}(B^2)$. Finally, with $b_i$ the eigenvalues of $B$, $\operatorname{tr}(B^2) = \sum b_i^2 \ge (\sum b_i)^2/n$
by Cauchy–Schwarz, with equality iff they are all equal. (Identifying $K=B$ for a self-similar critic is a modelling assumption.) $\square$
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
[Panickssery et al. 2024].

**Test (T1) — the conjecture is falsified as stated.** Along the frozen native attack (ELF-B, Game24, AdamW actor lr $10^{-6}$,
600 steps, 256 fixed probe prompts) we measured the base denoiser's relative self-consistency residual on the recorded states,
$\rho = \|(\mathrm d_{\theta_0}(s_k)-\hat x_{k-1})\odot m\|/\|\hat x_{k-1}\odot m\|$ at $k=40$, and its correlation with the true reward and with each
critic's logit **[measured]**:

| iter | acc | $\rho$ | corr$(-\rho, r)$ | corr$(-\rho,\ell)$ native / MM7 / MA | logit native / MM7 / MA | AUC native / MM7 / MA |
|---|---|---|---|---|---|---|
| 0 | 0.824 | 0.100 | 0.29 | 0.24 / 0.30 / 0.28 | 791 / 867 / 1430 | 0.89 / 0.95 / 0.90 |
| 200 | 0.875 | 0.135 | 0.56 | 0.50 / 0.62 / 0.48 | 911 / 869 / 1548 | 0.89 / 0.94 / 0.96 |
| 400 | 0.676 | 0.206 | 0.49 | 0.55 / 0.57 / 0.50 | 820 / 666 / 1266 | 0.71 / 0.86 / 0.84 |
| 500 | 0.340 | 0.254 | 0.43 | 0.38 / 0.60 / 0.53 | 786 / 395 / 778 | 0.59 / 0.85 / 0.85 |
| 600 | 0.031 | 0.379 | 0.50 | **−0.03** / 0.53 / 0.50 | **1021** / 31 / 71 | **0.51** / 0.99 / 0.95 |

Readings. (i) Self-consistency does predict success on the policy's own states, and it predicts every critic's logit about equally
(0.24–0.62), so it is not specific to the self-similar critic. (ii) The attack *lowers* self-consistency monotonically ($\rho$ 0.10 → 0.38), the opposite of
the conjecture, whose mechanism was that the policy raises it. (iii) At collapse the native critic decouples from self-consistency
(corr $-0.03$, AUC 0.51). Its logit *rises* to 1021, while the independent critics fall to 31 and 71 and still rank correctly (AUC 0.99 and 0.95). The exploited
states are therefore ones the critic's own frozen trunk ($=\theta_0$) regards as *inconsistent*, yet its head scores them highly. The
self-similarity route acts through the native head's readout of $\theta_0$ features along directions orthogonal to self-consistency, not by
raising self-consistency. It remains nonlinear (Proposition 6.1) and specific to the native critic (no transfer: the other critics' logits
fall 28-fold and 20-fold). Its precise form is open (M3).

**Hierarchy.** The H-ELF critic built on its own policy is *not* exploited by 600 frozen ascent steps (82.8 → 85.5;
flat: 82 → 3) **[measured]**: the jointly denoised coarse code blocks the self-similarity route. A mechanism
compatible with the conjecture above: in H-ELF, self-consistency includes code–token agreement, which the policy
cannot raise without matching tokens to their classes, i.e. without moving towards well-formed answers **[conjecture]**.

---

## 7. Summary of the mechanism

1. *Optimizer fingerprint.* Polar (Muon) updates give every singular direction the same step in a basis that rotates
   with the data (Lemmas 2.1–2.3), so Muon-trained layers become sensitive to low-variance input directions
   (Theorem 3.1 and Corollary 3.2: sensitivity $\propto\lambda^{-1/2}$ vs $\propto\lambda^{1/2}$ for GD, under orthogonal targets; Corollary 3.5′: per
   step under a stable step size, GD fits at most $2k\lambda_i/\lambda_{\max}$ of a direction while polar descent fits it at unit speed). The
   continuous-time flows do *not* separate the optimizers (Theorem 3.5). AdamW empirically keeps the spectral bias unless the data
   are axis-aligned (Proposition 3.3 and the numerical check; the mechanism is not proved). On ELF, Muon updates have stable
   rank/n 0.18–0.24 vs 0.03.
2. *Critic gradients.* A head refit on a trunk with large off-manifold gain has an input gradient whose noise is
   off-manifold and whose direction changes with every refit, at unchanged accuracy (Theorem 4.1, Corollary 4.3).
3. *Loop.* Frozen critics are eventually exploited (Proposition 5.1). With refits, a stable, steep, benign direction
   (AdamW trunks) absorbs the actor's normalised steps (Proposition 5.4); a refit gradient that rotates and depends on
   the policy's state can destabilise the loop (Proposition 5.3, conditional). Independent refit noise alone would be
   harmless (Proposition 5.2), which the data falsify for Muon critics. In the linear–Gaussian model refits are restoring when
   the state noise is block-diagonal in $(T,N)$ and can be destabilising when it couples them (Proposition 5.5). On ELF (T2) refits
   are restoring near the base and turn destabilising at full drift for the independent Muon critic only.
4. *Self-similarity* is a second, nonlinear route (frozen own-trunk critic → 3%), blocked by the hierarchy. It does not act by raising the
   denoiser's self-consistency: the attack lowers it (T1, §6).
5. *Persistence and remedy.* Gradient-flow fine-tuning cannot remove the Muon component, weight decay can (Theorem 3.4); for AdamW
   fine-tuning this is shown in the toy only.

**Practical rule.** For pathwise RL: never use a critic trunk that is the policy's own model; use a trunk whose optimizer
kept the spectral bias (AdamW/SGD), or remove the low-variance component with weight decay strong enough that
$\mu\eta k\gg1$. Head-side remedies (more data, more refits, sharpness-aware objectives, invariance augmentation) cannot fix
a whitened trunk: Corollary 4.3 holds at fixed on-manifold accuracy, and the toy augmentation and every ELF head-side
change failed.

---

## 8. Connections to prior theory

- **Steepest descent in operator norms.** Shampoo without accumulation is steepest descent in the spectral norm and Adam (without
  EMAs) in the $\ell_\infty$ norm [Bernstein & Newhouse 2024]; the orthogonalised update that Muon approximates with Newton–Schulz
  iterations is that spectral-norm steepest-descent direction [Jordan et al. 2024; Bernstein & Newhouse, *Modular duality*, 2024].
  Lemmas 2.1–2.3 are the elementary consequences used here.
- **Implicit bias of spectral descent.** Normalised steepest descent and Muon converge to the spectral-norm max-margin
  classifier on separable data, Adam to the $\ell_\infty$ max-margin one [Fan, Schmidt & Thrampoulidis 2025]: the descent norm determines
  which max-margin solution is selected (the margin itself still depends on the data).
- **Whitening makes data information unusable.** Second-order optimisation of a linear model equals first-order
  optimisation of whitened data, and whitening can prevent generalisation [Wadia et al. 2021]. Theorem 3.1 and Corollary 3.5′ are a
  matrix-polar analogue for the quantity that matters for pathwise RL, off-manifold sensitivity — with the caveat of Theorem 3.5 that
  the separation is per step, not along continuous-time flows.
- **Preconditioning and generalisation.** [Amari et al. 2021] give exact bias–variance results for preconditioned vs.
  plain GD in ridgeless regression; preconditioning helps under label noise or misaligned signal *in distribution*. Our
  results concern sensitivity *off* the data distribution, where preconditioning hurts; the two are compatible.
- **Eigenvalue spread in linear least squares.** GD on a linear model learns input eigendirections at rates set by their variance,
  so ill-conditioned inputs are learned slowly along small-eigenvalue directions [LeCun, Kanter & Solla 1991; Advani & Saxe 2020];
  Theorem 3.1(a) is this classical statement.
- **Early stopping ≈ ridge.** Early-stopped GD on least squares tracks ridge regression [Ali, Kolter & Tibshirani 2019],
  justifying the ridge-head model of section 4 for a head refit with a few plain gradient steps (the correspondence is for
  gradient flow; it does not cover AdamW-trained heads, which our RL refits use).
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
| Thm 3.5 (normalised flows fit everything in finite time) | polar and normalised GD fit all columns by $1.2\,T$ | proved + verified; **not a separation** (earlier claim withdrawn) |
| Cor 3.5′ (per-step separation, stable GD step) | Thm 3.1 discrete forms | proved (polar side conditional on orthogonality) |
| Thm 3.6 (fixed features, $m\ge p$: Mahalanobis sensitivity) | random ReLU features: fitted 1.18 vs 10.11 predicted (MC 1.16 / 9.65); GD 0.01 | proved + verified |
| Prop 3.3 (sign flow whitens when axis-aligned) | AdamW 0.32 → 0.82 when axis-aligned; SGD, Muon unchanged | proved (sign flow) + verified; AdamW's spectral bias in rotated data is empirical |
| Thm 3.4 (persistence; weight decay) | residuals match the discrete factors (case $\bar W'=0$); toy: gain 1.67 unchanged by AdamW, 0.13 with weight decay | proved + verified |
| Muon fingerprint on ELF | update stable rank/n 0.18–0.24 vs 0.03; weights 0.143 vs 0.034 | measured |
| Thm 4.1, Cor 4.3 (refit gradients) | Monte Carlo exact; ELF refit rotation 0.60 vs 0.84; agreement 0.20 (same Muon trunk) vs 0.81–0.85 | proved + measured |
| Cor 1.3 factors | ELF critic off-share 0.39–0.42 vs 0.33 vs 0.24; policy Jacobian 15× | measured (critic factor modest) |
| Prop 5.1 (frozen critics exploited) | toy, all trunks; ELF self / sibling / AdamW exploited by 600 steps | proved + measured; independent Muon not exploited (unexplained) |
| Prop 5.2 (i.i.d. refits harmless) | frozen-head MM7 64.2 vs refit 50.7 at it750 | **falsified** for Muon critics |
| Prop 5.3 (feedback instability) | refit rotation; coherence decay; toy loop | conditional (growth geometric only while $\|\Gamma\delta\|\ll\|\bar g\|$, then ballistic) |
| Prop 5.5 (linear refit feedback) | spiked case: slope −0.018 / −0.935 vs predicted −0.018 / −0.947; coupled case destabilising (review check) | proved; restoring only for block-diagonal state noise |
| Prop 5.1 orders frozen exploitation | proxies predict AdamW < indep. Muon < native ≈ sibling; observed native < AdamW ≈ sibling < indep. Muon | **falsified** (with available proxies) |
| Prop 5.4 (benign sink) | AdamW critics: steep shared answer-preserving direction; coherence 0.95; frozen attack +5,900 at flat accuracy | proved + measured |
| Prop 6.1 (linear self-alignment) | verified $n/\mathrm{PR}$ | proved; does not explain the measured self-similarity route |
| Self-similarity route | frozen own critic 82 → 3%, no transfer; H-ELF own critic 85.5 | measured; mechanism conjectural |
| Off-manifold drift during collapse | not observed (off-manifold energy flat) | **falsified** |
| Attack transfer between critics | none (other critics score exploited states lower) | measured; consistent with Thm 4.1 |

---

## 10. What the theory does not (yet) explain

1. **The nonlinear self-similarity route** (section 6): why the policy's own frozen critic is exploited to 3% while sibling
   and independent Muon critics are not. Proposition 6.1 shows linear self-alignment cannot be the cause.
2. **The feedback term $\Gamma$** of Proposition 5.3: Proposition 5.5 shows it is restoring for linear trunks with block-diagonal state
   noise and can be destabilising when the noise couples $T$ and $N$. T2 (§5) measures it on ELF: restoring near the base for every
   critic, and destabilising at full drift only for the independent Muon critic, so the mechanism is nonlinear. A model of that sign flip is open.
3. **The ordering of frozen-critic exploitation** (self ≪ AdamW ≈ sibling ≪ independent Muon) is not reproduced by
   Proposition 5.1 with the available alignment and off-share proxies (§5).
4. **Depth, attention and the general separation.** All proofs are for linear layers (sections 3–4), fixed features with $m\ge p$
   (Theorem 3.6), or a free policy (section 5); the per-step Muon-vs-GD separation is proved only for orthogonal targets
   (Corollary 3.5′), and why AdamW keeps the spectral bias in rotated data is not proved. Lazy/NTK and feature-learning regimes are
   not covered. The transfer to transformers rests on the measured spectra
   and gradient statistics, not on a theorem.
5. **ELF test of the weight-decay remedy** is underpowered ($\mu\eta k\approx0.16$ vs the toy's 3–9). The weight-decayed critic
   held to it500 (64.2 / 63.2) and then collapsed (final −6.9), as expected for an underpowered decay.

---

## References

- Ali, Kolter & Tibshirani (2019). A continuous-time view of early stopping for least squares regression. AISTATS.
- Amari, Ba, Grosse, Li, Nitanda, Suzuki, Wu & Xu (2021). When does preconditioning help or hurt generalization? ICLR.
- Balles & Hennig (2018). Dissecting Adam: the sign, magnitude and variance of stochastic gradients. ICML.
- Advani & Saxe (2020). High-dimensional dynamics of generalization error in neural networks. Neural Networks (arXiv:1710.03667).
- Bernstein & Newhouse (2024). Old Optimizer, New Norm: An Anthology. arXiv:2409.20325 (NeurIPS 2024 OPT workshop).
- Bernstein & Newhouse (2024). Modular Duality in Deep Learning. arXiv:2410.21265.
- Eisenstein et al. (2023). Helping or herding? Reward model ensembles mitigate but do not eliminate reward hacking.
- Fan, Schmidt & Thrampoulidis (2025). Implicit bias of spectral descent and Muon on multiclass separable data. NeurIPS (arXiv:2502.04664).
- Gao, Schulman & Hilton (2023). Scaling laws for reward model overoptimization. ICML.
- Ilyas, Santurkar, Tsipras, Engstrom, Tran & Madry (2019). Adversarial examples are not bugs, they are features. NeurIPS.
- Jordan et al. (2024). Muon: an optimizer for hidden layers in neural networks. Technical report.
- Panickssery, Bowman & Feng (2024). LLM evaluators recognize and favor their own generations. NeurIPS.
- LeCun, Kanter & Solla (1991). Eigenvalues of covariance matrices: application to neural-network learning. Physical Review Letters 66(18).
- Wadia, Duckworth, Schoenholz, Dyer & Sohl-Dickstein (2021). Whitening and second order optimization both make information in the dataset unusable during training, and can reduce or prevent generalization. ICML.

---

## Appendix: code

Theory checks in `hcfrl/theory/`: `verify_theorems.py` (closed forms of Thm 3.1, Thm 3.4, Thm 4.1 / Cor 4.3,
Prop 6.1; log `runs/theory/verify_theorems.log`), `whitening.py` (Cor 3.2 in an MLP), `dose.py`, `axis_check.py`
(Prop 3.3), `readout_agreement.py`, `head_churn.py` (Thm 4.1 in an MLP), `loop_toy.py`, `loop_fix.py`,
`loop_refine.py` (section 5, Thm 3.4). ELF measurements: `hcfrl/analysis/{spectra,cross_critic,offspan,attack_transfer,
multi_onestep,pull_effects,refit_churn}.py`; offline summaries in `hcfrl/analysis/offline/`.

---

## 11. Status of every open item (final for the current data, 2026-10-31)

**The theory is complete for the data currently available.** Every statement the existing measurements can support is
stated, proved in its model, and numerically checked; nothing in this document is pending. The unchecked items below are
*future work* that requires either new experiments (new data) or a different mathematical framework (mean-field /
random-matrix analysis of feature learning); they are recorded so that they are not lost, not because this write-up is unfinished.

**Resolved**
- [~] **M1. Orthogonality assumption of Theorem 3.1(b).** *Corrected after the proof review.* Theorem 3.5 removes the assumption for the
  continuous-time fit, but it holds for every normalised steepest-descent flow, including normalised GD, so it does not separate Muon
  from GD. The separation is per step under a stable step size (Corollary 3.5′), proved only under orthogonality; the general
  discrete-time case remains open.
- [~] **M2. Sign of the refit feedback $\Gamma$.** *Corrected after the proof review.* Proposition 5.5: restoring (eigenvalue sense) for linear
  trunks with block-diagonal state noise; possibly destabilising when the noise couples $T$ and $N$. The earlier conclusion that the
  feedback must be nonlinear is withdrawn. T2 (done) shows that on ELF the destabilising feedback is displacement-dependent (MM7 at $\epsilon=1$), i.e. nonlinear.
- [x] **T3. Does Proposition 5.1 explain the frozen-attack ordering?** Tested with existing data (§5): no, with the available proxies
  (falsified as stated); self-similarity, which the proposition lacks, dominates the native case.
- [x] **R1. Refined-trunk critics to it1500.** AdamW-refined −9.6, weight-decay-refined −6.9 (held to it500, then collapsed);
  consistent with Theorem 3.4 given the underpowered decay ($\mu\eta k\approx0.16$).

**Requires new data (outside the scope "given the data we have"; proposed experiments, optional)**
- [x] **T1. Self-consistency test of the self-similarity route (§6).** Done (2026-10-31, `attack_transfer.py --self_consistency 1`):
  the "raise self-consistency" conjecture is falsified. The attack raises the base residual 0.10 → 0.38, and the native critic decouples from it at collapse (table in §6).
- [x] **T2. Measure $\Gamma$ on ELF.** Done (2026-10-31, `refit_feedback.py`, table in §5). $\Gamma_v<0$ at $\epsilon=0.5$ for all four critics.
  At $\epsilon=1$ it is $+0.08$ (92% of probes) for MM7 and still negative for native and the AdamW critics. Follow-up (optional): MM7's own drift, more seeds.
- [ ] **T4. Powered test of Theorem 3.4 on ELF:** weight-decayed critic trunk with $\mu\eta k\gtrsim3$, off-manifold gain and RL, 3 seeds.
- [ ] **R2. ELF-M (3×) replication** of the fingerprint and of the gradient-agreement / refit-churn measurements (GPU partition queue).

**Requires mathematics beyond this write-up**
- [ ] **M3. Nonlinear self-similarity model** (critic = next denoiser call). Proposition 6.1 shows the linear model cannot produce it;
  T1 rules out self-consistency as the exploited quantity, so a model must explain a head that, on frozen $\theta_0$ features, scores
  states that are inconsistent for $\theta_0$ highly, and only for the policy that shares $\theta_0$.
- [x] **M4 (partly). Beyond linear layers.** Theorem 3.6 extends the noise-sensitivity law to any model linear in its trained parameters
  given fixed features with $m\ge p$ (random features; a frozen trunk with a linear readout), with the Mahalanobis norm
  $J_u^\top\hat\Sigma_\varphi^{-1}J_u$ as the vulnerability measure. *Corrected after review:* it does not cover the lazy/NTK regime ($p\gg m$).
- [ ] **M4 (rest).** Lazy/NTK regime ($p\gg m$, pseudo-inverse), feature-learning regime and the polar dynamics of all layers jointly: needs a mean-field or
  random-matrix analysis of Muon beyond this write-up.


---

## 12. Proof review (2026-10-31)

An independent adversarial review of every [proved] item (separate agent; its numerical checks in
`scratchpad/review/check{1,2,3}.py`) found the algebra correct throughout and three substantive overclaims, all now corrected:
(1) Theorem 3.5 as first stated drew a Muon-vs-GD separation from a flow argument that holds for every normalised steepest-descent
flow, including normalised GD — withdrawn and replaced by the per-step statement Corollary 3.5′; (2) "linear refits are always
restoring" (Prop. 5.5 consequence) holds only for block-diagonal state noise, and a $T$–$N$ coupling can destabilise even a linear
trunk — corrected; (3) Theorem 3.6's claimed lazy/NTK coverage contradicts its $m\ge p$ assumption — withdrawn. Smaller fixes:
Filippov existence/uniqueness arguments added (Lemma 2.3, Theorem 3.1(b)); Corollary 3.2's polar half marked conditional on
orthogonality; Proposition 5.1's sign condition; Proposition 5.2's symmetry assumption; Proposition 5.3's growth limited to
the constant-normalisation regime; $\kappa>0$ and $p\ge n$ made explicit; Proposition 3.3 no longer claims that AdamW keeps the spectral
bias in rotated data (empirical only); Theorem 3.4's numerical claim corrected to the discrete-time factors; citations corrected
(Bernstein & Newhouse 2024 is Shampoo; Muon via Jordan et al. and *Modular duality*; eigenvalue-spread results via LeCun et al. 1991 and
Advani & Saxe 2020 instead of Saxe et al. 2014 / Rahaman et al. 2019; an unverifiable 2026 reference removed). The pre-review version is
kept as `notes/theory_pre-review_2026-10-31.md`.
