"""Numerical verification of the closed forms stated in notes/theory.md (Thm 3.1, Thm 3.4, Thm 3.5, Thm 3.6, Thm 4.1 / Cor 4.3, Prop 5.5, Prop 6.1)."""
import math
import torch
torch.manual_seed(0); torch.set_default_dtype(torch.float64)

def polar(G):
    U, S, Vh = torch.linalg.svd(G, full_matrices=False)
    r = (S > 1e-12 * S.max()).sum()
    return U[:, :r] @ Vh[:r]

# --- Thm 3.1: discrete GD vs discrete polar descent on L(W) = 1/2 tr((W - Wb) S (W - Wb)^T), Wb column-orthogonal
n, d, eta, K = 6, 10, 0.01, 400
lam = torch.tensor([1.0, 0.5, 0.1, 1e-2, 1e-3, 1e-4]); S = torch.diag(lam)
Q, _ = torch.linalg.qr(torch.randn(d, n)); c = torch.tensor([0.5037, 1.0071, 1.5013, 2.0049, 3.0029, 5.0063]); Wb = Q * c   # not multiples of eta: avoids exact zero columns, at which polar() amplifies round-off
Wg = torch.zeros(d, n); Wp = torch.zeros(d, n)
for k in range(K):
    Wg = Wg - eta * (Wg - Wb) @ S          # GD with step eta
    Wp = Wp - eta * polar((Wp - Wb) @ S)   # polar descent with step eta
gd_pred = (1 - (1 - eta * lam) ** K) * c
pol_pred_err = torch.clamp(c - K * eta, min=0)
print("Thm3.1 GD column norms  :", [round(x, 4) for x in Wg.norm(dim=0).tolist()])
print("        GD prediction   :", [round(x, 4) for x in gd_pred.tolist()])
print("        polar |W e_i - Wb e_i|:", [round(x, 4) for x in (Wp - Wb).norm(dim=0).tolist()], " bound max(c-K eta, eta):", [round(max(x, eta), 4) for x in pol_pred_err.tolist()])

# --- Thm 3.3: GD fine-tuning from a Muon solution keeps low-variance components; weight decay removes them
W0 = Wb.clone(); Wb2 = torch.zeros(d, n); mu = 0.5
Wg = W0.clone(); Ww = W0.clone()
for k in range(2000):
    Wg = Wg - eta * (Wg - Wb2) @ S
    Ww = Ww - eta * ((Ww - Wb2) @ S + mu * Ww)
print("Thm3.4 residual of initial component after GD  :", [round(x, 4) for x in (Wg.norm(dim=0) / c).tolist()], " predicted (1-eta lam)^k:", [round((1 - eta * l) ** 2000, 4) for l in lam.tolist()])
print("       after GD + weight decay mu=0.5          :", [round(x, 6) for x in (Ww.norm(dim=0) / c).tolist()], " predicted (1-eta(lam+mu))^k:", [round((1 - eta * (l + mu)) ** 2000, 6) for l in lam.tolist()])

# --- Thm 4.1 / Prop 4.3: ridge head on linear trunk, Monte Carlo mean/cov of the input gradient
nT, nN, p, m, s, kappa = 4, 20, 40, 200, 0.5, 0.1
n = nT + nN; lamT, lamN = 1.0, 1e-2
for aN in [0.1, 1.0, 10.0]:
    aT = 1.0
    Uf, _ = torch.linalg.qr(torch.randn(p, n))                  # orthonormal feature directions
    A = torch.cat([aT * Uf[:, :nT], aN * Uf[:, nT:]], 1)        # p x n trunk: gain aT on T, aN on N
    beta = torch.zeros(n); beta[:nT] = torch.randn(nT)
    O, _ = torch.linalg.qr(torch.randn(m, n))
    X = math.sqrt(m) * O * torch.sqrt(torch.cat([torch.full((nT,), lamT), torch.full((nN,), lamN)]))   # X^T X / m = Sigma exactly
    Sig = X.T @ X / m
    M = A.T @ torch.linalg.solve(A @ Sig @ A.T + kappa * torch.eye(p), A)
    gbar = M @ Sig @ beta; C = s ** 2 / m * M @ Sig @ M
    gs = []
    for r in range(4000):
        y = X @ beta + s * torch.randn(m)
        w = torch.linalg.solve(A @ Sig @ A.T + kappa * torch.eye(p), A @ X.T @ y / m)
        gs.append(A.T @ w)
    G = torch.stack(gs)
    muT = aT**2 / (aT**2 * lamT + kappa); muN = aN**2 / (aN**2 * lamN + kappa)
    stab_pred = gbar.norm()**2 / (gbar.norm()**2 + torch.trace(C))
    i, j = torch.randint(0, 4000, (2, 4000))
    stab_mc = (G[i] * G[j]).sum(1).mean() / (G ** 2).sum(1).mean()
    offshare_pred = (s**2 / m * nN * muN**2 * lamN) / (gbar.norm()**2 + torch.trace(C))
    offshare_mc = (G[:, nT:] ** 2).sum(1).mean() / (G ** 2).sum(1).mean()
    print(f"Thm4.1/Cor4.3 aN={aN:5.1f}: |mean err| {(G.mean(0) - gbar).norm():.2e}  cov rel err {((torch.cov(G.T) - C).norm() / C.norm()):.3f}  "
          f"stability pred {stab_pred:.3f} mc {stab_mc:.3f}  off-share pred {offshare_pred:.3f} mc {offshare_mc:.3f}  mu_N {muN:.2f}")

# --- Prop 6.1: Haar average of tr(B K) for K a rotated copy, vs aligned
nn_ = 30
for spec in ["flat", "spiky"]:
    sv = torch.ones(nn_) if spec == "flat" else torch.tensor([10.0] * 3 + [0.3] * (nn_ - 3))
    B = torch.diag(sv ** 2)
    aligned = torch.trace(B @ B)
    rots = [torch.linalg.qr(torch.randn(nn_, nn_))[0] for _ in range(2000)]
    haar = torch.stack([torch.trace(B @ R @ B @ R.T) for R in rots]).mean()
    pred_haar = torch.trace(B) ** 2 / nn_
    print(f"Prop6.1 {spec}: aligned {aligned:.2f}  Haar mc {haar:.2f}  pred tr(B)^2/n {pred_haar:.2f}  ratio n/PR {aligned / pred_haar:.2f}")

# --- Thm 3.5 (general Wb, no orthogonality): polar flow fits everything by T = sqrt(2 L0 / lam_min); GD does not
torch.manual_seed(1)
n, d, eta = 6, 10, 1e-3
lam = torch.tensor([1.0, 0.5, 0.1, 1e-2, 1e-3, 1e-4]); S = torch.diag(lam)
Wb = torch.randn(d, n) * torch.tensor([0.5, 0.7, 1.0, 3.0, 10.0, 30.0])     # generic (non-orthogonal) columns, noise-like large on low-variance dirs
L0 = 0.5 * torch.trace(Wb @ S @ Wb.T)
Tb = math.sqrt(2 * L0 / lam.min())
K = int(1.2 * Tb / eta)
Wp = torch.zeros(d, n); Wg = torch.zeros(d, n)
for k in range(K):
    Wp = Wp - eta * polar((Wp - Wb) @ S)
    Wg = Wg - eta * (Wg - Wb) @ S
print(f"Thm3.5 L0 {L0:.3f}  bound T {Tb:.1f}  ran t={K*eta:.1f}: polar loss {0.5*torch.trace((Wp-Wb)@S@(Wp-Wb).T):.2e}, "
      f"polar |W e_i|/|Wb e_i| {[round(x,3) for x in (Wp.norm(dim=0)/Wb.norm(dim=0)).tolist()]}, GD {[round(x,3) for x in (Wg.norm(dim=0)/Wb.norm(dim=0)).tolist()]}")

# --- Prop 5.5 (refit bias is restoring in the linear-Gaussian model): slope of the refit gradient along a displacement
nT, nN, p, m, kappa, gam = 3, 6, 20, 4000, 0.1, 2.0
n = nT + nN
for aN in [0.3, 3.0]:
    Uf, _ = torch.linalg.qr(torch.randn(p, n))
    A = torch.cat([Uf[:, :nT], aN * Uf[:, nT:]], 1)
    beta = torch.zeros(n); beta[:nT] = torch.randn(nT)
    PN = torch.zeros(n, n); PN[nT:, nT:] = torch.eye(nN)
    sd = torch.cat([torch.full((nT,), 1.0), torch.full((nN,), 0.1)])
    def refit_grad(delta):
        X = delta + sd * torch.randn(m, n)
        R = X @ beta - 0.5 * gam * ((X @ PN) ** 2).sum(1) + 0.1 * torch.randn(m)
        F = X @ A.T; Fc = F - F.mean(0); Rc = R - R.mean()
        w = torch.linalg.solve(Fc.T @ Fc / m + kappa * torch.eye(p), Fc.T @ Rc / m)
        return A.T @ w
    v = torch.zeros(n); v[nT] = 1.0
    eps = 0.5
    slopes = [float((refit_grad(eps * v) - refit_grad(torch.zeros(n))) @ v / eps) for _ in range(20)]
    muN = aN**2 / (aN**2 * 0.01 + kappa)
    print(f"Prop5.5 aN={aN}: measured slope <dg,v>/eps {sum(slopes)/20:+.3f}  predicted -gam*mu_N*lam_N {-gam*muN*0.01:+.3f}")

# --- Thm 3.6 (any fixed feature map): noise sensitivity along input direction u is the Mahalanobis norm of J_phi u
torch.manual_seed(2)
n_in, p, m, dout, s = 8, 40, 400, 5, 0.3
Tdim = 3
Fm = torch.randn(p, n_in) / math.sqrt(n_in)
def phi(x): return torch.relu(x @ Fm.T)
Xin = torch.cat([torch.randn(m, Tdim), 1e-2 * torch.randn(m, n_in - Tdim)], 1)      # data on a 3-dim manifold, tiny off-manifold variance
Phi = phi(Xin); Sh = Phi.T @ Phi / m
Wstar = torch.randn(dout, p) / math.sqrt(p)
x0 = Xin[0]
J = torch.autograd.functional.jacobian(phi, x0)                                   # p x n_in
for name, u in [("on-manifold", torch.eye(n_in)[0]), ("off-manifold", torch.eye(n_in)[-1])]:
    Ju = J @ u
    pred_pol = dout * s**2 / m * Ju @ torch.linalg.solve(Sh, Ju)
    t = 50.0
    ev, V = torch.linalg.eigh(Sh); c = V.T @ Ju
    pred_gd = dout * s**2 / m * ((1 - torch.exp(-ev * t))**2 / ev * c**2).sum()
    mc_pol, mc_gd = [], []
    for r in range(400):
        Y = Phi @ Wstar.T + s * torch.randn(m, dout)
        Wb = torch.linalg.solve(Sh, Phi.T @ Y / m).T                                 # least squares (polar flow limit)
        Wgd = Wb @ (torch.eye(p) - V @ torch.diag(torch.exp(-ev * t)) @ V.T)          # GD flow at time t
        mc_pol.append(((Wb - Wstar) @ Ju).pow(2).sum()); mc_gd.append(((Wgd - Wstar @ (torch.eye(p) - V @ torch.diag(torch.exp(-ev * t)) @ V.T)) @ Ju).pow(2).sum())
    print(f"Thm3.6 {name}: noise sensitivity fitted pred {pred_pol:.4f} mc {torch.stack(mc_pol).mean():.4f} | GD(t=50) pred {pred_gd:.4f} mc {torch.stack(mc_gd).mean():.4f}")
