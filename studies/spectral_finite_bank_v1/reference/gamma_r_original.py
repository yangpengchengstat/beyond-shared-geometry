"""
Type-constrained invariant-subspace budget gamma_r (Appendix B.3, Eq. 9-10)
for a complete labeled finite bank.

Input : Z  (n_ctx, p, r)  fixed-coordinate observations z_{c,s}  (e.g. q=16 PCA coords)
        group = (perms, irreps)  -- see make_C7 / make_D4 / make_abelian
Output: gamma_r, bounds on L*, untyped budget, type correction, per-type energies.

Convention: product gh = "first h, then g";  (g.s) = perms[g][s].
"""
import itertools
import numpy as np


# ----------------------------------------------------------------- groups
def _close_table(perms):
    """index of perm_g o perm_h  (first h then g)."""
    key = {tuple(p): i for i, p in enumerate(perms)}
    m = len(perms)
    tab = np.zeros((m, m), dtype=int)
    for g in range(m):
        for h in range(m):
            comp = tuple(perms[g][perms[h][s]] for s in range(len(perms[0])))
            tab[g, h] = key[comp]
    return tab


def abelian_real_irreps(orders, elems):
    """Real irreps of a finite abelian group. elems: list of tuples (g_1..g_k)."""
    irreps, seen = [], set()
    for a in itertools.product(*[range(N) for N in orders]):
        if a in seen:
            continue
        conj = tuple((-aj) % N for aj, N in zip(a, orders))
        seen.add(a); seen.add(conj)
        psi = np.array([np.exp(2j * np.pi * sum(aj * gj / N for aj, gj, N in zip(a, g, orders)))
                        for g in elems])
        if a == conj:
            irreps.append(dict(chi=psi.real, d=1, n=1, name=f"real{a}"))
        else:
            irreps.append(dict(chi=2 * psi.real, d=2, n=1, name=f"plane{a}"))
    return irreps


def make_C7():
    perms = [tuple((s + k) % 7 for s in range(7)) for k in range(7)]
    irreps = abelian_real_irreps([7], [(k,) for k in range(7)])
    return perms, irreps


def make_D4():
    # element (k,b): reflect first if b=1 (0<->1, 2<->3), then rotate k times (i -> i+1 mod 4)
    refl = {0: 1, 1: 0, 2: 3, 3: 2}
    perms, labels = [], []
    for b in (0, 1):
        for k in range(4):
            perms.append(tuple(((refl[s] if b else s) + k) % 4 for s in range(4)))
            labels.append((k, b))
    chis = []
    for a, bb in itertools.product((1, -1), (1, -1)):       # 1-dim characters chi(r)=a, chi(s)=bb
        chis.append(dict(chi=np.array([a ** k * bb ** b for k, b in labels], float),
                         d=1, n=1, name=f"chi({a:+d},{bb:+d})"))
    chiE = np.array([2 * np.cos(np.pi * k / 2) if b == 0 else 0.0 for k, b in labels])
    chis.append(dict(chi=chiE, d=2, n=2, name="E"))          # real type, complex dim n=2
    return perms, chis


# ----------------------------------------------------------------- core
def build_K(Z, perms, ridge=0.0):
    """Whiten by uncentered second moment, return K (D x D), D=m*r, and block perms Pi_t."""
    Z = np.asarray(Z, float)
    n_ctx, p, r = Z.shape
    m = len(perms)
    flat = Z.reshape(-1, r)
    Sigma = flat.T @ flat / flat.shape[0]
    w, U = np.linalg.eigh(Sigma)
    if w.min() <= 1e-10 * w.max():
        raise ValueError(f"Sigma not nonsingular: min/max eig = {w.min()/w.max():.2e}")
    S_is = U @ np.diag(w ** -0.5) @ U.T
    X = Z @ S_is                                              # (n_ctx, p, r)
    # v(c,s) = m^{-1/2} (x_{c, g.s})_g
    V = np.stack([X[:, [perms[g][s] for s in range(p)], :] for g in range(m)], axis=2)
    V = V.reshape(n_ctx, p, m * r) / np.sqrt(m)
    Vf = V.reshape(-1, m * r)
    K = Vf.T @ Vf / Vf.shape[0]
    return K, X, m, r


def block_perms(perms, tab, r):
    """Pi_t with (v Pi_t)_g = v_{gt}."""
    m = len(perms); D = m * r
    Pis = []
    for t in range(m):
        P = np.zeros((D, D))
        for g in range(m):
            gt = tab[g, t]
            P[gt * r:(gt + 1) * r, g * r:(g + 1) * r] = np.eye(r)
        Pis.append(P)
    return Pis


def knapsack(types, r):
    """types: list of (d, lam_desc). Maximize sum d*cumsum(lam)[k] s.t. sum d*k <= r."""
    NEG = -1e18
    dp = np.full(r + 1, NEG); dp[0] = 0.0
    choice = []
    for d, lam in types:
        cs = np.concatenate([[0.0], np.cumsum(lam)])
        new = np.full(r + 1, NEG); pick = np.zeros(r + 1, dtype=int)
        for b in range(r + 1):
            for k in range(0, min(len(lam), b // d) + 1):
                if dp[b - d * k] > NEG / 2:
                    val = dp[b - d * k] + d * cs[k]
                    if val > new[b] + 1e-15:
                        new[b], pick[b] = val, k
        dp = new; choice.append(pick)
    b = int(np.argmax(dp)); best = dp[b]
    ks = [0] * len(types)
    for i in range(len(types) - 1, -1, -1):
        k = choice[i][b]; ks[i] = int(k); b -= types[i][0] * k
    return best, ks


def gamma_r(Z, perms, irreps, tol=1e-8):
    n_ctx, p, r = np.asarray(Z).shape
    tab = _close_table(perms)
    K, X, m, r = build_K(Z, perms)
    D = m * r
    out = dict(trace_K=np.trace(K), r=r)                           # must equal r
    Pis = block_perms(perms, tab, r)
    out["commutation_err"] = max(np.linalg.norm(K @ P - P @ K) for P in Pis)   # data-completeness check

    types, info = [], []
    e_sum = 0.0
    for ir in irreps:
        P = (ir["n"] / m) * sum(ir["chi"][t] * Pis[t] for t in range(m))
        wP, UP = np.linalg.eigh((P + P.T) / 2)
        B = UP[:, wP > 0.5]
        assert np.allclose(np.sort(wP[wP > 0.5]), 1, atol=1e-6), "projector not idempotent"
        dim = B.shape[1]
        e_tau = np.trace(B.T @ K @ B) / r
        e_sum += e_tau
        if dim == 0:
            info.append(dict(name=ir["name"], d=ir["d"], dim=0, e=0.0)); continue
        lam = np.sort(np.linalg.eigvalsh(B.T @ K @ B))[::-1]
        d = ir["d"]
        assert dim % d == 0
        chunks = lam.reshape(-1, d)
        spread = float((chunks.max(1) - chunks.min(1)).max())   # Schur check: eigs repeat d times
        lam_mult = np.clip(chunks.mean(1), 0, None)
        types.append((d, lam_mult))
        info.append(dict(name=ir["name"], d=d, dim=dim, mult=len(lam_mult), e=e_tau, spread=spread,
                         lam_top=lam_mult[:3].round(4)))
    out["energy_sum"] = e_sum                                  # must equal 1

    best, ks = knapsack(types, r)
    top = np.sort(np.linalg.eigvalsh(K))[::-1][:r].sum()
    j = 0
    for it in info:
        if it["dim"] > 0:
            d, lam = types[j]
            it["k"] = ks[j]; it["b"] = d * lam[:ks[j]].sum() / r; j += 1
    g = best / r
    out.update(gamma=g, C_r=best, C_untyped=top, delta_type=top - best,
               L_lower=2 * (1 - g), L_upper=2 * (1 - g ** 2),
               eta=1 - np.trace(K @ K) / r, types=info)
    # direct check of Lemma 6: eta from per-element least squares in whitened coords
    Xf = X.reshape(-1, r)
    errs = []
    for g_ in range(m):
        Xg = np.stack([X[:, perms[g_][s], :] for s in range(p)], axis=1).reshape(-1, r)
        A = Xf.T @ Xg / Xf.shape[0]                            # E[x^T x]=I
        errs.append(np.mean(np.sum((Xf @ A - Xg) ** 2, 1)) / r)
    out["eta_direct"] = float(np.mean(errs))
    return out


def report(o, title=""):
    print(f"== {title}")
    print(f"  trace K={o['trace_K']:.6f} (should be r)  energy sum={o['energy_sum']:.6f}  "
          f"commutation err={o['commutation_err']:.1e}")
    print(f"  eta={o['eta']:.6f}  eta_direct={o['eta_direct']:.6f}")
    print(f"  gamma_r={o['gamma']:.4f}  untyped={o['C_untyped']/o['r']:.4f}  "
          f"delta_type={o['delta_type']:.4f}")
    print(f"  L* in [{o['L_lower']:.4f}, {o['L_upper']:.4f}]")
    for it in o["types"]:
        if it["dim"] > 0:
            print(f"    {it['name']:<16s} d={it['d']} e={it['e']:.4f} k={it['k']:>2d} "
                  f"b={it['b']:.4f} spread={it['spread']:.1e}")


# ----------------------------------------------------------------- self-tests
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # 1) paper's analytical example: G=S=C2xC2xC7, r=1  -> gamma_A=1/2, gamma_B=0, eta=1/2
    orders = [2, 2, 7]
    elems = list(itertools.product(range(2), range(2), range(7)))
    idx = {e: i for i, e in enumerate(elems)}
    perms = [tuple(idx[tuple((a + b) % N for a, b, N in zip(g, s, orders))] for s in elems) for g in elems]
    irr = abelian_real_irreps(orders, elems)
    xA = np.array([((-1) ** a + (-1) ** b) / np.sqrt(2) for a, b, k in elems])
    xB = np.array([np.sqrt(2) * np.cos(2 * np.pi * k / 7) for a, b, k in elems])
    for nm, x in (("A", xA), ("B", xB)):
        report(gamma_r(x.reshape(1, -1, 1), perms, irr), f"paper example {nm} (expect gamma={'0.5' if nm=='A' else '0'})")

    # 2) C7: shared rotation action vs context-specific conjugation (M1 vs M2-like), r=16
    r, n = 16, 28
    def rot(theta):
        return np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    R = np.eye(r)
    for j in range(3):
        R[2*j:2*j+2, 2*j:2*j+2] = rot(2 * np.pi * (j + 1) / 7)
    perms7, irr7 = make_C7()
    Rp = [np.linalg.matrix_power(R, k) for k in range(7)]
    z0 = rng.normal(size=(n, r))
    Zshared = np.stack([[z0[c] @ Rp[s] for s in range(7)] for c in range(n)])
    Zshared += 0.05 * rng.normal(size=Zshared.shape)
    report(gamma_r(Zshared, perms7, irr7), "C7 shared linear action (M1), noise 0.05")
    Zm2 = []
    for c in range(n):
        O, _ = np.linalg.qr(rng.normal(size=(r, r)))
        Zm2.append([z0[c] @ O.T @ Rp[s] @ O for s in range(7)])
    Zm2 = np.array(Zm2) + 0.05 * rng.normal(size=(n, 7, r))
    report(gamma_r(Zm2, perms7, irr7), "C7 context-specific conjugation (M2-like)")
    # state-code (M3-like): z = mu_s + b_c + noise
    mu = rng.normal(size=(7, r)); b = rng.normal(size=(n, 1, r))
    Zm3 = mu[None] + b + 0.05 * rng.normal(size=(n, 7, r))
    report(gamma_r(Zm3, perms7, irr7), "C7 state-centroid + context offset (M3-like)")

    # 3) D4: shared permutation-rep action (4 copies), and null by shuffling states per context
    perms4, irr4 = make_D4()
    a = rng.normal(size=(n, 4))
    Zd = np.stack([[np.kron(np.eye(4)[s], a[c]) for s in range(4)] for c in range(n)])
    Zd += 0.05 * rng.normal(size=Zd.shape)
    report(gamma_r(Zd, perms4, irr4), "D4 shared action (permutation rep x4)")
    Zn = np.stack([Zd[c, rng.permutation(4)] for c in range(n)])
    report(gamma_r(Zn, perms4, irr4), "D4 null: states shuffled within each context")
