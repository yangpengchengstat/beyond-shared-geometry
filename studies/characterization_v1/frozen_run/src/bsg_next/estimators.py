"""Prospective row-vector estimators; all model selection uses explicit validation data."""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations
import numpy as np

LAMBDA_GRID = (0., 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.)
RETENTION_GRID = (0., .25, .5, .75, 1., 1.25)
LOCAL_BANDWIDTHS = (.125, .25, .5, 1., 2., 4., 8., np.inf)


def numeric_code_identity():
    from pathlib import Path
    from .io import identity, file_sha256
    root=Path(__file__).parent
    names=("estimators.py","statistics.py","structural.py","synthetic.py","offline.py","groups.py","io.py")
    return identity({n:file_sha256(root/n) for n in names})


@dataclass
class LinearModel:
    coef: np.ndarray
    intercept: np.ndarray
    alpha: float = 0.
    normalized_lambda: float = 0.
    effective_dof: float = 0.
    diagnostics: dict = field(default_factory=dict)

    def predict(self, x):
        return np.asarray(x, dtype=np.float64) @ self.coef + self.intercept


def support_diagnostics(x):
    x = np.asarray(x, dtype=np.float64)
    s = np.linalg.svd(x, compute_uv=False)
    cutoff = (s[0] if len(s) else 0.) * 1e-12
    rank = int(np.sum(s > cutoff))
    return {"rank": rank, "singular_values": s.tolist(),
            "condition_on_support": float(s[0] / s[rank - 1]) if rank else None,
            "frobenius_norm": float(np.linalg.norm(x)),
            "spectral_norm": float(s[0]) if len(s) else 0.}


def fit_ridge(x, y, normalized_lambda=0., affine=False):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or not len(x):
        raise ValueError("Ridge requires nonempty aligned 2D train arrays")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite fitting data")
    xm, ym = (x.mean(0), y.mean(0)) if affine else (np.zeros(x.shape[1]), np.zeros(y.shape[1]))
    xc, yc = x - xm, y - ym
    scale = float(np.sum(xc * xc) / x.shape[1])
    lam = float(normalized_lambda)
    if lam < 0:
        raise ValueError("Negative ridge penalty")
    alpha = lam * scale
    u, s, vt = np.linalg.svd(xc, full_matrices=False)
    if alpha == 0:
        coef = np.linalg.pinv(xc, rcond=1e-12) @ yc
        df = float(np.sum(s > (s[0] * 1e-12 if len(s) else 0)))
    else:
        coef = (vt.T * (s / (s * s + alpha))) @ (u.T @ yc)
        df = float(np.sum(s * s / (s * s + alpha)))
    return LinearModel(coef, ym - xm @ coef, alpha, lam,
                       y.shape[1] * df + (y.shape[1] if affine else 0),
                       {"train_support": support_diagnostics(xc),
                        "operator": support_diagnostics(coef), "affine": bool(affine)})


def mse(y, pred):
    return float(np.mean(np.sum((np.asarray(y) - np.asarray(pred)) ** 2, axis=-1)))


def select_ridge(x, y, xv, yv, grid=LAMBDA_GRID, affine=False):
    if not len(xv):
        raise ValueError("Selection requires explicit nonempty validation data")
    candidates = [(fit_ridge(x, y, v, affine), float(v)) for v in grid]
    scores = [mse(yv, m.predict(xv)) for m, _ in candidates]
    best_score = min(scores)
    eligible = [i for i, v in enumerate(scores) if v <= best_score + 1e-8]
    i = min(eligible, key=lambda j: (np.linalg.norm(candidates[j][0].coef), candidates[j][1]))
    model = candidates[i][0]
    model.diagnostics["validation_scores"] = {str(v): s for (_, v), s in zip(candidates, scores)}
    return model


def select_action_ridge(train_pairs, validation_pairs, grid=LAMBDA_GRID, affine=False):
    """Choose one lambda per cell using only the supplied action keys.

    Generator-only callers MUST supply generator keys only in both dictionaries.
    """
    if set(train_pairs) != set(validation_pairs) or not train_pairs:
        raise ValueError("Train/validation action sets must agree and be nonempty")
    candidates, scores = [], []
    for value in grid:
        fitted = {g: fit_ridge(*xy, value, affine) for g, xy in train_pairs.items()}
        score = np.mean([mse(validation_pairs[g][1], fitted[g].predict(validation_pairs[g][0]))
                         for g in fitted])
        candidates.append(fitted); scores.append(float(score))
    best = min(scores)
    eligible = [i for i, v in enumerate(scores) if v <= best + 1e-8]
    i = min(eligible, key=lambda j: sum(np.linalg.norm(m.coef) for m in candidates[j].values()))
    meta = {"normalized_lambda": float(grid[i]), "validation_scores": dict(zip(map(str, grid), scores)),
            "action_count": len(train_pairs), "affine": bool(affine)}
    return candidates[i], meta


def fit_procrustes(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    u, _, vt = np.linalg.svd(x.T @ y, full_matrices=False)
    coef = u @ vt
    return LinearModel(coef, np.zeros(y.shape[1]), diagnostics={"orthogonal": True})


def state_centroids(z, states, train_mask):
    z, states, mask = np.asarray(z), np.asarray(states), np.asarray(train_mask, bool)
    labels = sorted(set(states.tolist()))
    if any(not np.any(mask & (states == x)) for x in labels):
        raise ValueError("A state has no training centroid; do not estimate it from test rows")
    return {x: np.asarray(z[mask & (states == x)].mean(0), float) for x in labels}


def decode_states(x, centroids):
    labels = list(centroids)
    d = np.sum((np.asarray(x)[:, None, :] - np.stack([centroids[k] for k in labels])[None]) ** 2, 2)
    return np.asarray(labels)[np.argmin(d, 1)]


def select_retention(x, y, source_means, target_means, grid=RETENTION_GRID):
    scores = [mse(y, target_means + a * (x - source_means)) for a in grid]
    best = min(scores)
    idx = next(i for i, v in enumerate(scores) if v <= best + 1e-8)
    return float(grid[idx]), dict(zip(map(str, grid), scores))


@dataclass
class LocalDisplacementModel:
    """Oracle-source-state kernel field retaining construction displacements.

    The single bandwidth multiplier is selected across all changed validation
    actions. No validation target or held-out activation is retained in the fit.
    """
    sources: dict
    displacements: dict
    scales: dict
    multiplier: float
    chunk_size: int = 128
    metadata: dict = field(default_factory=dict)

    def _weights(self, query, state, multiplier):
        source=self.sources[state]
        if np.isinf(multiplier) or self.scales[state]<=0:
            return np.full((len(query),len(source)),1/len(source))
        sigma=self.scales[state]*multiplier
        distance=np.maximum(np.sum(query*query,1)[:,None]+
                            np.sum(source*source,1)[None,:]-2*query@source.T,0.)
        logits=-distance/(2*sigma*sigma)
        weights=np.exp(logits-logits.max(1,keepdims=True))
        return weights/weights.sum(1,keepdims=True)

    def predict(self, x, states, action, multiplier=None):
        x=np.asarray(x,dtype=np.float64);states=np.asarray(states)
        if x.ndim!=2 or len(x)!=len(states):raise ValueError("Local queries require aligned 2D sources and states")
        result=x.copy();multiplier=self.multiplier if multiplier is None else float(multiplier)
        for state in np.unique(states):
            key=int(state)
            if key not in self.sources:raise ValueError("No construction pairs for oracle source state")
            ids=np.flatnonzero(states==state)
            for start in range(0,len(ids),self.chunk_size):
                batch=ids[start:start+self.chunk_size]
                result[batch]+=self._weights(x[batch],key,multiplier)@self.displacements[action][key]
        return result


def fit_local_displacement(x, states, targets, validation_pairs, *,
                           bandwidths=LOCAL_BANDWIDTHS, chunk_size=128,
                           max_scale_pairs=4096):
    """Fit a cached secondary kernel control, using construction and validation.

    ``validation_pairs[g]=(source_z,target_z,source_state)`` contains changed
    states only. Scales use at most 4096 deterministically spaced upper-triangle
    construction pairs per state; queries and smoother traces use bounded chunks.
    """
    x=np.asarray(x,dtype=np.float64);states=np.asarray(states)
    if x.ndim!=2 or len(x)!=len(states) or not len(x):raise ValueError("Local fitting requires aligned nonempty sources")
    if int(chunk_size)<1 or int(max_scale_pairs)<1:raise ValueError("Local chunk and scale pair budgets must be positive")
    if not validation_pairs or not set(validation_pairs)<=set(targets):raise ValueError("Local selection requires explicit validation actions")
    sources={};scales={};scale_info={};retained={g:{} for g in targets}
    for state in np.unique(states):
        key=int(state);ids=np.flatnonzero(states==state);source=x[ids].copy();sources[key]=source
        count=len(source);pairs=count*(count-1)//2
        ranks=np.unique(np.linspace(0,pairs-1,min(pairs,int(max_scale_pairs)),dtype=np.int64)) if pairs else np.empty(0,int)
        if len(ranks):
            starts=np.arange(count,dtype=np.int64)*(2*count-np.arange(count)-1)//2
            left=np.searchsorted(starts,ranks,side="right")-1
            right=left+1+ranks-starts[left]
            distance=np.linalg.norm(source[left]-source[right],axis=1)
            positive=distance[distance>1e-12]
        else:positive=np.empty(0)
        scales[key]=float(np.median(positive)) if len(positive) else 0.
        scale_info[str(key)]={"train_rows":count,"possible_pairs":pairs,"sampled_pairs":len(ranks),
                              "positive_sampled_pairs":len(positive),"median_positive_distance":scales[key],
                              "source_rank":support_diagnostics(source-source.mean(0))["rank"]}
        for g,y in targets.items():
            y=np.asarray(y,dtype=np.float64)
            if y.shape!=x.shape:raise ValueError("Local targets must be complete construction pairs")
            retained[g][key]=(y[ids]-source).copy()
    model=LocalDisplacementModel(sources,retained,scales,np.inf,int(chunk_size))
    scores=[]
    for multiplier in bandwidths:
        errors=[mse(yv,model.predict(xv,sv,g,multiplier)) for g,(xv,yv,sv) in validation_pairs.items()]
        scores.append(float(np.mean(errors)))
    # The broadest member wins a declared absolute numerical tie.
    best=min(scores);chosen=max(i for i,value in enumerate(scores) if value<=best+1e-8)
    model.multiplier=float(bandwidths[chosen])
    trace=0.
    for state,source in sources.items():
        for start in range(0,len(source),model.chunk_size):
            ids=np.arange(start,min(start+model.chunk_size,len(source)))
            weights=model._weights(source[ids],state,model.multiplier)
            trace+=float(weights[np.arange(len(ids)),ids].sum())
    model.metadata={"fit_scope":"construction_training_pairs_only","selection_scope":"changed_validation_actions_only",
        "oracle_source_state":True,"target_hidden_states_used_at_test":False,
        "bandwidth_grid":["inf" if np.isinf(v) else float(v) for v in bandwidths],
        "selected_multiplier":"inf" if np.isinf(model.multiplier) else model.multiplier,
        "validation_losses":dict(zip(("inf" if np.isinf(v) else str(v) for v in bandwidths),scores)),
        "global_multiplier_across_actions":True,"validation_action_count":len(validation_pairs),
        "training_rows_per_action":len(x),"training_action_count":len(targets),"states":scale_info,
        "scale_pair_sampling":"deterministically_evenly_spaced_upper_triangle_ranks",
        "max_scale_pairs_per_state":int(max_scale_pairs),"query_chunk_size":int(chunk_size),
        "retained_displacement_coordinates":len(x)*x.shape[1]*len(targets),
        "smoother_trace_per_action":trace,"conditional_smoother_dof_per_action":trace*x.shape[1],
        "capacity_scope":"fixed selected bandwidth; does not include model selection or source-coordinate dependence",
        "formal_calibration_claim":False}
    return model


def permutation_compose(p, q):
    """Apply p first, then q; matches row matrix multiplication."""
    return tuple(q[p[i]] for i in range(len(p)))


def permutation_power(p, k):
    r = tuple(range(len(p)))
    for _ in range(k):
        r = permutation_compose(r, p)
    return r


def consistent_action_images(group):
    """Exact faithful image subgroups. No random draws or automorphism exclusion."""
    name = str(group).upper()
    images = set()
    if name.startswith("C7"):
        for p in permutations(range(7)):
            if p != tuple(range(7)) and permutation_power(p, 7) == tuple(range(7)):
                images.add(tuple(sorted(permutation_power(p, k) for k in range(7))))
        assert len(images) == 120
    elif name.startswith("D4"):
        identity = tuple(range(4))
        ps = list(permutations(range(4)))
        for r in ps:
            if permutation_power(r, 4) != identity or permutation_power(r, 2) == identity:
                continue
            cyclic = {permutation_power(r, k) for k in range(4)}
            for s in ps:
                if (s in cyclic or permutation_power(s, 2) != identity or
                        permutation_compose(permutation_compose(s, r), s) != permutation_power(r, 3)):
                    continue
                image = cyclic | {permutation_compose(permutation_power(r, k), s) for k in range(4)}
                images.add(tuple(sorted(image)))
        assert len(images) == 3
    else:
        raise ValueError("Exact registered image reference is C7/D4 only")
    return list(sorted(images))


def inclusive_reference_rank(observed, values):
    a = np.asarray(values, float)
    tol = 1e-10 * max(1., abs(float(observed)))
    return float(np.mean(a <= observed + tol))


def fit_joint_secondary(train_pairs, init_models, product, identity_index, steps=500,
                        learning_rate=.002, weights=None, seed=42, device="cpu"):
    """Explicit secondary relation-supervised estimator; never selected as primary.

    product[i,j] is the row action applying i then j. This generic objective also
    handles S3; the legacy optimizer remains byte unchanged.
    """
    import torch
    torch.manual_seed(int(seed))
    keys = list(train_pairs); weights = weights or {"hom": 1., "rel":1., "identity": 1., "orth": .05}
    dtype = torch.float64
    r = torch.nn.Parameter(torch.tensor(np.stack([init_models[k].coef for k in keys]), dtype=dtype, device=device))
    x = {k: torch.as_tensor(train_pairs[k][0], dtype=dtype, device=device) for k in keys}
    y = {k: torch.as_tensor(train_pairs[k][1], dtype=dtype, device=device) for k in keys}
    eye = torch.eye(r.shape[-1], dtype=dtype, device=device)
    optimizer = torch.optim.AdamW([r], lr=float(learning_rate), weight_decay=1e-5)
    last = {}
    for _ in range(int(steps)):
        optimizer.zero_grad()
        data = torch.stack([((x[k] @ r[i] - y[k]) ** 2).mean() for i, k in enumerate(keys)]).mean()
        hom = torch.stack([((r[i] @ r[j] - r[int(product[i, j])]) ** 2).mean()
                           for i in range(len(keys)) for j in range(len(keys))]).mean()
        ident = ((r[int(identity_index)] - eye) ** 2).mean()
        orth = ((r.transpose(-2, -1) @ r - eye) ** 2).mean()
        if len(keys)==7 and 1 in keys:
            rel=((torch.linalg.matrix_power(r[keys.index(1)],7)-eye)**2).mean()
        elif len(keys)==8 and (1,0) in keys and (0,1) in keys:
            rr,ss=r[keys.index((1,0))],r[keys.index((0,1))]
            rel=(((torch.linalg.matrix_power(rr,4)-eye)**2).mean()+((ss@ss-eye)**2).mean()+
                 ((ss@rr@ss-r[keys.index((3,0))])**2).mean())/3
        elif len(keys)==6 and (1,0,2) in keys and (0,2,1) in keys:
            aa,bb=r[keys.index((1,0,2))],r[keys.index((0,2,1))]
            rel=(((aa@aa-eye)**2).mean()+((bb@bb-eye)**2).mean()+((torch.linalg.matrix_power(aa@bb,3)-eye)**2).mean())/3
        elif len(keys)==16 and (1,0) in keys and (0,1) in keys:
            aa,bb=r[keys.index((1,0))],r[keys.index((0,1))]
            rel=(((torch.linalg.matrix_power(aa,4)-eye)**2).mean()+((torch.linalg.matrix_power(bb,4)-eye)**2).mean()+((aa@bb-bb@aa)**2).mean())/3
        else:raise ValueError("Joint defining relations require registered C7/D4/S3 keys")
        loss = data + weights.get("hom", 1.) * hom + weights.get("rel",1.)*rel + weights.get("identity", 1.) * ident + weights.get("orth", .05) * orth
        loss.backward(); optimizer.step()
        last = {"data": float(data.detach()), "hom": float(hom.detach()), "relation":float(rel.detach()),
                "identity": float(ident.detach()), "orth": float(orth.detach()), "total": float(loss.detach())}
    out = r.detach().cpu().numpy()
    return {k: LinearModel(out[i], np.zeros(out.shape[-1])) for i, k in enumerate(keys)}, last


def fit_potential_field(x, y, xv, yv, widths=(32, 128), steps=500, seed=42):
    """Train-only linear/MLP potential, validation-selected. CPU; no model downloads."""
    linear = select_ridge(x, y, xv, yv, affine=True)
    candidates = [("linear", linear, mse(yv, linear.predict(xv)))]
    if widths:
        import torch
        xt, yt = torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
        for width in widths:
            torch.manual_seed(int(seed) + int(width))
            net = torch.nn.Sequential(torch.nn.Linear(x.shape[1], int(width)), torch.nn.ReLU(),
                                      torch.nn.Linear(int(width), y.shape[1])).double()
            opt = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=1e-4)
            for _ in range(int(steps)):
                opt.zero_grad(); loss = ((net(xt) - yt) ** 2).mean(); loss.backward(); opt.step()
            net.eval()
            class TorchPredictor:
                def __init__(self, module): self.module = module
                def predict(self, a):
                    with torch.no_grad(): return self.module(torch.tensor(a, dtype=torch.float64)).numpy()
            predictor = TorchPredictor(net)
            candidates.append((f"mlp_{width}", predictor, mse(yv, predictor.predict(xv))))
    best = min(v[2] for v in candidates)
    chosen = next(v for v in candidates if v[2] <= best + 1e-8)
    return chosen[1], {"selected": chosen[0], "validation_scores": {k: s for k, _, s in candidates}}
