"""Held-out cross-model transport, Hu equation 12, and implementation-family tests."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .estimators import fit_ridge, fit_procrustes, select_ridge, support_diagnostics
from .io import atomic_json, prepare_run, file_sha256
from .peers import PRIMARY_SOURCES


def hu_transport(source_displacements, target_train_displacements, train_indices, test_indices,
                 *, weight_permutation=None):
    """Hu equation 12, including its source-model test displacement requirement.

    Target test displacements are deliberately NOT an argument. Source test
    before/after activations are allowed information for this cross-model task.
    """
    s = np.asarray(source_displacements, float)
    train, test = np.asarray(train_indices, int), np.asarray(test_indices, int)
    y = np.asarray(target_train_displacements, float)
    if len(y) != len(train) or np.intersect1d(train, test).size:
        raise ValueError("Hu transport requires aligned train rows and disjoint held-out words/orbits")
    if not np.isfinite(s).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite displacement")
    normalized = s / np.maximum(np.linalg.norm(s, axis=1, keepdims=True), 1e-12)
    weights = normalized[test] @ normalized[train].T
    if weight_permutation is not None:
        permutation = np.asarray(weight_permutation, int)
        if sorted(permutation.tolist()) != list(range(len(train))):
            raise ValueError("Weight null must permute training relational weights")
        weights = weights[:, permutation]
    denominator = np.abs(weights).sum(1, keepdims=True)
    return np.divide(weights @ y, denominator, out=np.zeros((len(test), y.shape[1])), where=denominator > 1e-12)


def transport_scores(prediction, target, training_target):
    from scipy.stats import spearmanr
    p, y, train = map(lambda a: np.asarray(a, float), (prediction, target, training_target))
    baseline = train.mean(0)
    pc, yc = p - baseline, y - baseline
    denominator = np.linalg.norm(pc, axis=1) * np.linalg.norm(yc, axis=1)
    cos = np.divide(np.sum(pc * yc, 1), denominator, out=np.full(len(p), np.nan), where=denominator > 1e-12)
    a, b = np.linalg.norm(p, axis=1), np.linalg.norm(y, axis=1)
    rho = float(spearmanr(a, b).statistic) if len(a) > 1 and np.std(a) and np.std(b) else None
    return {"mean_squared_error": float(np.mean(np.sum((p-y) ** 2, 1))),
            "row_normalized_squared_error": float(np.mean(np.sum((p-y) ** 2, 1) / (np.sum(y*y,1)+1e-12))),
            "magnitude_spearman": rho,
            "residual_cosine": float(np.nanmean(cos)) if np.any(np.isfinite(cos)) else None,
            "undefined_residual_cosine_rows": int(np.sum(~np.isfinite(cos)))}


def fit_intertwiner(source_actions, target_actions, train_elements, *, source_train=None, target_train=None):
    source, target = np.asarray(source_actions, float), np.asarray(target_actions, float)
    idx = np.asarray(train_elements, int)
    if source.ndim != 3 or target.ndim != 3 or source.shape[0] != target.shape[0] or not len(idx):
        raise ValueError("Aligned action families and construction elements are required")
    ds, dt = source.shape[1], target.shape[1]
    system = np.concatenate([np.kron(np.eye(dt), source[g]) - np.kron(target[g].T, np.eye(ds)) for g in idx], 0)
    _, singular, vt = np.linalg.svd(system, full_matrices=False)
    a = vt[-1].reshape((ds, dt), order="F")
    a /= max(np.linalg.norm(a), 1e-12)
    scale = 1.
    if source_train is not None and target_train is not None:
        p, y = np.asarray(source_train) @ a, np.asarray(target_train)
        scale = float(np.sum(p * y) / max(np.sum(p*p), 1e-12))
    return a, {"normalization": "unit_Frobenius_homogeneous_solution",
               "constraint_singular_values": singular.tolist(), "rank_conditioning": support_diagnostics(a),
               "prediction_scale_fit_on_construction_only": scale,
               "rank_warning": "a low-rank or singular intertwiner does not certify equivalent mechanisms"}


def intertwiner_scores(a, source_actions, target_actions, elements):
    source, target = np.asarray(source_actions), np.asarray(target_actions)
    return {str(int(g)): float(np.linalg.norm(source[g] @ a - a @ target[g]) /
                              (np.linalg.norm(a @ target[g]) + 1e-12)) for g in elements}


def validate_action_permutation(permutation, multiplication):
    p, m = np.asarray(permutation, int), np.asarray(multiplication, int)
    if sorted(p.tolist()) != list(range(len(p))) or m.shape != (len(p), len(p)):
        raise ValueError("Action null requires a complete permutation and product table")
    if not np.array_equal(p[m], m[p[:,None], p[None,:]]):
        raise ValueError("Registered legal action relabeling must preserve multiplication")
    return p


def _contraction_fit(x, y, steps=1000, tolerance=1e-9):
    """Convex projected-gradient empirical equation G.1 fit, spectral norm <=1."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    a = np.zeros((x.shape[1], y.shape[1]))
    lipschitz = max(np.linalg.norm(x, 2) ** 2, 1e-12)
    converged = False
    for it in range(int(steps)):
        proposal = a - x.T @ (x @ a - y) / lipschitz
        u, s, vt = np.linalg.svd(proposal, full_matrices=False)
        proposal = (u * np.minimum(s, 1.)) @ vt
        difference = np.linalg.norm(proposal-a)
        a = proposal
        if difference <= tolerance * (1 + np.linalg.norm(a)):
            converged = True; break
    return a, {"iterations": it+1, "converged": converged,
               "spectral_norm": float(np.linalg.norm(a,2))}


def implementation_distance(x, y, fit_mask, evaluation_mask, **kwargs):
    x, y = np.asarray(x,float), np.asarray(y,float)
    fit, test = np.asarray(fit_mask,bool), np.asarray(evaluation_mask,bool)
    if np.any(fit & test) or not fit.any() or not test.any():
        raise ValueError("Implementation distance needs independent alignment/evaluation cases")
    a, ai = _contraction_fit(x[fit], y[fit], **kwargs)
    b, bi = _contraction_fit(y[fit], x[fit], **kwargs)
    distance = max(np.sqrt(np.mean(np.sum((x[test]@a-y[test])**2,1))),
                   np.sqrt(np.mean(np.sum((y[test]@b-x[test])**2,1))))
    return float(distance), {"source_to_target": ai, "target_to_source": bi,
                             "norm": "empirical_direct_sum_L2_RMS", "heldout": True}


def congruity(implementations_a, implementations_b, fit_mask, evaluation_mask, *,
              preservation_a, preservation_b, draws=1000, seed=42, tie_policy="algorithm_leq"):
    """Sun–Toneva Algorithm 1 on provided behavior-validated implementation sets.

    This function does not manufacture GetImpl from rotations/PCA/one alignment.
    Preservation arrays must provide one held-out check per implementation.
    """
    a, b = np.asarray(implementations_a,float), np.asarray(implementations_b,float)
    pa, pb = np.asarray(preservation_a,bool), np.asarray(preservation_b,bool)
    if a.ndim != 3 or b.ndim != 3 or len(a) < 2 or len(b) < 2 or a.shape[1] != b.shape[1]:
        raise ValueError("Congruity requires >=2 actual implementations per network on shared cases")
    if pa.shape != (len(a),) or pb.shape != (len(b),) or not pa.all() or not pb.all():
        raise ValueError("Every implementation must pass its separately declared preservation check")
    if tie_policy not in ("algorithm_leq", "strict", "half"):
        raise ValueError("Unknown comparison tie policy")
    rng = np.random.default_rng(seed); memo = {}; diagnostics = []
    def distance(side1, i, side2, j):
        key = tuple(sorted(((side1,int(i)),(side2,int(j)))))
        if key not in memo:
            left = a if side1 == 0 else b; right = a if side2 == 0 else b
            memo[key], diag = implementation_distance(left[i], right[j], fit_mask, evaluation_mask)
            diagnostics.append(diag)
        return memo[key]
    def indicator(x,y):
        if tie_policy == "algorithm_leq": return float(x <= y)
        if tie_policy == "strict": return float(x < y)
        return .5 if np.isclose(x,y,atol=1e-12,rtol=1e-10) else float(x < y)
    values = []
    for _ in range(int(draws)):
        i, ip = rng.integers(len(a),size=2); j, jp = rng.integers(len(b),size=2)
        values.append((indicator(distance(0,i,0,ip),distance(0,i,1,j)),
                       indicator(distance(1,j,1,jp),distance(1,j,0,i))))
    means = np.mean(values,0)
    return {"congruity": float(1-abs(means.sum()-1)), "p1": float(means[0]), "p2": float(means[1]),
            "draws": int(draws), "tie_policy": tie_policy, "seed": int(seed),
            "scope": "empirical_Algorithm1_on_validated_implementation_families_not_unique_mechanism_proof",
            "all_distance_optimizers_converged": all(d[k]["converged"] for d in diagnostics for k in ("source_to_target","target_to_source")),
            "implementation_counts": [len(a),len(b)],
            "source": "https://arxiv.org/html/2603.30002v1"}


def compare_pair(data, config=None):
    cfg = config or {}; train = np.asarray(data["train_mask"],bool)
    test = np.asarray(data["test_mask"],bool)
    val = np.asarray(data.get("validation_mask", np.zeros(len(train),bool)),bool)
    if np.any(train&test) or np.any(train&val) or np.any(test&val) or not train.any() or not test.any():
        raise ValueError("Cross-model construction/validation/test rows must be disjoint")
    if "units" in data:
        units = np.asarray(data["units"])
        if set(units[train]) & set(units[test]) or set(units[train]) & set(units[val]) or set(units[test]) & set(units[val]):
            raise ValueError("Held-out words/orbits must not leak across rows")
    x, y = np.asarray(data["source"],float), np.asarray(data["target"],float)
    ridge = select_ridge(x[train],y[train],x[val],y[val],affine=True) if val.any() else fit_ridge(x[train],y[train],float(cfg.get("fixed_lambda",.01)),affine=True)
    # Rectangular Procrustes is supported by thin SVD; coefficient is ds×dt.
    proc = fit_procrustes(x[train], y[train])
    methods = {"ridge_affine": transport_scores(ridge.predict(x[test]), y[test], y[train]),
               "procrustes": transport_scores(proc.predict(x[test]),y[test],y[train])}
    result = {"methods": methods, "split_counts": {"train":int(train.sum()),"validation":int(val.sum()),"test":int(test.sum())},
              "ridge_diagnostics":ridge.diagnostics, "procrustes_diagnostics":support_diagnostics(proc.coef),
              "information": "paired construction activations; held-out destination activations only scored"}
    if "source_displacements" in data:
        sd, td = np.asarray(data["source_displacements"]), np.asarray(data["target_displacements"])
        ti, hi = np.flatnonzero(train), np.flatnonzero(test)
        p = hu_transport(sd,td[train],ti,hi)
        result["hu_equation12"] = {"scores": transport_scores(p,td[test],td[train]), "source":PRIMARY_SOURCES["hu"],
                                    "source_test_displacements_required":True,
                                    "information": "source-model before/after test activations ARE observed; destination test displacement is NOT observed"}
        rng=np.random.default_rng(int(cfg.get("seed",42)))
        null_count=int(cfg.get("nulls",cfg.get("null_replicates",5)))
        result["hu_refitted_weight_nulls"]=[transport_scores(hu_transport(sd,td[train],ti,hi,weight_permutation=rng.permutation(len(ti))),td[test],td[train]) for _ in range(null_count)]
        gaussian=rng.normal(size=(len(hi),td.shape[1]))
        gaussian/=np.maximum(np.linalg.norm(gaussian,axis=1,keepdims=True),1e-12)
        gaussian*=np.sqrt(np.mean(np.sum(td[train]**2,axis=1)))
        result["hu_gaussian_training_RMS_matched"]={"scores":transport_scores(gaussian,td[test],td[train]),
                                                   "information":"magnitude fitted on destination construction displacements only"}
    if "source_actions" in data:
        tr = np.asarray(data["fit_elements"],int); te=np.asarray(data["heldout_elements"],int)
        if np.intersect1d(tr,te).size or not len(te): raise ValueError("Intertwiner requires held-out elements")
        sa,ta=np.asarray(data["source_actions"]),np.asarray(data["target_actions"])
        a,meta=fit_intertwiner(sa,ta,tr,source_train=x[train],target_train=y[train])
        result["intertwiner"]={"metadata":meta,"construction":intertwiner_scores(a,sa,ta,tr),
                                 "heldout":intertwiner_scores(a,sa,ta,te),
                                 "prediction":transport_scores(x[test]@a*meta["prediction_scale_fit_on_construction_only"],y[test],y[train])}
        nulls=[]
        for perm in data.get("legal_action_permutations",[]):
            perm=validate_action_permutation(perm,data["multiplication_table"])
            na,nm=fit_intertwiner(sa[perm],ta,tr,source_train=x[train],target_train=y[train])
            nulls.append({"permutation":perm.tolist(),"refit":True,"heldout":intertwiner_scores(na,sa[perm],ta,te),"metadata":nm})
        result["intertwiner_legal_refitted_nulls"]=nulls
    # Stable unit relabeling, refit rather than corrupting only the score labels.
    if "units" in data and int(cfg.get("nulls",cfg.get("null_replicates",5))):
        rng=np.random.default_rng(int(cfg.get("seed",42))); nulls=[]
        # Equal-size construction unit blocks are required for consistent swaps.
        units=np.asarray(data["units"]); labels=list(dict.fromkeys(units[train].tolist()))
        blocks=[np.flatnonzero(train & (units==label)) for label in labels]
        if len({len(v) for v in blocks})==1:
            for _ in range(int(cfg.get("nulls",cfg.get("null_replicates",5)))):
                permutation=rng.permutation(len(blocks)); targets=np.concatenate([y[blocks[j]] for j in permutation])
                source=np.concatenate([x[v] for v in blocks]); nm=fit_ridge(source,targets,ridge.normalized_lambda,affine=True)
                nulls.append({"refit":True,"unit_permutation":permutation.tolist(),"scores":transport_scores(nm.predict(x[test]),y[test],y[train])})
        result["registration_refitted_nulls"]=nulls
    return result


def run_crossmodel(config, output, cache_root=None, *, arrays=None):
    cfg=config.get("crossmodel",{}); prepare_run(output,config,"crossmodel")
    if arrays is None:
        path=cfg.get("input_npz")
        if not path: raise ValueError("Cross-model module requires explicit paired cache input_npz or arrays")
        loaded=np.load(path,allow_pickle=False)
        arrays={k:loaded[k] for k in loaded.files}
        provenance={"input_npz":str(path),"sha256":file_sha256(path)}
    else: provenance={"input":"explicit_arrays"}
    pairs=arrays if "source" not in arrays else {cfg.get("pair_id","pair"):arrays}
    out={name:compare_pair(data,cfg) for name,data in pairs.items()}
    if cfg.get("congruity"):
        data=cfg["congruity"]
        if not data.get("implementation_npz"):
            raise ValueError("Congruity cannot be replaced by one intertwiner; provide actual validated implementation families")
        v=np.load(data["implementation_npz"],allow_pickle=False)
        out["congruity"]=congruity(v["implementations_a"],v["implementations_b"],v["fit_mask"],v["evaluation_mask"],
                                     preservation_a=v["preservation_a"],preservation_b=v["preservation_b"],
                                     draws=int(data.get("draws",1000)),seed=int(data.get("seed",42)))
    result={"study":"crossmodel_heldout_transport", "technical_run":bool(cfg.get("technical_run",False)),
            "provenance":provenance,"pairs":out}
    atomic_json(Path(output)/"crossmodel.json",result); return result
