"""Executable peer comparators and bounded original-task anchors.

No published scores are shipped as observations. All fits receive construction
arrays explicitly; callers are responsible for family-separated validation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import copy
import numpy as np
from scipy.interpolate import CubicSpline

from .groups import DAYS, get_group
from .io import atomic_json, identity, prepare_run, file_sha256, read_json, source_identity

PRIMARY_SOURCES = {
    "engels": {"paper": "https://proceedings.iclr.cc/paper_files/paper/2025/file/d3221cdb27e49d9c1cd35ad254feccfe-Paper-Conference.pdf",
               "sections": "5.1, equations 5–6", "code": "https://github.com/JoshEngels/MultiDimensionalFeatures"},
    "wurgaft": {"paper": "https://arxiv.org/html/2605.05115v1", "sections": "2–3, Appendix A.1–A.8",
                "code": "https://github.com/goodfire-ai/causalab/tree/manifold_steering"},
    "arora": {"paper": "https://arxiv.org/html/2609.35970v1", "sections": "2.2, 4–5, Appendix B.2",
              "code_status": "not_verified_from_inspected_primary_source"},
    "hu": {"paper": "https://arxiv.org/html/2607.04525v1", "sections": "5.1, equation 12",
           "code_status": "not_verified_from_inspected_primary_source"},
}


def pca(x, rank):
    x = np.asarray(x, float)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("PCA requires finite nonempty construction observations")
    mean = x.mean(0)
    _, s, vt = np.linalg.svd(x - mean, full_matrices=False)
    q = min(int(rank), int(np.sum(s > max(s[0], 1.) * 1e-12)))
    if q < 1:
        raise ValueError("Construction carrier has zero rank")
    basis = vt[:q].T
    return mean, basis, {"rank": q, "requested_rank": int(rank),
                         "retained_variance": float(np.sum(s[:q] ** 2) / np.sum(s ** 2)),
                         "singular_values": s.tolist(), "fit_rows": len(x)}


@dataclass
class CalendarPlane:
    mean: np.ndarray
    basis: np.ndarray
    probe: np.ndarray
    state_count: int
    metadata: dict

    def predict(self, x, target_states, preserve=True):
        x = np.atleast_2d(np.asarray(x, float))
        target = np.asarray(target_states, int).reshape(-1)
        if len(target) != len(x):
            raise ValueError("One target state per calendar edit")
        angles = 2 * np.pi * target / self.state_count
        circles = np.stack((np.cos(angles), np.sin(angles)), 1)
        right_inverse = np.linalg.pinv(self.probe, rcond=1e-12)
        if preserve:
            current = (x - self.mean) @ self.basis @ self.probe
            return x + (circles - current) @ right_inverse @ self.basis.T
        return self.mean + circles @ right_inverse @ self.basis.T


def fit_calendar_plane(x, states, state_count=7, pca_rank=5):
    """Engels-style supervised circular probe; no SAE discovery is claimed."""
    x, states = np.asarray(x, float), np.asarray(states, int)
    if state_count not in (7, 12):
        raise ValueError("Calendar plane is registered only for 7 weekdays or 12 months")
    mean, basis, info = pca(x, pca_rank)
    theta = 2 * np.pi * states / state_count
    targets = np.stack((np.cos(theta), np.sin(theta)), 1)
    probe = np.linalg.lstsq((x - mean) @ basis, targets, rcond=1e-12)[0]
    if np.linalg.matrix_rank(probe) < 2:
        raise ValueError("Calendar probe lacks two independent circular directions")
    meta = {"family": "Engels_calendar_plane", "scope": "method_adaptation",
            "source": PRIMARY_SOURCES["engels"], "pca": info, "action_rank": 2,
            "parameter_count": int(probe.size + mean.size + basis.size),
            "information": "construction state labels mapped to a known circle; target label supplied",
            "closure": "known circular coordinates impose the law, not independent discovery"}
    return CalendarPlane(mean, basis, probe, state_count, meta)


@dataclass
class CyclicManifold:
    mean: np.ndarray
    basis: np.ndarray
    centroids: np.ndarray
    angles: np.ndarray
    spline: CubicSpline
    metadata: dict

    def path(self, source, target, points=50, signed_turns=None):
        a, b = float(self.angles[int(source)]), float(self.angles[int(target)])
        shift = (b - a + np.pi) % (2 * np.pi) - np.pi
        if signed_turns is not None:
            shift += 2 * np.pi * int(signed_turns)
        return self.spline(a + np.linspace(0., 1., int(points)) * shift)

    def predict(self, x, targets, preserve=True):
        x = np.atleast_2d(np.asarray(x, float))
        z = self.centroids[np.asarray(targets, int)]
        # Original manifold endpoint replacement preserves off-PCA components.
        return x + (z - (x - self.mean) @ self.basis) @ self.basis.T if preserve else self.mean + z @ self.basis.T


def fit_manifold(x, states, state_count=7, pca_rank=64):
    """Periodic cubic-spline calendar manifold; D4/S3 are explicitly unsupported."""
    x, states = np.asarray(x, float), np.asarray(states, int)
    if state_count not in (7, 12):
        raise ValueError("This registered spline is cyclic; do not score D4/S3 as failures")
    if set(states.tolist()) != set(range(state_count)):
        raise ValueError("Every cyclic state needs a construction centroid")
    mean, basis, info = pca(x, pca_rank)
    z = (x - mean) @ basis
    centroids = np.stack([z[states == i].mean(0) for i in range(state_count)])
    if basis.shape[1] < 2:
        raise ValueError("Cyclic manifold requires at least two construction dimensions")
    # Wurgaft A.3 estimates intrinsic angle from the top two centroid coordinates.
    angles = np.mod(np.arctan2(centroids[:, 1], centroids[:, 0]), 2 * np.pi)
    order = np.argsort(angles)
    theta = angles[order]
    if np.min(np.diff(theta)) < 1e-8:
        raise ValueError("Coincident intrinsic angles; cyclic spline is unidentifiable")
    spline = CubicSpline(np.r_[theta, theta[0] + 2 * np.pi],
                         np.concatenate((centroids[order], centroids[order[:1]])),
                         bc_type="periodic", extrapolate="periodic")
    return CyclicManifold(mean, basis, centroids, angles, spline,
                         {"family": "Wurgaft_periodic_spline", "scope": "method_adaptation",
                          "source": PRIMARY_SOURCES["wurgaft"], "pca": info,
                          "centroid_count": state_count,
                          "parameter_count": int(spline.c.size + mean.size + basis.size),
                          "information": "construction state centroids; unsupervised intrinsic angles",
                          "endpoint_warning": "integer-state endpoints interpolate training centroids exactly",
                          "complement": "preserved for manifold; full replacement only in named faithful linear comparator"})


class ConditionalMean:
    """Arora Appendix B.2 conditional-mean steering, adapted to supplied labels."""
    def __init__(self, x, labels):
        self.mean = np.asarray(x, float).mean(0)
        self.means = {label: np.asarray(x, float)[np.asarray(labels) == label].mean(0)
                      for label in sorted(set(labels))}
        self.metadata = {"family": "Arora_conditional_mean", "scope": "method_adaptation",
                         "source": PRIMARY_SOURCES["arora"], "fit_rows": len(x),
                         "parameter_count": len(self.means) * np.asarray(x).shape[1],
                         "information": "construction labels only; no full-ANOVA claim"}

    def predict(self, x, sources, targets):
        return np.asarray(x) + np.stack([self.means[t] - self.means[s] for s, t in zip(sources, targets)])


def functional_anova(activations):
    """Exact complete-grid three-factor Hoeffding decomposition.

    Shape A×B×C×D. A missing/incomplete sparse sample must not be padded with
    zeros: use complete_factorial_array to validate the grid first.
    """
    x = np.asarray(activations, float)
    if x.ndim != 4 or not np.isfinite(x).all():
        raise ValueError("ANOVA requires a finite complete A×B×C×hidden array")
    terms = {(): x.mean((0, 1, 2), keepdims=True)}
    from itertools import combinations
    for degree in (1, 2, 3):
        for subset in combinations(range(3), degree):
            axes = tuple(i for i in range(3) if i not in subset)
            term = x.mean(axes, keepdims=True) if axes else x.copy()
            for other, value in terms.items():
                if set(other).issubset(subset):
                    term = term - value
            terms[subset] = term
    return terms


def complete_factorial_array(records, activations, factors=("A", "B", "C")):
    x = np.asarray(activations, float)
    if len(x) != len(records):
        raise ValueError("Factorial records and activations differ")
    levels = [sorted({r[k] for r in records}) for k in factors]
    output = np.empty(tuple(map(len, levels)) + (x.shape[1],))
    seen = set()
    for row, values in zip(records, x):
        idx = tuple(levels[j].index(row[k]) for j, k in enumerate(factors))
        if idx in seen:
            raise ValueError("Duplicate cell: separate replicate/family grids before ANOVA")
        seen.add(idx); output[idx] = values
    if len(seen) != int(np.prod([len(v) for v in levels])):
        raise ValueError("Incomplete factorial grid; exact ANOVA is not estimable")
    return output, levels


def anova_diagnostics(x, terms=None):
    x = np.asarray(x, float); terms = functional_anova(x) if terms is None else terms
    denominator = np.mean(np.sum((x - terms[()]) ** 2, axis=-1))
    reconstructed = sum(terms.values())
    return {"reconstruction_error": float(np.max(np.abs(reconstructed - x))),
            "energy_fractions": {"".join("ABC"[i] for i in key):
                                 float(np.mean(np.sum(value ** 2, -1)) / denominator) if denominator else None
                                 for key, value in terms.items() if key},
            "scope": "exact_complete_factorial_decomposition_not_a_native_mechanism_claim"}


def build_arora_factorial(domain="C7", replicates=1):
    group = get_group(domain)
    if group.name != "C7":
        raise ValueError("Registered Arora adaptation currently supports weekday relation triples")
    records = []
    for rep in range(int(replicates)):
        for a in range(7):
            for b in range(7):
                for c in range(7):
                    text = (f"The event is scheduled from {DAYS[a]} to {DAYS[b]}. "
                            f"This is the same interval as from {DAYS[c]} to")
                    records.append({"text": text, "A": a, "B": b, "C": c,
                                    "family_id": f"arora_adapted_{rep}", "record_id": f"arora/{rep}/{a}/{b}/{c}",
                                    "answer": DAYS[(c + b - a) % 7], "candidates": list(DAYS)})
    return records


def build_native_factorial(config, domain="C7", split="train", test_families=None):
    """Complete state × first primitive × second primitive native programs.

    This is a common-benchmark adaptation, not Arora's original relation task.
    Every family/paraphrase is its own complete grid. C7 uses r and r^-1;
    D4_free8 uses r and s. Both factors are actual model instructions.
    """
    from .native import _build, _domain
    from .data import render_native_record
    group=get_group(_domain(domain))
    templates=_build(config,group.name,split,test_families=test_families)
    exemplars={}
    for row in templates:exemplars.setdefault((row["family_id"],row["paraphrase_id"]),row)
    primitives=list(group.generators)
    if len(primitives)==1:primitives.append(group.inverse(primitives[0]))
    if len(primitives)!=2:raise ValueError("Registered native factorial requires exactly two primitive choices")
    records=[]
    for (family,paraphrase),base in sorted(exemplars.items()):
        for a in range(group.state_count):
            for b in range(2):
                for c in range(2):
                    row=render_native_record(base,initial_state=a,prefix_elements=[],
                                             continuation_elements=[primitives[b],primitives[c]])
                    row.update(A=a,B=b,C=c,record_id=f"{family}/p{paraphrase}/factorial/{a}/{b}/{c}",
                               scenario_id=f"factorial/{a}/{b}/{c}",word_split="development",
                               factorial_scope="native_state_two_executed_primitives")
                    records.append(row)
    return records


def _factorial_groups(records):
    groups={}
    for index,row in enumerate(records):groups.setdefault((row["family_id"],row.get("paraphrase_id",0)),[]).append(index)
    return groups


def run_factorial_stage(config, output, *, runtime_factory=None, cache_root=None, mode="full"):
    """Run exact complete-grid ANOVA and construction-fitted causal adaptations.

    A/B/C are source state and two *natively executed* operations. Construction
    grids fit all intervention tables; held-out grids only report decomposition
    and score frozen edits. Test ANOVA is descriptive, never a fitted editor.
    """
    from .native import normalized_config,_models,_domain
    from .runtime_records import CHECKPOINT_SCHEMA,recorded_session
    config=normalized_config(config)
    if runtime_factory is None and not config.get("execution",{}).get("allowed_model_execution",False):
        raise PermissionError("Factorial model execution requires explicit authorization")
    if mode not in ("full","prepare","pilot"):
        raise ValueError("Factorial mode must be full, prepare, or pilot")
    prepare_run(output,config,"peer_factorial",provenance={
        "checkpoint_schema":CHECKPOINT_SCHEMA,"numeric_source_identity":source_identity(Path(__file__).parent),
        "mode":mode})
    cfg=config["native"];extra=config.get("peers",{}).get("factorial",{})
    splits=["validation"] if mode in ("prepare","pilot") else ["validation","test"]
    total=len(_models(config))*sum(len(build_native_factorial(config,_domain(domain),split,
        test_families=extra.get("test_families") if split=="test" else None))
        for domain in cfg.get("domains",["C7","D4_free8"]) for split in splits)
    if not total:raise ValueError("Factorial stage has no planned recipients")
    with recorded_session(output,"peer_factorial",total) as progress:
        return _run_factorial_locked(config,Path(output),runtime_factory,cache_root,mode,progress)


def _run_factorial_locked(config,output,runtime_factory,cache_root,mode,progress):
    import hashlib
    from .native import _models,_domain,_sites,make_runtime,collect_features,_write_rows
    from .runtime_records import load_record,save_record
    cfg=config["native"];extra=config.get("peers",{}).get("factorial",{})
    cache=Path(cache_root) if cache_root else output/"cache"
    domains=[_domain(v) for v in cfg.get("domains",["C7","D4_free8"])]
    result={"study":"complete_native_three_factor_ANOVA","technical_run":bool(cfg.get("technical_run",False)),
            "scope":"Arora_inspired_common_benchmark_adaptation_not_original_paper_reproduction",
            "source":PRIMARY_SOURCES["arora"],"cells":{},
            "information":"all editor tables and subspaces fit construction grids only; exact held-out ANOVA descriptive"}
    for mi,model in enumerate(_models(config)):
        runtime=make_runtime(model,config.get("runtime",{}),runtime_factory)
        try:
            for domain in domains:
                group=get_group(domain);cell=f"m{mi}.{domain}";directory=output/cell
                train=build_native_factorial(config,domain,"train")
                validation=build_native_factorial(config,domain,"validation")
                candidates=[s for s in _sites(config,model,runtime) if s[2]=="answer"]
                if not candidates:raise ValueError("Factorial diagnostic needs a declared final-token site")
                # A predeclared late site, with no test- or validation-effect search.
                layer=candidates[0][0];readout="final_token"
                x=collect_features(runtime,train,[layer],readout,cache/cell)[layer]
                grids=[]
                for ids in _factorial_groups(train).values():
                    array,_=complete_factorial_array([train[i] for i in ids],x[ids]);grids.append(array)
                population=np.mean(grids,axis=0);terms=functional_anova(population)
                # Optional matched native q support; full-D and rank-q edits are
                # separately named and carry their own parameter/rank declaration.
                mean,basis,info=pca(x,int(cfg.get("rank",16)))
                frozen={"layer":layer,"readout":readout,"train_records_identity":identity(train),
                        "grid_shape":list(population.shape),"pca":info,
                        "editor_scope":"construction_family_average_complete_ANOVA",
                        "terms":{str(k):v.tolist() for k,v in terms.items()}}
                frozen_path=directory/"factorial_frozen.json"
                if frozen_path.exists():
                    if identity(read_json(frozen_path))!=identity(frozen):
                        raise ValueError("Factorial frozen construction changed; use a new output")
                else:atomic_json(frozen_path,frozen)
                splits=[("validation",validation)]
                if mode not in ("prepare","pilot"):
                    splits.append(("test",build_native_factorial(config,domain,"test",test_families=extra.get("test_families"))))
                cell_result={"frozen_identity":identity(frozen),"splits":{},"construction_rows":len(train)}
                for split,records in splits:
                    arrays=collect_features(runtime,records,[layer],readout,cache/cell)[layer]
                    descriptive=[]
                    for key,ids in _factorial_groups(records).items():
                        grid,_=complete_factorial_array([records[i] for i in ids],arrays[ids])
                        descriptive.append({"family_id":key[0],"paraphrase_id":key[1],**anova_diagnostics(grid)})
                    checkpoints=[]
                    for record,h in zip(records,arrays):
                        descriptor={"record_identity":identity(record),"frozen_identity":identity(frozen),
                                    "recipient_feature_sha256":hashlib.sha256(np.ascontiguousarray(h).tobytes()).hexdigest(),
                                    "basis_sha256":hashlib.sha256(np.ascontiguousarray(basis).tobytes()).hexdigest(),
                                    "cell":cell,"split":split,"layer":layer,"readout":readout}
                        target=directory/"recipient_checkpoints"/split/(identity(record["record_id"])+".json")
                        resumed=target.exists()
                        if resumed:
                            payload=load_record(target)
                            if payload.get("descriptor")!=descriptor:
                                raise ValueError("Factorial recipient checkpoint descriptor mismatch")
                            _validate_factorial_rows(payload["rows"],record)
                        else:
                            progress.update(detail={"cell":cell,"split":split,"record_id":record["record_id"],"state":"scoring"})
                            rows=_score_factorial_recipient(runtime,record,h,group,population,terms,basis,layer,readout)
                            _validate_factorial_rows(rows,record)
                            save_record(target,{"descriptor":descriptor,"rows":rows})
                        checkpoints.append(target)
                        progress.advance(resumed=resumed,detail=f"{cell}/{split}/{record['record_id']}")
                    # Rebuild the compatible JSONL export from completed atomic records.
                    # An incomplete recipient is retried (at most eight scores), never
                    # committed as partial or counted twice after an interruption.
                    _write_rows(directory/f"{split}_evaluations.jsonl",
                                (row for path in checkpoints for row in load_record(path)["rows"]))
                    atomic_json(directory/f"{split}_ANOVA.json",descriptive)
                    cell_result["splits"][split]={"recipients":len(records),"evaluations":len(checkpoints)*8,"complete_grids":len(descriptive),
                                                  "test_fit_used_for_interventions":False}
                result["cells"][cell]=cell_result
        finally:
            if hasattr(runtime,"close"):runtime.close()
    atomic_json(output/"factorial_results.json",result);return result


def _validate_factorial_rows(rows,record):
    expected={(kind,support) for kind in ("state","operation","joint") for support in ("full_D","matched_q")}
    expected.update({("three_way_ablation","full_D"),("delta_zero_sham","full_D")})
    if len(rows)!=8 or {(r["kind"],r["support"]) for r in rows}!=expected or any(
            r["record_id"]!=record["record_id"] or r["family_id"]!=record["family_id"] for r in rows):
        raise ValueError("Factorial recipient has incomplete or mismatched evaluations")


def _score_factorial_recipient(runtime,record,h,group,population,terms,basis,layer,readout):
    from .native import make_patch,score_case
    from .data import render_native_record
    rows=[]
    index=(int(record["A"]),int(record["B"]),int(record["C"]))
    a,b,c=index
    changed_a=(a+1)%group.state_count;changed_b=1-b
    source=population[index]
    target_specs=(("state",(changed_a,b,c)),("operation",(a,changed_b,c)),("joint",(changed_a,changed_b,c)))
    for kind,target in target_specs:
        natural=render_native_record(record,initial_state=target[0],
                  continuation_elements=[group.generators[0] if target[1]==0 else (group.generators[1] if len(group.generators)>1 else group.inverse(group.generators[0])),
                                         group.generators[0] if target[2]==0 else (group.generators[1] if len(group.generators)>1 else group.inverse(group.generators[0]))])
        scored_record=dict(record,counterfactual_answer=natural["answer"])
        full_delta=population[target]-source
        for support,delta in (("full_D",full_delta),("matched_q",full_delta@basis@basis.T)):
            rows.append({"record_id":record["record_id"],"family_id":record["family_id"],
                         "kind":kind,"support":support,"rank":len(h) if support=="full_D" else basis.shape[1],
                         "delta_norm":float(np.linalg.norm(delta)),"layer":layer,
                         **score_case(runtime,scored_record,patch=make_patch(layer,record,delta,readout=readout),generation=True)})
    # A genuine high-order ANOVA ablation uses only the
    # construction-fitted interaction table, not test H.
    interaction=terms[(0,1,2)][index]
    natural_record=dict(record,counterfactual_answer=record["answer"])
    for method,delta in (("three_way_ablation",-interaction),("delta_zero_sham",np.zeros_like(h))):
        rows.append({"record_id":record["record_id"],"family_id":record["family_id"],"kind":method,
                     "support":"full_D","delta_norm":float(np.linalg.norm(delta)),"layer":layer,
                     **score_case(runtime,natural_record,patch=make_patch(layer,record,delta,readout=readout),generation=True)})
    return rows


def _score_prediction(score):
    return score.get("prediction", score.get("constrained_prediction"))


def _run_reproduction_anchor(config, output, *, runtime_factory=None):
    """Runnable bounded §5.1 Engels or Wurgaft A.1–A.7 calendar anchor.

    This does not reproduce SAE discovery, all modalities, or an entire paper.
    Native common-benchmark adaptation remains a separate run identity.
    """
    from .native import make_runtime, collect_features, make_patch
    cfg = config.get("peers", config)
    name = cfg.get("anchor", "engels")
    if name not in ("engels", "wurgaft"):
        raise ValueError("Known original-task anchors: engels, wurgaft")
    prepare_run(output, config, "peer_anchor")
    model = cfg.get("model", config.get("models", [{}])[0])
    runtime = make_runtime(model, config.get("runtime", {}), runtime_factory)
    mid = model.get("model_id", model.get("id", "")) if isinstance(model, dict) else str(model)
    allowed = ("Mistral-7B", "Meta-Llama-3-8B") if name == "engels" else ("Llama-3.1-8B",)
    if not any(v in mid for v in allowed) and not cfg.get("technical_run", False):
        raise ValueError("Original-task anchor requires an original model; use adaptation for another checkpoint")
    layers = list(map(int, cfg.get("layers", [cfg.get("layer", 28)])))
    records = []
    numbers = ("one", "two", "three", "four", "five", "six", "seven")
    for i, day in enumerate(DAYS):
        for k in range(1, 8):
            text = ((f"Let's do some day of the week math. {numbers[k-1].capitalize()} days from {day} is")
                    if name == "engels" else f"Q: What day is {numbers[k-1]} days after {day}?\nA:")
            start = text.index(day)
            records.append({"text": text, "source_span": [start, start + len(day)], "state_id": i,
                            "answer_state": (i + k) % 7, "offset": k,
                            "answer": DAYS[(i + k) % 7], "candidates": list(DAYS),
                            "record_id": f"{name}/{i}/{k}", "family_id": "original_single_template"})
    readout = "span_mean" if name == "engels" else "final_token"
    arrays = collect_features(runtime, records, layers, readout)
    results = []
    trajectories=[];path_summary=[]
    for layer in layers:
        x = arrays[layer]
        labels = [r["state_id"] if name == "engels" else r["answer_state"] for r in records]
        fitted = fit_calendar_plane(x, labels) if name == "engels" else fit_manifold(x, labels)
        for idx, rec in enumerate(records):
            source = labels[idx]
            targets = [i for i in range(7) if i != source]
            if cfg.get("max_pairs"):
                targets = targets[:int(cfg["max_pairs"])]
            for target in targets:
                replacement = fitted.predict(x[idx:idx+1], [target], preserve=name != "engels")[0]
                patch = make_patch(layer, rec, replacement - x[idx], readout=readout)
                scores = runtime.score(rec["text"], rec["candidates"], patch=patch,
                                       generation=bool(cfg.get("generation", False)))
                final_target = DAYS[(target + rec["offset"]) % 7] if name == "engels" else DAYS[target]
                row = {"layer": layer, "record_id": rec["record_id"], "target": final_target,
                       "prediction": _score_prediction(scores), "correct": _score_prediction(scores) == final_target}
                logits = scores.get("candidate_first_token_logits")
                if logits is not None:
                    row["original_target_logit_difference"] = float(logits[DAYS.index(final_target)] - logits[DAYS.index(rec["answer"])])
                results.append(row)
        if name=="wurgaft" and cfg.get("paths",True):
            points=int(cfg.get("path_points",50))
            if points<3:raise ValueError("A path needs at least3 predeclared points")
            for idx,rec in enumerate(records[:int(cfg.get("path_base_prompts",16))]):
                source=int(labels[idx]);targets=[(source+step)%7 for step in range(1,7)]
                for target in targets[:int(cfg.get("path_targets_per_prompt",1))]:
                    curve=fitted.path(source,target,points=points)
                    full_source=fitted.mean+fitted.centroids[source]@fitted.basis.T
                    full_target=fitted.mean+fitted.centroids[target]@fitted.basis.T
                    for method in ("spline_preserve_off_PCA","linear_full_D_original_comparator","linear_matched_preserve_off_PCA"):
                        probabilities=[]
                        for j,t in enumerate(np.linspace(0.,1.,points)):
                            if method=="spline_preserve_off_PCA":
                                replacement=x[idx]+(curve[j]-(x[idx]-fitted.mean)@fitted.basis)@fitted.basis.T
                            elif method=="linear_full_D_original_comparator":replacement=(1-t)*full_source+t*full_target
                            else:replacement=x[idx]+((1-t)*fitted.centroids[source]+t*fitted.centroids[target]-(x[idx]-fitted.mean)@fitted.basis)@fitted.basis.T
                            scores=runtime.score(rec["text"],rec["candidates"],patch=make_patch(layer,rec,replacement-x[idx],readout=readout),generation=False)
                            token_ids=scores.get("candidate_first_token_ids")
                            if token_ids is not None and len(set(token_ids))!=len(token_ids):
                                raise ValueError("Calendar path probabilities require distinct answer first tokens")
                            probs=scores.get("candidate_first_token_probabilities");other=scores.get("other_first_token_probability")
                            if probs is None or other is None:raise ValueError("Path geometry requires actual FULL-vocabulary first-token probabilities")
                            probability=np.r_[probs,other].astype(float)
                            if not np.isclose(probability.sum(),1.,atol=2e-5):raise ValueError("Reported answer-plus-other probabilities must sum to1")
                            probabilities.append(probability)
                            trajectories.append({"record_id":rec["record_id"],"layer":layer,"source":source,"target":target,
                                                 "method":method,"step":j,"t":float(t),"first_token_probabilities":probability.tolist(),
                                                 "delta_norm":float(np.linalg.norm(replacement-x[idx])),"forward_counts":scores.get("forward_counts",{})})
                        roots=np.sqrt(np.maximum(probabilities,0.));energy=2*(points-1)*np.sum(np.diff(roots,axis=0)**2)
                        path_summary.append({"record_id":rec["record_id"],"layer":layer,"source":source,"target":target,"method":method,
                                             "discrete_Bhattacharyya_Hellinger_energy":float(energy),"points":points,
                                             "definition":"2 sum_j ||sqrt(p[j+1])-sqrt(p[j])||² / uniform_dt; finite-grid diagnostic",
                                             "scope":"observed_output_distribution_curve_energy_not_full_continuous_pullback_metric_reproduction"})
    result = {"study": "peer_original_task_anchor", "technical_run": bool(cfg.get("technical_run", False)),
              "source": PRIMARY_SOURCES[name], "checkpoint": model,
              "scope": "calendar_subexperiment_not_full_paper_reproduction",
              "implementation":{"entrypoint":"bsg_next.peers.run_reproduction_anchor",
                                "source_sha256":file_sha256(__file__),"prompt_bank_sha256":identity(records),
                                "author_code_executed":False,"author_code_commit":"not_verified",
                                "layer_convention":"raw block l-1 output for configured1-based l; author convention equivalence not certified",
                                "exact_author_prompt_equivalence":"not_certified; explicit rendered prompt bank is authoritative"},
              "original_data_fitting": "all49 explicitly rendered calendar prompts; no held-out claim or certified author-prompt equivalence",
              "adaptation_warning": "complete-sequence scores are additional endpoints; original logit metric requires returned first-token logits",
              "evaluations": results, "accuracy": float(np.mean([r["correct"] for r in results])) if results else None,
              "path_summaries":path_summary,"path_evaluations":len(trajectories),
              "path_scope":"bounded_calendar_paths_with_faithful_complement_asymmetry_and_matched_adaptation; not_complete_paper_reproduction" if name=="wurgaft" else None}
    if trajectories:
        from .native import _write_rows
        _write_rows(Path(output)/"path_evaluations.jsonl",trajectories)
    atomic_json(Path(output) / "anchor_results.json", result)
    return result


def run_reproduction_anchor(config,output,*,runtime_factory=None):
    """Resource-safe anchor, including early validation failures."""
    from .native import make_runtime
    created=[]
    def tracked(model,cfg):
        runtime=make_runtime(model,cfg,runtime_factory);created.append(runtime);return runtime
    try:return _run_reproduction_anchor(config,output,runtime_factory=tracked)
    finally:
        for runtime in created:
            if hasattr(runtime,"close"):runtime.close()


def run_peers(config, output, *, runtime_factory=None, arrays=None):
    cfg = config.get("peers", {})
    if cfg.get("anchor"):
        return run_reproduction_anchor(config, output, runtime_factory=runtime_factory)
    if arrays is not None:
        prepare_run(output, config, "peer_array_fits")
        x, states = np.asarray(arrays["train"]), np.asarray(arrays["states"])
        y, targets = np.asarray(arrays["test"]), np.asarray(arrays["target_states"])
        methods = {"calendar_plane_matched": fit_calendar_plane(x, states), "spline_matched": fit_manifold(x, states)}
        result = {"study": "peer_common_benchmark_arrays", "technical_run": bool(cfg.get("technical_run", False)),
                  "methods": {name: {"metadata": model.metadata,
                                     "predictions": model.predict(y, targets).tolist()} for name, model in methods.items()}}
        atomic_json(Path(output) / "peers.json", result); return result
    from .native import run_native
    adapted = copy.deepcopy(config)
    adapted.setdefault("native", {})["include_peers"] = True
    return run_native(adapted, output, runtime_factory=runtime_factory)
