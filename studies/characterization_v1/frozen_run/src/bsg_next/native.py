"""Prospective native counterfactuals with construction-only fits and sealed choices.

All scientific model scoring is deferred until explicitly invoked. A supplied
runtime_factory enables deterministic CPU/mock end-to-end verification without
downloading weights. Technical runs are permanently marked in their outputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import copy
import json
import re
import time
import numpy as np
from scipy import stats

from .groups import get_group
from .estimators import (select_action_ridge, fit_procrustes, select_retention,
                         support_diagnostics, RETENTION_GRID)
from .statistics import holm, bootstrap_sensitivity, binomial_interval
from .io import atomic_json, read_json, identity, prepare_run, file_sha256
from .peers import pca, fit_calendar_plane, fit_manifold

INFORMED = ("centroid", "context_shrinkage", "affine", "procrustes", "local_displacement")
RANDOM_SEEDS = (2027101701, 2027101702, 2027101703, 2027101704, 2027101705)
BANDWIDTHS = (.125, .25, .5, 1., 2., 4., 8., np.inf)


def make_runtime(model, runtime_config=None, runtime_factory=None):
    cfg = dict(runtime_config or {})
    if isinstance(model, str): model = {"model_id": model}
    if runtime_factory is not None:
        return runtime_factory(model, cfg)
    from .hf import runtime_from_config
    mid = model.get("model_id", model.get("id", model.get("hf_id")))
    if not mid: raise ValueError("Pinned model_id is required")
    revision = model.get("revision")
    if not revision: raise ValueError("Pin the exact checkpoint revision before native collection")
    return runtime_from_config(dict(model,id=mid,revision=revision),cfg)


def collect_features(runtime, records, layers, readout="carrier_span", cache_dir=None):
    """Collect only requested readouts, reusing content-addressed native caches."""
    records=list(records); layers=list(map(int,layers))
    key=identity({"records":records,"layers":layers,"readout":readout,
                  "model":getattr(runtime,"model_id",str(type(runtime))),
                  "revision":getattr(runtime,"revision",None),"runtime_class":str(type(runtime)),
                  "runtime_identity":runtime.identity() if hasattr(runtime,"identity") else None,
                  "backend_source_sha256":file_sha256(Path(__file__).parent/"hf.py")})
    path=Path(cache_dir)/f"{key}.npz" if cache_dir else None
    if path and path.exists():
        manifest=path.with_suffix(".json")
        if not manifest.exists() or read_json(manifest).get("sha256")!=file_sha256(path):
            raise ValueError("Native feature cache checksum mismatch")
        with np.load(path,allow_pickle=False) as loaded:
            return {layer:loaded[str(layer)] for layer in layers}
    result={}; covered=np.zeros(len(records),bool)
    for batch in runtime.features(records,layers,readout=readout):
        indices, values=batch[:2]; indices=np.asarray(indices,int)
        if np.any(covered[indices]): raise ValueError("Feature runtime emitted duplicate rows")
        covered[indices]=True
        for layer in layers:
            array=np.asarray(values[layer],np.float32)
            if layer not in result: result[layer]=np.empty((len(records),array.shape[1]),np.float32)
            result[layer][indices]=array
    if not covered.all(): raise ValueError("Incomplete native feature extraction")
    if any(not np.isfinite(v).all() for v in result.values()): raise ValueError("Nonfinite native activations")
    if path:
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_suffix(".tmp.npz")
        np.savez(temporary,**{str(k):v for k,v in result.items()}); temporary.replace(path)
        atomic_json(path.with_suffix(".json"),{"identity":key,"sha256":file_sha256(path),"rows":len(records),"readout":readout,"layers":layers})
    return result


def make_patch(layer, record, delta=None, *, replacement_mean=None, readout="carrier_span", dose=1., mode="delta"):
    from .hf import PatchSpec
    span=record.get("carrier_span") if readout=="carrier_span" else record.get("source_span") if readout=="span_mean" else None
    return PatchSpec(layer=int(layer), span=tuple(span) if span else None,
                     positions=None if span else [-1], delta=delta,
                     replacement_mean=replacement_mean, dose=float(dose),mode=mode)


def _domain(name):
    return "D4_free8" if str(name).lower() in ("d4_free","d4_native","d4") else str(name)


def _key(element):
    return tuple(element) if isinstance(element,list) else element


def _site_state(record,group,site):
    intermediate=int(record["intermediate_state"])
    return group.apply_word(intermediate,record["continuation_elements"]) if site=="answer" else intermediate


def _site_edit(record,group,site):
    g=group.normalize(record["edit_element"])
    if site=="answer":
        v=group.evaluate_word(record["continuation_elements"])
        return group.compose(v,group.compose(g,group.inverse(v)))
    return g


def _record_with_edit(record,group,desired,site):
    from .data import render_native_record
    desired=group.normalize(desired)
    if site=="answer":
        v=group.evaluate_word(record["continuation_elements"])
        desired=group.compose(group.inverse(v),group.compose(desired,v))
    return render_native_record(record,edit_element=desired)


def _counterpart(record):
    from .data import render_native_record
    return render_native_record(record,initial_state=int(record["counterfactual_initial_state"]))


def _all_pairs(records, group, site):
    pairs={}
    for generator in group.generators:
        edits=[_record_with_edit(r,group,generator,site) for r in records]
        pairs[generator]=[_counterpart(r) for r in edits]
    return pairs


@dataclass
class LocalDisplacement:
    residuals: dict
    displacements: dict
    means: np.ndarray
    bandwidth: dict
    median_distance: float
    validation_scores: dict

    @classmethod
    def fit(cls,z,states,targets,zv,sv,targets_v,means):
        residual=z-means[states]
        distances=[]
        for i in range(len(means)):
            r=residual[states==i]
            if len(r)>1:
                d=np.linalg.norm(r[:,None]-r[None],axis=2)
                positive=d[np.triu_indices(len(r),1)]; distances.extend(positive[positive>1e-12].tolist())
        median=float(np.median(distances)) if distances else 0.
        bandwidth={}; val_scores={}
        temporary=cls({i:residual[states==i] for i in range(len(means))},
                      {g:{i:(y-z)[states==i] for i in range(len(means))} for g,y in targets.items()},
                      means,{},median,{})
        for g in targets:
            scores=[]
            for multiplier in BANDWIDTHS:
                sigma=median*multiplier if median and np.isfinite(multiplier) else np.inf
                temporary.bandwidth[g]=sigma
                pred=np.stack([temporary.predict(v,int(i),g) for v,i in zip(zv,sv)])
                scores.append(float(np.mean(np.sum((pred-targets_v[g])**2,1))))
            idx=min(range(len(scores)),key=lambda j:(scores[j],-j))
            bandwidth[g]=median*BANDWIDTHS[idx] if median and np.isfinite(BANDWIDTHS[idx]) else np.inf
            val_scores[str(g)]={str(m):s for m,s in zip(BANDWIDTHS,scores)}
        temporary.bandwidth=bandwidth; temporary.validation_scores=val_scores
        return temporary

    def predict(self,z,state,g):
        r=self.residuals[int(state)]; d=self.displacements[g][int(state)]
        if not len(r): raise ValueError("No construction pairs for oracle source state")
        sigma=self.bandwidth[g]
        if np.isinf(sigma): weights=np.ones(len(r))/len(r)
        else:
            logits=-np.sum((r-(z-self.means[int(state)]))**2,1)/(2*sigma*sigma)
            weights=np.exp(logits-logits.max()); weights/=weights.sum()
        return z+weights@d


@dataclass
class NativeFit:
    group: object
    mean: np.ndarray
    basis: np.ndarray
    centroids: np.ndarray
    operators: dict
    affine: dict
    procrustes: dict
    retention: dict
    local: LocalDisplacement
    words: dict
    metadata: dict
    peer_models: dict

    def encode(self,x): return (np.asarray(x)-self.mean)@self.basis

    def _chain(self,z,g,models):
        for primitive in self.words[g]: z=models[primitive].predict(np.atleast_2d(z))[0]
        return z

    def predict(self,method,h,state,g):
        g=self.group.normalize(g); state=int(state); z=self.encode(h)
        target=self.group.apply(g,state)
        if method=="group": result=self._chain(z,g,self.operators)
        elif method=="affine": result=self._chain(z,g,self.affine)
        elif method=="procrustes": result=self._chain(z,g,self.procrustes)
        elif method in ("centroid","context_shrinkage","centroid_decoded"):
            if method=="centroid_decoded":
                state=int(np.argmin(np.sum((self.centroids-z)**2,1))); target=self.group.apply(g,state)
            alpha=1. if method!="context_shrinkage" else self.retention.get(g,1.)
            result=self.centroids[target]+alpha*(z-self.centroids[state])
        elif method=="local_displacement":
            result=z.copy(); current=state
            for primitive in self.words[g]:
                result=self.local.predict(result,current,primitive); current=self.group.apply(primitive,current)
        elif method.startswith("calendar_plane"):
            return self.peer_models["calendar"].predict(np.atleast_2d(h),[target],preserve=method.endswith("matched"))[0]
        elif method=="spline_matched":
            return self.peer_models["spline"].predict(np.atleast_2d(h),[target],preserve=True)[0]
        else: raise ValueError(f"Unknown fitted native method {method}")
        return np.asarray(h)+(result-z)@self.basis.T


def _generator_words(group):
    paths={group.identity:()}; frontier=[group.identity]
    while frontier:
        current=frontier.pop(0)
        for g in group.generators:
            target=group.compose(g,current)
            if target not in paths: paths[target]=paths[current]+(g,);frontier.append(target)
    if len(paths)!=group.order: raise ValueError("Declared generators do not generate the group")
    return paths


def fit_native(train_records, validation_records, x, xv, targets, targets_v, group, *, rank=16, site="intermediate", include_peers=False):
    states=np.asarray([_site_state(r,group,site) for r in train_records],int)
    sv=np.asarray([_site_state(r,group,site) for r in validation_records],int)
    mean,basis,pinfo=pca(x,rank); z=(x-mean)@basis; zv=(xv-mean)@basis
    centroids=[]
    for i in range(group.state_count):
        if not np.any(states==i): raise ValueError(f"No native construction source state {i}")
        centroids.append(z[states==i].mean(0))
    means=np.stack(centroids)
    yz={g:(y-mean)@basis for g,y in targets.items()}; yv={g:(y-mean)@basis for g,y in targets_v.items()}
    tr={g:(z,y) for g,y in yz.items()}; va={g:(zv,y) for g,y in yv.items()}
    operators,ometa=select_action_ridge(tr,va,affine=False)
    affine,ameta=select_action_ridge(tr,va,affine=True)
    proc={g:fit_procrustes(z,y) for g,y in yz.items()}
    retention={}; retention_scores={}
    for g in group.elements:
        # Generator losses select retention; other edits use nearest primitive
        # only in optional composite stages and record the fixed mean rule.
        if g in yv:
            tm=means[np.asarray([group.apply(g,int(i)) for i in sv])]
            retention[g],retention_scores[str(g)]=select_retention(zv,yv[g],means[sv],tm)
        else: retention[g]=1.
    local=LocalDisplacement.fit(z,states,yz,zv,sv,yv,means)
    decoded=np.argmin(np.sum((zv[:,None]-means[None])**2,2),1)
    peer={}; unsupported={}
    if include_peers:
        if group.name=="C7":
            for name, fitter in (("calendar",lambda:fit_calendar_plane(x,states)),
                                 ("spline",lambda:fit_manifold(x,states,pca_rank=rank))):
                try:peer[name]=fitter()
                except ValueError as error:unsupported[name]="N_A_unidentifiable_construction_fit: "+str(error)
        else: unsupported={"calendar_plane":"N_A_not_a_registered_D4_calendar_method",
                           "spline":"N_A_cyclic_spline_does_not_define_D4_geometry"}
    metadata={"pca":pinfo,"site_variable":site,"generator_ridge":ometa,"affine":ameta,
              "decoder_validation_accuracy":float(np.mean(decoded==sv)),"retention_grid":list(RETENTION_GRID),
              "retention_selection":retention_scores,"local_bandwidth_grid":[str(v) for v in BANDWIDTHS],
              "local_median_positive_training_distance":local.median_distance,
              "local_validation_losses":local.validation_scores,
              "local_information":"oracle source identity + native construction paired residual displacements",
              "capacity":{"group_coefficients":len(operators)*basis.shape[1]**2,
                          "centroid_coordinates":means.size,"local_training_pairs":len(z)*len(yz)},
              "operator_diagnostics":{str(g):m.diagnostics for g,m in operators.items()},
              "peer_metadata":{k:v.metadata for k,v in peer.items()},"unsupported_peers":unsupported,
              "late_information":"answer carrier uses conjugated edit v g v^-1 and answer-state labels" if site=="answer" else None}
    return NativeFit(group,mean,basis,means,operators,affine,proc,retention,local,_generator_words(group),metadata,peer)


def _normalized_answer(value):
    return re.sub(r"\s+","",str(value or "").strip()).rstrip(".").lower()


def score_case(runtime, record, *, patch=None, generation=True):
    out=runtime.score(record["text"],record["candidates"],patch=patch,generation=generation,max_new_tokens=48)
    scores=np.asarray(out.get("candidate_logprobs",[]),float)
    prediction=out.get("prediction",out.get("constrained_prediction"))
    generated=out.get("generated_text")
    if generation and generated is None: raise ValueError("Primary exact-answer endpoint requires actual generated_text")
    answer=generated if generation else prediction
    target=record.get("counterfactual_answer",record["answer"])
    idx=record["candidates"].index(target)
    margin=float(scores[idx]-np.max(np.delete(scores,idx))) if len(scores)>1 else None
    return {"prediction":prediction,"generated_text":generated,"correct":_normalized_answer(answer)==_normalized_answer(target),
            "constrained_correct":_normalized_answer(prediction)==_normalized_answer(target),
            "sequence_margin":margin,"candidate_logprobs":scores.tolist(),
            "first_token_probabilities":out.get("candidate_first_token_probabilities"),
            "other_first_token_probability":out.get("other_first_token_probability"),
            "forward_counts":out.get("forward_counts",{})}


def evaluate_methods(runtime, records, fit, features, counterparts, layer, *, readout="carrier_span", methods=None,
                     dose=1., generation=True, random_seeds=RANDOM_SEEDS, site="intermediate"):
    default_methods=methods is None
    methods=list(methods) if methods is not None else ["group",*INFORMED,"centroid_decoded","delta_sham","legacy_mean_sham","natural_donor",*[f"random_{s}" for s in random_seeds]]
    if default_methods:
        if "calendar" in fit.peer_models:methods += ["calendar_plane_matched","calendar_plane_original_ablation"]
        if "spline" in fit.peer_models:methods += ["spline_matched"]
    random={}
    for seed in random_seeds:
        rng=np.random.default_rng(int(seed)); q=fit.basis.shape[1]
        random[f"random_{seed}"]=np.linalg.qr(rng.normal(size=(features.shape[1],q)))[0][:,:q]
    rows=[]
    for n,(record,h,donor) in enumerate(zip(records,features,counterparts)):
        state=_site_state(record,fit.group,site); g=_site_edit(record,fit.group,site)
        for method in methods:
            if method=="delta_sham": delta=np.zeros_like(h); patch=make_patch(layer,record,delta,readout=readout,dose=dose)
            elif method=="legacy_mean_sham":
                delta=np.zeros_like(h);patch=make_patch(layer,record,replacement_mean=h,readout=readout,mode="mean_overwrite")
            elif method=="natural_donor":
                delta=((donor-h)@fit.basis)@fit.basis.T;patch=make_patch(layer,record,delta,readout=readout,dose=dose)
            elif method in random:
                b=random[method];delta=((donor-h)@b)@b.T;patch=make_patch(layer,record,delta,readout=readout,dose=dose)
            else:
                replacement=fit.predict(method,h,state,g);delta=replacement-h
                patch=make_patch(layer,record,delta,readout=readout,dose=dose)
            scored=score_case(runtime,record,patch=patch,generation=generation)
            rows.append({"record_id":record.get("record_id",str(n)),"family_id":record["family_id"],
                         "scenario_id":record.get("scenario_id"),"paraphrase_id":record.get("paraphrase_id"),
                         "word_split":record.get("word_split"),"order_disagreement":record.get("order_disagreement"),
                         "selection_eligible":record.get("selection_eligible",False),"site_variable":site,
                         "method":method,"layer":int(layer),"readout":readout,"dose":float(dose),
                         "target":record["counterfactual_answer"],"delta_norm":float(np.linalg.norm(delta)),
                         "full_D_exception":method=="legacy_mean_sham",**scored})
    return rows


def native_decisions(family_contrasts, *, alpha=.05, practical=.05, bootstrap=0, seed=42):
    """Four two-sided t/Holm contrasts; five-point estimate gate, not H0=.05."""
    count=len(family_contrasts); summaries={}; pvalues={}
    for key,values in family_contrasts.items():
        x=np.asarray(values,float)
        if x.ndim!=1 or len(x)<2 or not np.isfinite(x).all(): raise ValueError("Native inference requires >=2 finite family contrasts")
        se=float(x.std(ddof=1)/np.sqrt(len(x))); mean=float(x.mean()); valid=se>0
        critical=float(stats.t.ppf(1-alpha/(2*count),len(x)-1))
        ci=[mean-critical*se,mean+critical*se] if valid else [None,None]
        p=float(2*stats.t.sf(abs(mean/se),len(x)-1)) if valid else 1.
        summaries[key]={"mean":mean,"se":se,"families":len(x),"df":len(x)-1,"interval":ci,
                        "interval_level":1-alpha/count,"valid_inference":valid,"planning_point_threshold":practical}
        if bootstrap: summaries[key]["bootstrap_sensitivity"]=bootstrap_sensitivity(x,replicates=bootstrap,seed=seed)
        pvalues[key]=p
    tests=holm(pvalues,alpha)
    for key,row in summaries.items():
        passed=bool(row["valid_inference"] and row["mean"]>=practical and row["interval"][0]>0 and tests[key]["reject"])
        row["status"]="practical_advantage" if passed else "inconclusive" if not row["valid_inference"] or row["interval"][1]>=practical else "practical_target_excluded"
    return {"method":"paired_family_t_Holm_with_Bonferroni_t_intervals","alpha":alpha,"tests":tests,
            "contrasts":summaries,"all_four_pass":len(summaries)==4 and all(v["status"]=="practical_advantage" for v in summaries.values()),
            "assumption":"independently generated scenario families; approximate finite-family t reference"}


def _native_power_pass(x,alpha=.05,practical=.05):
    means=x.mean(1); se=x.std(1,ddof=1)/np.sqrt(x.shape[1]);valid=se>0
    with np.errstate(divide="ignore",invalid="ignore"):
        p=2*stats.t.sf(np.abs(means/se),x.shape[1]-1)
    p[~valid|~np.isfinite(p)]=1.
    order=np.argsort(p,1,kind="stable"); sortedp=np.take_along_axis(p,order,1)
    adj=np.minimum(1,np.maximum.accumulate(sortedp*np.arange(4,0,-1),axis=1)); original=np.empty_like(adj)
    np.put_along_axis(original,order,adj,1)
    lower=means-stats.t.ppf(1-alpha/8,x.shape[1]-1)*se
    return valid & (means>=practical) & (lower>0) & (original<=alpha)


def native_power(config, pilot=None):
    cfg=config.get("native",{}).get("power",{}); rng=np.random.default_rng(int(cfg.get("seed",2027101742)))
    alpha=float(cfg.get("alpha",config.get("native",{}).get("alpha",.05)))
    point=float(cfg.get("point_threshold",config.get("native",{}).get("primary_observed_margin",.05)))
    marginal_target=float(cfg.get("marginal_target",config.get("native",{}).get("power_marginal_target",.90)))
    joint_target=float(cfg.get("joint_target",config.get("native",{}).get("power_joint_target",.80)))
    reps=int(cfg.get("replicates",2000)); sizes=list(map(int,cfg.get("test_families",[32,48,64])))
    if pilot is not None:
        pilot=np.asarray(pilot,float)
        if pilot.ndim!=2 or pilot.shape[1]!=4 or len(pilot)<2: raise ValueError("Excluded pilot family×four-cell contrasts are required")
        sd=pilot.std(0,ddof=1);source="excluded_pilot_variation"
        # Constant pilot effects do not justify a zero variance population.
        sd=np.maximum(sd,float(cfg.get("minimum_sd",.025)))
    else:
        sd=np.asarray(cfg.get("assumed_sd",[.10]*4),float);source="explicit_assumptions_not_measured_pilot"
    if sd.shape!=(4,) or np.any(sd<=0): raise ValueError("Four positive planning SDs required")
    alternatives=sorted(set([float(cfg.get("alternative",.10)),float(cfg.get("sensitivity",.05)),0.]))
    rows=[]
    for n in sizes:
        for correlation in cfg.get("correlations",[0.,.5,.8]):
            covariance=np.full((4,4),float(correlation));np.fill_diagonal(covariance,1.)
            chol=np.linalg.cholesky(covariance)
            for stress in cfg.get("stress",["normal","t5"]):
                noise=rng.normal(size=(reps,n,4))@chol.T
                if stress=="t5":noise*=np.sqrt(3/rng.chisquare(5,size=(reps,n,1)))
                noise*=sd
                for alternative in alternatives:
                    values=_native_power_pass(noise+alternative,alpha=alpha,practical=point)
                    rows.append({"families":n,"correlation":float(correlation),"stress":stress,"alternative":alternative,
                                 "marginal":[binomial_interval(values[:,j].sum(),reps) for j in range(4)],
                                 "joint":binomial_interval(np.all(values,1).sum(),reps)})
    planning=float(cfg.get("alternative",.10));passing=[]
    for n in sizes:
        considered=[r for r in rows if r["families"]==n and r["alternative"]==planning]
        if all(all(v["interval"][0]>=marginal_target for v in r["marginal"]) and r["joint"]["interval"][0]>=joint_target for r in considered):passing.append(n)
    return {"study":"native_full_decision_power","source":source,"sd":sd.tolist(),"replicates":reps,
            "selected_families":min(passing) if passing else None,"planning_alternative":planning,
            "point_gate":point,"interval_lower_gate":0.,"marginal_target":marginal_target,"joint_target":joint_target,
            "warning":"true .05 cannot provide .90 power for a point>=.05 gate",
            "scenarios":rows,"status":"gate_met" if passing else "insufficient_planning_power"}


def _models(config):
    models=config.get("native",{}).get("models",config.get("models",[]))
    if not models: raise ValueError("Configure pinned native model checkpoints")
    registry={m.get("id",m.get("model_id")):m for m in config.get("models",[]) if isinstance(m,dict)}
    return [m if isinstance(m,dict) else dict(registry.get(m,{"model_id":m})) for m in models]


def normalized_config(config):
    out=copy.deepcopy(config);cfg=out.setdefault("native",{})
    cfg.setdefault("families",{k:cfg.get(f"{k}_families",v) for k,v in (("train",16),("validation",8),("test",32),("pilot",12))})
    cfg.setdefault("rank",cfg.get("q",16));cfg.setdefault("doses",cfg.get("dose_grid",[0,.25,.5,1,2]))
    cfg.setdefault("include_peers",cfg.get("include_peer_methods",False))
    cfg.setdefault("technical_run",out.get("technical_test",False))
    cfg.setdefault("sites",{"early":cfg.get("early_depth_candidates",[.2,.4,.6]),"late":cfg.get("late_depth_candidates",[.8])})
    power=cfg.setdefault("power",{})
    power.setdefault("replicates",cfg.get("power_replicates",2000));power.setdefault("alternative",cfg.get("power_alternative",.10))
    power.setdefault("sensitivity",cfg.get("power_sensitivity",.05))
    secondary=cfg.setdefault("secondary",{})
    secondary.setdefault("operation_joint",bool(cfg.get("include_operations",False) or cfg.get("include_joint_variables",False)))
    secondary.setdefault("pathways",dict(cfg.get("pathway",{}),enabled=True) if cfg.get("include_pathway",False) else {})
    if cfg.get("include_pathway") and not secondary["pathways"]:secondary["pathways"]={"enabled":True}
    return out


def _family_subset(records, count):
    if count is None:return records
    names=sorted({r["family_id"] for r in records})[:int(count)]
    return [r for r in records if r["family_id"] in names]


def _build(config,domain,split,*,test_families=None):
    from .data import build_native_dataset
    cfg=config.get("native",{}); counts=cfg.get("families",{})
    requested=int(test_families if test_families is not None else counts.get("test",32))
    pool=next((n for n in (32,48,64) if n>=requested),None)
    if pool is None:raise ValueError("Native family count exceeds the prespecified64-family pool")
    records=build_native_dataset(domain,seed=int(cfg.get("seed",2027101700)),
                                 test_families=pool,
                                 pilot=split=="pilot",factorial=bool(cfg.get("factorial",False)),splits=[split])
    return _family_subset(records,requested if split=="test" else counts.get(split,12 if split=="pilot" else None))


def _sites(config,model,runtime):
    cfg=config.get("native",{}); raw=cfg.get("sites",{})
    mid=model.get("model_id",model.get("id")); raw=raw.get(mid,raw)
    count=model.get("num_hidden_layers",model.get("n_layers",getattr(runtime,"n_layers",None)))
    if count is None and not raw: raise ValueError("Provide known n_layers or explicit native site candidates")
    early=raw.get("early",[max(1,round(int(count)*.55))] if count else [])
    late=raw.get("late",[max(1,round(int(count)*.90))] if count else [])
    def resolve(v):
        if isinstance(v,float) and 0<v<=1:
            if count is None:raise ValueError("Relative sites require the pinned model's layer count")
            return min(int(count),max(1,round(int(count)*v)))
        value=int(v)
        if value<1 or (count is not None and value>int(count)):raise ValueError("Native site outside model blocks")
        return value
    return [(resolve(v),"carrier_span","intermediate") for v in early]+[(resolve(v),"final_token","answer") for v in late]


def _clean_evaluation(runtime,records,generation=True):
    out=[]
    for r in records:
        natural=dict(r,counterfactual_answer=r["answer"])
        out.append({"record_id":r["record_id"],"family_id":r["family_id"],"word_split":r["word_split"],
                    **score_case(runtime,natural,generation=generation)})
    return out


def _competence(rows,threshold):
    rates={word:float(np.mean([r["correct"] for r in rows if r["word_split"]==word]))
           for word in ("development","transfer") if any(r["word_split"]==word for r in rows)}
    return {"rates":rates,"threshold":threshold,"passed":bool(rates and all(v>=threshold for v in rates.values())),
            "no_filtering_of_incorrect_test_items":True}


def _accuracy(rows,method,selection=False):
    values=[r["correct"] for r in rows if r["method"]==method and (not selection or r["selection_eligible"])]
    return float(np.mean(values)) if values else -np.inf


def _family_contrasts(rows,baseline):
    pairs={}
    for r in rows:
        if r["method"] in ("group",baseline):
            pairs.setdefault((r["family_id"],r["record_id"]),{})[r["method"]]=float(r["correct"])
    output={}
    for (family,_),v in pairs.items():
        if "group" not in v or baseline not in v:raise ValueError("Incomplete paired native baseline scoring")
        output.setdefault(family,[]).append(v["group"]-v[baseline])
    return {k:float(np.mean(v)) for k,v in sorted(output.items())}


def _write_rows(path,rows):
    from .io import jsonable
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(".tmp.jsonl")
    with temporary.open("w") as handle:
        for row in rows:handle.write(json.dumps(jsonable(row),sort_keys=True,allow_nan=False)+"\n")
    temporary.replace(path)


def _fit_site(runtime,train,validation,group,layer,readout,site,cfg,cache):
    original_train=collect_features(runtime,train,[layer],readout,cache)[layer]
    original_val=collect_features(runtime,validation,[layer],readout,cache)[layer]
    train_pairs=_all_pairs(train,group,site);val_pairs=_all_pairs(validation,group,site)
    target={g:collect_features(runtime,rows,[layer],readout,cache)[layer] for g,rows in train_pairs.items()}
    target_val={g:collect_features(runtime,rows,[layer],readout,cache)[layer] for g,rows in val_pairs.items()}
    fit=fit_native(train,validation,original_train,original_val,target,target_val,group,
                   rank=int(cfg.get("rank",16)),site=site,include_peers=bool(cfg.get("include_peers",False)))
    return fit


def _rank_candidates(cfg):
    values=cfg.get("rank_sensitivity",[])
    if not isinstance(values,(list,tuple)):raise ValueError("native.rank_sensitivity must be a list of positive integers")
    if any(isinstance(q,bool) or int(q)!=q or int(q)<1 for q in values):raise ValueError("Native sensitivity ranks must be positive integers")
    return sorted({int(q) for q in values if int(q)!=int(cfg.get("rank",16))})


def _baseline_names(fit):
    return list(INFORMED)+(["calendar_plane_matched"] if "calendar" in fit.peer_models else [])+(["spline_matched"] if "spline" in fit.peer_models else [])


def _prepare_rank_sensitivity(runtime,train,validation,group,selected,cfg,cache,output):
    """Fit every requested rank at the primary's fixed sites before any test H.

    Ranks are reported as a prespecified secondary panel, not selected to
    replace the primary dimension. Only baseline selection uses familiar
    validation behavior. Peer circle direction fits retain their own declared
    information/capacity and rank-two support rather than pretending to be the
    same PCA axes as a rank-two action fit.
    """
    metadata={"scope":"secondary_prespecified_rank_panel","primary_rank":int(cfg.get("rank",16)),
              "site_selection":"reuse_primary_rank_sites_without_a_rank_specific_site_search",
              "frozen_before_test":True,"test_endpoint":"all_length6_to8_transfer_recipients",
              "inference":"descriptive_secondary; primary_power_and_four_contrasts_unchanged","ranks":{}}
    fitted={};generation=bool(cfg.get("generation",True))
    for q in _rank_candidates(cfg):
        rank_cfg=dict(cfg,rank=q);metadata["ranks"][str(q)]={};fitted[str(q)]={}
        for site,primary in selected.items():
            layer=primary["layer"];readout=primary["readout"]
            fit=_fit_site(runtime,train,validation,group,layer,readout,site,rank_cfg,cache)
            h=collect_features(runtime,validation,[layer],readout,cache)[layer]
            donor=collect_features(runtime,[_counterpart(r) for r in validation],[layer],readout,cache)[layer]
            alternatives=_baseline_names(fit)
            methods=["group",*alternatives]
            rows=evaluate_methods(runtime,validation,fit,h,donor,layer,readout=readout,
                                  methods=methods,generation=generation,site=site)
            baseline=max(alternatives,key=lambda method:(_accuracy(rows,method,selection=True),-alternatives.index(method)))
            entry={"requested_rank":q,"actual_rank":int(fit.basis.shape[1]),"layer":layer,"readout":readout,
                   "site_variable":site,"baseline":baseline,"baseline_validation_accuracy":_accuracy(rows,baseline,selection=True),
                   "group_validation_accuracy":_accuracy(rows,"group",selection=True),"fit":fit.metadata,
                   "primary_site_identity":identity({k:primary[k] for k in ("layer","readout","site_variable")}),
                   "validation_selection_uses_transfer":False}
            metadata["ranks"][str(q)][site]=entry;fitted[str(q)][site]=fit
            _write_rows(Path(output)/"rank_sensitivity"/f"q{q}"/f"validation_{site}.jsonl",
                         [dict(row,requested_rank=q,actual_rank=fit.basis.shape[1],secondary=True) for row in rows])
    return metadata,fitted


def _restore_rank_sensitivity(runtime,train,validation,group,selection,cfg,cache):
    frozen=selection.get("rank_sensitivity",{})
    if set(frozen.get("ranks",{}))!={str(q) for q in _rank_candidates(cfg)}:
        raise ValueError("Native rank panel differs from its frozen development specification")
    result={}
    for q,sites in frozen.get("ranks",{}).items():
        result[q]={}
        for site,entry in sites.items():
            primary=selection["selected"][site]
            if (entry["layer"],entry["readout"])!=(primary["layer"],primary["readout"]):
                raise ValueError("Rank sensitivity site no longer matches the frozen primary site")
            fit=_fit_site(runtime,train,validation,group,entry["layer"],entry["readout"],site,dict(cfg,rank=int(q)),cache)
            if identity(fit.metadata)!=identity(entry["fit"]):raise ValueError("Native rank sensitivity construction fit changed after freeze")
            result[q][site]=fit
    return result


def run_rank_sensitivity(runtime,records,selection,rank_fits,output,cache=None,*,generation=True):
    """Score frozen rank alternatives on every long-word recipient at fixed sites."""
    records=[r for r in records if r.get("word_split")=="transfer"]
    if not records:raise ValueError("Rank sensitivity requires the registered long-word transfer panel")
    metadata=selection["rank_sensitivity"];result={"scope":metadata["scope"],"primary_rank":metadata["primary_rank"],
             "primary_power_or_decision_changed":False,"test_endpoint":metadata["test_endpoint"],
             "recipients_per_site_rank":len(records),"ranks":{},"evaluations":0}
    for q,sites in metadata["ranks"].items():
        result["ranks"][q]={}
        for site,entry in sites.items():
            fit=rank_fits[q][site];layer=entry["layer"];readout=entry["readout"]
            h=collect_features(runtime,records,[layer],readout,cache)[layer]
            donor=collect_features(runtime,[_counterpart(r) for r in records],[layer],readout,cache)[layer]
            peer_methods=(["calendar_plane_matched"] if "calendar" in fit.peer_models else [])+(["spline_matched"] if "spline" in fit.peer_models else [])
            methods=list(dict.fromkeys(["group",entry["baseline"],"centroid_decoded","natural_donor","delta_sham",*peer_methods]))
            scored=evaluate_methods(runtime,records,fit,h,donor,layer,readout=readout,
                                    methods=methods,generation=generation,site=site)
            rows=[dict(row,requested_rank=int(q),actual_rank=fit.basis.shape[1],secondary=True,
                       primary_site_identity=entry["primary_site_identity"]) for row in scored]
            _write_rows(Path(output)/f"q{q}"/f"test_{site}.jsonl",rows)
            contrasts=_family_contrasts(rows,entry["baseline"])
            result["ranks"][q][site]={"requested_rank":int(q),"actual_rank":fit.basis.shape[1],
                         "layer":layer,"readout":readout,"baseline":entry["baseline"],"recipients":len(records),
                         "evaluations":len(rows),"methods":methods,"accuracy":{method:_accuracy(rows,method) for method in methods},
                         "family_contrasts":contrasts,"mean_gain":float(np.mean(list(contrasts.values()))),
                         "fit_identity":identity(fit.metadata),"inferential_status":"secondary_descriptive_no_confirmatory_claim"}
            result["evaluations"]+=len(rows)
    atomic_json(Path(output)/"rank_sensitivity_results.json",result);return result


def _prepare_cell(config,model,domain,runtime,output,cache):
    cfg=config.get("native",{}); group=get_group(domain); generation=bool(cfg.get("generation",True))
    train=[r for r in _build(config,domain,"train") if r["fit_eligible"]]
    validation=_build(config,domain,"validation")
    selection=[r for r in validation if r["selection_eligible"]]
    if not train or not selection:raise ValueError("Native construction and familiar validation rows are required")
    clean=_clean_evaluation(runtime,validation,generation)
    competence=_competence(clean,float(cfg.get("competence_threshold",.8)))
    _write_rows(Path(output)/"validation_clean.jsonl",clean)
    if not competence["passed"] and not cfg.get("technical_run",False):
        return {"status":"clean_competence_failed","competence":competence},None
    candidates=[];fits={}
    for layer,readout,site in _sites(config,model,runtime):
        fit=_fit_site(runtime,train,selection,group,layer,readout,site,cfg,cache)
        x=collect_features(runtime,selection,[layer],readout,cache)[layer]
        donor=collect_features(runtime,[_counterpart(r) for r in selection],[layer],readout,cache)[layer]
        rows=evaluate_methods(runtime,selection,fit,x,donor,layer,readout=readout,
                              generation=generation,site=site,random_seeds=cfg.get("random_seeds",RANDOM_SEEDS))
        score=_accuracy(rows,"group",selection=True)
        key=f"{site}:{layer}:{readout}";fits[key]=fit
        alternatives=_baseline_names(fit)
        best=max(alternatives,key=lambda m:(_accuracy(rows,m,selection=True),-alternatives.index(m)))
        candidates.append({"key":key,"layer":layer,"readout":readout,"site_variable":site,
                           "group_validation_accuracy":score,"baseline":best,
                           "baseline_validation_accuracy":_accuracy(rows,best,selection=True),"fit":fit.metadata})
        _write_rows(Path(output)/f"validation_{site}_{layer}.jsonl",rows)
    selected={}
    for site in ("intermediate","answer"):
        eligible=[v for v in candidates if v["site_variable"]==site]
        if not eligible:raise ValueError("Both earlier and late candidate site sets are required")
        selected[site]=max(eligible,key=lambda r:(r["group_validation_accuracy"],-r["layer"]))
    metadata={"status":"selection_frozen","competence":competence,"selected":selected,
              "candidates":candidates,"train_ids_sha256":identity([r["record_id"] for r in train]),
              "validation_ids_sha256":identity([r["record_id"] for r in validation]),
              "selection_excludes_transfer_outcomes":True}
    secondary=cfg.get("secondary",{})
    if secondary.get("operation_joint"):
        metadata["operation_alignment"]=_prepare_operation_alignment(runtime,config,domain,metadata,cache)
    if secondary.get("pathways"):
        metadata["pathway_selection"]=_prepare_pathway_selection(runtime,selection,metadata,secondary["pathways"])
    selected_fits={k:fits[v["key"]] for k,v in selected.items()}
    if _rank_candidates(cfg):
        metadata["rank_sensitivity"],selected_fits["rank_sensitivity"]=_prepare_rank_sensitivity(runtime,train,selection,group,selected,cfg,cache,output)
    return metadata,selected_fits


def _evaluate_cell(config,model,domain,runtime,selection,fits,output,cache,split="test",test_families=None):
    cfg=config.get("native",{}); records=_build(config,domain,split,test_families=test_families)
    rows=[];generation=bool(cfg.get("generation",True)); seeds=cfg.get("random_seeds",RANDOM_SEEDS)
    clean=_clean_evaluation(runtime,records,generation)
    _write_rows(Path(output)/f"{split}_clean.jsonl",clean)
    family_names=sorted({r["family_id"] for r in records})
    sensitivity_names=set(family_names[::4]); counter=[_counterpart(r) for r in records]
    clean_by_id={r["record_id"]:r for r in clean}
    for site in ("intermediate","answer"):
        meta=selection["selected"][site];fit=fits[site];layer=meta["layer"];readout=meta["readout"]
        x=collect_features(runtime,records,[layer],readout,cache)[layer]
        y=collect_features(runtime,counter,[layer],readout,cache)[layer]
        methods=None if site=="intermediate" else ["group",meta["baseline"],"natural_donor"]
        rows+=evaluate_methods(runtime,records,fit,x,y,layer,readout=readout,methods=methods,
                               generation=generation,site=site,random_seeds=seeds)
        if split=="test" and cfg.get("dose_sensitivity",True):
            ids=[i for i,r in enumerate(records) if r["family_id"] in sensitivity_names]
            if 0. in cfg.get("doses",[0.,.25,.5,1.,2.]):
                for i in ids:
                    r=records[i];reference=clean_by_id[r["record_id"]];scores=reference["candidate_logprobs"]
                    idx=r["candidates"].index(r["counterfactual_answer"])
                    for method in ("group",meta["baseline"],"natural_donor"):
                        rows.append({**reference,"site_variable":site,"layer":layer,"readout":readout,"method":method,
                                     "dose":0.,"delta_norm":0.,"target":r["counterfactual_answer"],
                                     "correct":_normalized_answer(reference["generated_text"])==_normalized_answer(r["counterfactual_answer"]),
                                     "constrained_correct":_normalized_answer(reference["prediction"])==_normalized_answer(r["counterfactual_answer"]),
                                     "sequence_margin":float(scores[idx]-max(s for j,s in enumerate(scores) if j!=idx)),
                                     "reused_unedited_score":True,"forward_counts":{}})
            for dose in cfg.get("doses",[0.,.25,.5,1.,2.]):
                if dose in (0.,1.):continue # alpha=0 is the earlier zero-delta/unedited reference; one already scored.
                rows+=evaluate_methods(runtime,[records[i] for i in ids],fit,x[ids],y[ids],layer,readout=readout,
                                       methods=["group",meta["baseline"],"natural_donor"],dose=float(dose),
                                       generation=generation,site=site,random_seeds=seeds)
    _write_rows(Path(output)/f"{split}_evaluations.jsonl",rows)
    primary=[r for r in rows if r["site_variable"]=="intermediate" and r["dose"]==1. and r.get("word_split")=="transfer"]
    baseline=selection["selected"]["intermediate"]["baseline"]
    return {"rows":len(records),"evaluations":len(rows),"family_contrasts":_family_contrasts(primary,baseline),
            "baseline":baseline,"clean_accuracy":float(np.mean([r["correct"] for r in clean])),
            "primary_endpoint":"native_continuation_length6_to8_transfer_only",
            "all_test_items_scored":True,"records_sha256":identity(records),
            "metrics_path":str(Path(output)/f"{split}_evaluations.jsonl")},rows,records


def _method_information():
    return {"group":"native construction paired generator transitions; no sealed target activations",
            "centroid":"oracle current state + train-only state means",
            "context_shrinkage":"oracle current state + validation-selected residual retention",
            "centroid_decoded":"construction-trained nearest centroid decoder; no oracle state",
            "local_displacement":"oracle state + construction residual-distance kernel; 8 validation bandwidths",
            "natural_donor":"oracle target activation; reference, excluded from best-baseline selection",
            "random":"5 fixed-seed rank-matched oracle donor subspaces",
            "legacy_mean_sham":"full-D flattening exception, same hook/span",
            "late_control":"answer variable with effective edit v g v^-1; program and source labels supplied"}


def _run_native_impl(config,output,cache_root=None,mode="full",*,runtime_factory=None):
    """Run development/power/freeze before sealed-test activation extraction.

    Modes: prepare (fit, validation, excluded pilot, freeze); confirm (verify
    frozen development identity, then test); full (both); pilot (prepare only).
    All model inference is real runtime scoring, or explicitly marked technical.
    """
    if mode not in ("full","prepare","confirm","pilot"):raise ValueError("Unknown native run mode")
    start=time.monotonic();config=normalized_config(config);cfg=config.get("native",{});output=Path(output)
    prepare_run(output,config,"native")
    cache=Path(cache_root) if cache_root else output/"native_cache"
    models=_models(config);domains=[_domain(v) for v in cfg.get("domains",["C7","D4_free8"])]
    runtimes={};selections={};fits={};pilot_contrasts={}
    frozen_path=output/"selection_frozen.json"
    old=read_json(frozen_path) if frozen_path.exists() else None
    if old is not None and old.get("selection_identity")!=identity({k:v for k,v in old.items() if k!="selection_identity"}):
        raise ValueError("Native frozen-selection checksum mismatch")
    if mode=="confirm" and old is None:raise ValueError("Confirm requires a completed selection_frozen.json; no test outcomes collected")
    for mi,model in enumerate(models):
        runtime=make_runtime(model,config.get("runtime",{}),runtime_factory);runtimes[mi]=runtime
        for domain in domains:
            cell=f"m{mi}.{domain}";directory=output/cell;directory.mkdir(parents=True,exist_ok=True)
            if old is not None:
                selection=old["cells"][cell]
                train=[r for r in _build(config,domain,"train") if r["fit_eligible"]]
                validation=[r for r in _build(config,domain,"validation") if r["selection_eligible"]]
                cellfits={site:_fit_site(runtime,train,validation,get_group(domain),entry["layer"],entry["readout"],site,cfg,cache/cell)
                          for site,entry in selection["selected"].items()}
                for site,value in cellfits.items():
                    if identity(value.metadata)!=identity(selection["selected"][site]["fit"]):
                        raise ValueError("Construction fit changed after freeze")
                if _rank_candidates(cfg):
                    cellfits["rank_sensitivity"]=_restore_rank_sensitivity(runtime,train,validation,get_group(domain),selection,cfg,cache/cell)
            else:selection,cellfits=_prepare_cell(config,model,domain,runtime,directory,cache/cell)
            selections[cell]=selection
            if cellfits is None:
                result={"study":"native","technical_run":bool(cfg.get("technical_run",False)),"status":"clean_competence_failed","cells":selections}
                atomic_json(output/"native_results.json",result);return result
            fits[cell]=cellfits
            if old is not None and identity(old["cells"].get(cell))!=identity(selection):
                raise ValueError("Development choices changed after freezing; choose a new run directory")
            if mode!="confirm":
                pilot,_,_=_evaluate_cell(config,model,domain,runtime,selection,cellfits,directory,cache/cell,split="pilot")
                pilot_contrasts[cell]=list(pilot["family_contrasts"].values())
        if cfg.get("include_rebinding"):
            run_rebinding_stage(runtime,config,model,output/f"m{mi}.rebinding",cache/f"m{mi}.rebinding",mode="prepare")
        if hasattr(runtime,"close"):runtime.close()
    if old is None:
        lengths={len(v) for v in pilot_contrasts.values()}
        measured=np.asarray(list(pilot_contrasts.values()),float).T if len(pilot_contrasts)==4 and len(lengths)==1 else None
        power=native_power(config,measured)
        requested=int(cfg.get("families",{}).get("test",32))
        gate=power["selected_families"]
        technical=bool(cfg.get("technical_run",False))
        if (gate is None or (cfg.get("power",{}).get("require_measured",True) and measured is None)) and not technical:
            result={"study":"native","technical_run":False,"status":"power_gate_not_met_no_test_outcomes","cells":selections,"power":power}
            atomic_json(output/"native_results.json",result);return result
        old={"schema_version":1,"config_identity":identity(config),"cells":selections,"power":power,
             "test_families":requested if technical else int(gate),"technical_run":technical,
             "method_information":_method_information(),"frozen_before_test_activations":True}
        old["selection_identity"]=identity(old);atomic_json(frozen_path,old)
    if mode in ("prepare","pilot"):
        return {"study":"native","status":"prepared_no_test_outcomes","technical_run":old["technical_run"],"selection":str(frozen_path)}
    if old["config_identity"]!=identity(config):raise ValueError("Frozen configuration identity mismatch")
    cells={};contrasts={}
    for mi,model in enumerate(models):
        runtimes[mi]=make_runtime(model,config.get("runtime",{}),runtime_factory)
        for domain in domains:
            cell=f"m{mi}.{domain}"
            result,rows,records=_evaluate_cell(config,model,domain,runtimes[mi],selections[cell],fits[cell],
                                                output/cell,cache/cell,test_families=old["test_families"])
            cells[cell]=result;contrasts[cell]=list(result["family_contrasts"].values())
            if _rank_candidates(cfg):
                cells[cell]["rank_sensitivity"]=run_rank_sensitivity(runtimes[mi],records,selections[cell],fits[cell]["rank_sensitivity"],
                                         output/cell/"rank_sensitivity",cache/cell,generation=bool(cfg.get("generation",True)))
            secondary=cfg.get("secondary",{})
            if secondary.get("operation_joint"):
                cells[cell]["operation_joint"]=run_operation_interchange(runtimes[mi],config,domain,records,
                                                    selections[cell],fits[cell],output/cell/"operation_joint",cache/cell)
            if secondary.get("pathways"):
                cells[cell]["pathways"]=run_pathway_tests(runtimes[mi],records,selections[cell]["pathway_selection"],output/cell/"pathways")
            if cfg.get("include_shams"):
                cells[cell]["shams"]=run_sham_controls(runtimes[mi],config,records,selections[cell],fits[cell],output/cell/"shams",cache/cell)
            if cfg.get("include_rebinding") and domain==domains[-1]:
                cells[cell]["rebinding"]=run_rebinding_stage(runtimes[mi],config,model,output/f"m{mi}.rebinding",cache/f"m{mi}.rebinding",old["test_families"])
        if hasattr(runtimes[mi],"close"):runtimes[mi].close()
    decision=native_decisions(contrasts,alpha=float(cfg.get("alpha",.05)),practical=float(cfg.get("primary_observed_margin",.05)),
                              bootstrap=int(cfg.get("bootstrap_replicates",0)))
    result={"study":"native_preregistered_counterfactual","technical_run":old["technical_run"],
            "status":"completed","selection_identity":old["selection_identity"],"cells":cells,
            "decisions":decision,"method_information":_method_information(),
            "limits":"intervention-relative carrier/abstraction, not unique mechanism or universal necessity",
            "wall_seconds":time.monotonic()-start}
    atomic_json(output/"native_results.json",result);return result


def run_native(config,output,cache_root=None,mode="full",*,runtime_factory=None):
    """Resource-safe public runner; even competence/power/freeze failures close models."""
    created=[]
    def tracked(model,runtime_config):
        runtime=make_runtime(model,runtime_config,runtime_factory)
        created.append(runtime);return runtime
    try:
        return _run_native_impl(config,output,cache_root,mode,runtime_factory=tracked)
    finally:
        for runtime in created:
            if hasattr(runtime,"close"):runtime.close()


def _operation_donor(record,group):
    from .data import render_native_record
    word=list(record["continuation_elements"])
    if not word:raise ValueError("Operation interchange requires a nonempty native continuation")
    primitive=group.normalize(word[0])
    choices=list(group.generators)
    word[0]=choices[(choices.index(primitive)+1)%len(choices)] if primitive in choices and len(choices)>1 else group.inverse(primitive)
    if word[0]==primitive:raise ValueError("Operation substitution must change a nonidentity primitive")
    return render_native_record(record,continuation_elements=word)


def _prepare_operation_alignment(runtime,config,domain,selection,cache):
    cfg=config.get("native",{}).get("operation",{});group=get_group(domain)
    layer=int(cfg.get("layer",selection["selected"]["intermediate"]["layer"]))
    readout=cfg.get("readout","final_token");rank=int(cfg.get("rank",2))
    construction=[r for r in _build(config,domain,"train") if r["fit_eligible"]]
    natural=[_operation_donor(r,group) for r in construction]
    a=collect_features(runtime,construction,[layer],readout,cache)[layer]
    b=collect_features(runtime,natural,[layer],readout,cache)[layer]
    # Uncentered differences retain a constant operation displacement. The
    # separate state support is fitted at this exact same layer/readout.
    _,singular,vt=np.linalg.svd(b-a,full_matrices=False)
    q=min(rank,int(np.sum(singular>max(singular[0],1.)*1e-12)))
    if q<1:raise ValueError("No construction-identifiable operation displacement")
    basis=vt[:q].T;_,state_basis,state_info=pca(a,int(config["native"].get("rank",16)))
    combined=np.concatenate((basis,state_basis),1);u,s,_=np.linalg.svd(combined,full_matrices=False)
    union=u[:,:int(np.sum(s>1e-10))]
    return {"fit_ids":identity([r["record_id"] for r in construction]),"layer":layer,"readout":readout,
            "basis":basis.tolist(),"joint_basis":union.tolist(),"singular_values":singular.tolist(),
            "state_support":state_info,"rank":q,"joint_rank":union.shape[1],
            "frozen_before_test":True,"alignment":"uncentered_natural_operation_difference_SVD",
            "scope":"separate_operation_variable_alignment_not_state_operator_reuse"}


def run_operation_interchange(runtime,config,domain,test_records,selection,state_fits,output,cache=None):
    """Separately fit operation-variable subspace, then test operation/joint swaps.

    PCA of construction natural operation differences is an explicit alignment
    method, not a reuse of state action matrices. All choices freeze on familiar
    validation before test evaluation; recipient/donor answer transplantation
    controls are reported per case.
    """
    from .data import render_native_record
    group=get_group(domain);frozen=selection.get("operation_alignment")
    if not frozen or not frozen.get("frozen_before_test"):raise ValueError("Operation alignment must be fitted/frozen during development")
    layer=frozen["layer"];readout=frozen["readout"];basis=np.asarray(frozen["basis"]);union=np.asarray(frozen["joint_basis"])
    atomic_json(Path(output)/"alignment_frozen.json",frozen)
    targets=[_operation_donor(r,group) for r in test_records]
    original=collect_features(runtime,test_records,[layer],readout,cache)[layer]
    changed=collect_features(runtime,targets,[layer],readout,cache)[layer]
    rows=[]
    for r,d,h,hd in zip(test_records,targets,original,changed):
        delta=((hd-h)@basis)@basis.T
        op_record=dict(r,counterfactual_answer=d["answer"])
        rows.append({"record_id":r["record_id"],"kind":"operation_only",**score_case(runtime,op_record,patch=make_patch(layer,r,delta,readout=readout),generation=True)})
        # A genuine natural joint counterpart changes BOTH the initial state
        # and operation, and its outcome need not equal the donor's own answer.
        joint=render_native_record(d,initial_state=r["counterfactual_initial_state"])
        hj=collect_features(runtime,[joint],[layer],readout,cache)[layer][0]
        joint_record=dict(r,counterfactual_answer=joint["answer"])
        rows.append({"record_id":r["record_id"],"kind":"state_operation_joint","donor_answer":d["answer"],
                     "joint_answer_differs_from_operation_donor":joint["answer"]!=d["answer"],
                     **score_case(runtime,joint_record,patch=make_patch(layer,r,((hj-h)@union)@union.T,readout=readout),generation=True)})
    _write_rows(Path(output)/"evaluations.jsonl",rows)
    return {"alignment":frozen["alignment"],"rank":frozen["rank"],"joint_rank":frozen["joint_rank"],"evaluations":len(rows),
            "scope":"secondary_no_state_operator_is_treated_as_an_operation_editor"}


def _prepare_pathway_selection(runtime,validation,selection,config):
    if not hasattr(runtime,"score_components"):raise ValueError("Pathway study requires real backend component hooks")
    if config.get("frozen_selection") and config.get("components"):return config
    layer=selection["selected"]["intermediate"]["layer"]
    candidates=config.get("candidates",[{"layer":layer,"module":module} for module in ("mlp","attention","block")])
    # Deterministic balanced prefix of familiar validation families; the cap is
    # fixed before observing any score and records its full identity.
    cap=int(config.get("validation_cases",24));records=list(validation)[:cap]
    scores=[]
    for candidate in candidates:
        correct=[]
        for r in records:
            component=dict(candidate,mode="zero",span=r["carrier_span"])
            out=runtime.score_components(r["text"],r["candidates"],[component],generation=True,max_new_tokens=48)
            correct.append(_normalized_answer(out.get("generated_text"))==_normalized_answer(r["answer"]))
        scores.append({"component":candidate,"clean_answer_accuracy_after_zero":float(np.mean(correct))})
    selected=min(scores,key=lambda row:(row["clean_answer_accuracy_after_zero"],str(row["component"])))
    # A fixed-seed different block of the same module supplies a matched
    # location control. It is not selected for weak validation disruption.
    n=int(getattr(runtime,"n_layers",max(layer,2)));choices=[i for i in range(1,n+1) if i!=int(selected["component"]["layer"])]
    random_layer=int(np.random.default_rng(int(config.get("seed",2027101799))).choice(choices)) if choices else layer
    random_component=dict(selected["component"],layer=random_layer)
    components=[]
    for label,c in (("validation_selected",selected["component"]),("fixed_random_layer",random_component)):
        for mode in ("zero","resample","mean"):
            components.append({"label":label,"components":[dict(c,mode=mode)]})
    frozen={"validation_identity":identity([r["record_id"] for r in records]),"validation_cases":len(records),
            "candidates":scores,"selected":selected,"random_control":random_component,
            "selection_rule":"largest_familiar_validation_clean_answer_disruption_by_zeroing",
            "frozen_before_test":True}
    return {"components":components,"frozen_selection":frozen}


def run_pathway_tests(runtime,records,config,output):
    """Execute predeclared component interventions supplied to the HF backend.

    Native component specs are validated by the backend. No post-test component
    search occurs here; a frozen construction/validation identity is required.
    """
    frozen=config.get("frozen_selection")
    if not frozen:raise ValueError("Pathway tests require construction/validation frozen component selection")
    specs=config.get("components",[])
    if not specs:raise ValueError("No actual pathway component interventions configured")
    if not hasattr(runtime,"score_components"):
        raise ValueError("Runtime lacks real component-intervention support; cannot substitute a state patch")
    rows=[]
    for r in records:
        for item in specs:
            label=item.get("label") if isinstance(item,dict) else None
            raw=item["components"] if isinstance(item,dict) else item
            donor_text=r.get("counterfactual_text","");marker=donor_text.index("Continue")
            spec=[dict(c,span=r["carrier_span"],donor_span=[marker,marker+len("Continue")]) for c in raw]
            out=runtime.score_components(r["text"],r["candidates"],components=spec,
                                          donor_prompt=r.get("counterfactual_text"),generation=True)
            rows.append({"record_id":r["record_id"],"family_id":r["family_id"],"label":label,"components":spec,
                         "clean_answer":r["answer"],"counterfactual_answer":r["counterfactual_answer"],
                         "clean_correct":_normalized_answer(out.get("generated_text"))==_normalized_answer(r["answer"]),
                         "counterfactual_correct":_normalized_answer(out.get("generated_text"))==_normalized_answer(r["counterfactual_answer"]),"score":out})
    _write_rows(Path(output)/"evaluations.jsonl",rows)
    return {"evaluations":len(rows),"selection_identity":identity(frozen),"scope":"secondary_component_interventions_not_unique_mechanism_identification"}


def run_sham_controls(runtime,config,records,selection,fits,output,cache=None):
    """Actual task-negative controls; state operators remain the frozen editors.

    Initial-state repetition is a positive localization diagnostic. List and
    unrelated-code tasks have unchanged answers, so a successful state edit
    should not systematically change their outputs. Clean task competence is
    reported rather than silently excluding failed sham items.
    """
    from .data import render_native_sham
    cfg=config["native"];entry=selection["selected"]["intermediate"]
    fit=fits["intermediate"];layer=entry["layer"];readout=entry["readout"]
    rows=[]
    for kind in ("list","initial_state","unrelated_attribute"):
        sham=[render_native_sham(r,kind) for r in records]
        x=collect_features(runtime,sham,[layer],readout,cache)[layer]
        donors=[]
        # The native counterpart is rendered *before* modifying the task, so
        # its state/program conventions remain valid and spans are regenerated.
        for original in records:donors.append(render_native_sham(_counterpart(original),kind))
        y=collect_features(runtime,donors,[layer],readout,cache)[layer]
        for r in sham:
            clean=score_case(runtime,dict(r,counterfactual_answer=r["answer"]),generation=True)
            rows.append({"record_id":r["record_id"],"family_id":r["family_id"],"task_sham":kind,"method":"clean",**clean})
        scored=evaluate_methods(runtime,sham,fit,x,y,layer,readout=readout,
                                  methods=["group",entry["baseline"],"natural_donor","delta_sham"],
                                  generation=True,site="intermediate")
        rows.extend(dict(r,task_sham=kind) for r in scored)
    _write_rows(Path(output)/"evaluations.jsonl",rows)
    return {"recipient_cases":3*len(records),"evaluations":len(rows),
            "scope":"task_specific_negative_and_positive_controls_with_frozen_state_editor",
            "training_on_sham_outcomes":False,"no_clean_correct_item_filter":True}


def run_rebinding_stage(runtime,config,model,output,cache=None,test_families=None,mode="full"):
    """Fixed-token whole-prompt S3 binding adaptation with unseen entity arms.

    This is a falsification probe inspired by binding work, not a reproduction
    of a paper's complete mechanism. Entities are not concatenated into the
    readout. All editors are trained at the very same final-token site.
    """
    from .data import build_rebinding_records,render_rebinding_record
    output=Path(output);cfg=config["native"];extra=cfg.get("rebinding",{});group=get_group("S3")
    seed=int(cfg.get("seed",2027101700));count=int(test_families or cfg["families"]["test"])
    def build(splits):
        pool=next((n for n in (32,48,64) if n>=count),None)
        if pool is None:raise ValueError("Rebinding family count exceeds the64-family pool")
        values=build_rebinding_records(seed=seed,test_families=pool,splits=splits)
        cap=extra.get("max_orbits_per_entity_split")
        if cap is not None:
            allowed={e:set(sorted({r["orbit_id"] for r in values if r["entity_split"]==e})[:int(cap)]) for e in ("train","validation","test")}
            values=[r for r in values if r["orbit_id"] in allowed[r["entity_split"]]]
        # Keep the registered family prefixes while retaining all three crossed
        # test arms. The split itself never determines fitted target activations.
        return [r for r in values if int(r["family_id"].rsplit("_",1)[-1])<int(cfg["families"].get(r["context_split"],count))]
    train=build(["train"]);validation=build(["validation"])
    layer=int(extra.get("layer",next(s[0] for s in _sites(config,model,runtime) if s[2]=="answer")))
    readout="final_token"
    x=collect_features(runtime,train,[layer],readout,cache)[layer]
    xv=collect_features(runtime,validation,[layer],readout,cache)[layer]
    target={};target_v={}
    for g in group.generators:
        target[g]=collect_features(runtime,[render_rebinding_record(r,group.apply(g,r["state_id"])) for r in train],[layer],readout,cache)[layer]
        target_v[g]=collect_features(runtime,[render_rebinding_record(r,group.apply(g,r["state_id"])) for r in validation],[layer],readout,cache)[layer]
    fit=fit_native(train,validation,x,xv,target,target_v,group,rank=int(cfg["rank"]),site="intermediate")
    frozen_path=output/"rebinding_frozen.json"
    if frozen_path.exists():
        frozen=read_json(frozen_path)
        if frozen["fit_identity"]!=identity(fit.metadata):raise ValueError("Rebinding construction fit changed after freeze")
    else:
        yv=collect_features(runtime,[render_rebinding_record(r,r["counterfactual_initial_state"]) for r in validation],[layer],readout,cache)[layer]
        scored=evaluate_methods(runtime,validation,fit,xv,yv,layer,readout=readout,methods=["group",*INFORMED],site="intermediate")
        baseline=max(INFORMED,key=lambda method:(_accuracy(scored,method),-INFORMED.index(method)))
        clean=_clean_evaluation(runtime,[dict(r,word_split="development") for r in validation])
        frozen={"fit_identity":identity(fit.metadata),"fit":fit.metadata,"layer":layer,"baseline":baseline,
                "train_ids":identity([r["record_id"] for r in train]),"validation_ids":identity([r["record_id"] for r in validation]),
                "frozen_before_test":True,"validation_clean_accuracy":float(np.mean([r["correct"] for r in clean]))}
        _write_rows(output/"validation_evaluations.jsonl",scored);atomic_json(frozen_path,frozen)
    if mode=="prepare":return {"status":"prepared_no_test_outcomes","frozen_identity":identity(frozen)}
    records=build(extra.get("test_splits",["test_context","test_entity","test_joint"]))
    h=collect_features(runtime,records,[layer],readout,cache)[layer]
    donor=collect_features(runtime,[render_rebinding_record(r,r["counterfactual_initial_state"]) for r in records],[layer],readout,cache)[layer]
    rows=evaluate_methods(runtime,records,fit,h,donor,layer,readout=readout,
                           methods=["group",frozen["baseline"],"centroid_decoded","natural_donor","delta_sham"],site="intermediate")
    _write_rows(output/"test_evaluations.jsonl",rows)
    result={"study":"fixed_token_S3_binding_adaptation","technical_run":bool(cfg.get("technical_run",False)),
            "frozen_identity":identity(frozen),"recipients":len(records),"evaluations":len(rows),
            "arms":{arm:sum(r["split"]==arm for r in records) for arm in sorted({r["split"] for r in records})},
            "information":"whole prompt final token; construction entity tuples only; no concatenated entity embeddings",
            "scope":"secondary_Todd_style_binding_falsification_not_original_reproduction"}
    atomic_json(output/"rebinding_results.json",result);return result
