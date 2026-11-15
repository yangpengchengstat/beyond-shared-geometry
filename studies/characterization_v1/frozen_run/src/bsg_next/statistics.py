"""Registered primary t/TOST/Holm decisions and separately labeled sensitivity."""
from __future__ import annotations

import time
from pathlib import Path
import numpy as np
from scipy import stats


def holm(pvalues, alpha=.05):
    keys = list(pvalues)
    p = np.asarray([pvalues[k] for k in keys], float)
    p[~np.isfinite(p)] = 1.
    order = np.argsort(p, kind="stable")
    adjusted = np.empty(len(p)); previous = 0.
    for rank, idx in enumerate(order):
        previous = max(previous, (len(p) - rank) * p[idx])
        adjusted[idx] = min(1., previous)
    return {k: {"p": float(p[i]), "p_adjusted": float(adjusted[i]),
                "reject": bool(adjusted[i] <= alpha)} for i, k in enumerate(keys)}


def contrast_summary(values=None, *, mean=None, se=None, df=None, n=None,
                     margin=.01, equivalence=.005, eligible=True, alpha=.04, tests=8):
    if values is not None:
        a = np.asarray(values, float)
        if a.ndim != 1 or not np.isfinite(a).all():
            raise ValueError("Family contrasts must be finite one-dimensional values")
        n = len(a); mean = float(a.mean()) if n else float("nan")
        se = float(a.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        df = n - 1
    valid = bool(eligible and df is not None and df > 0 and se is not None and se > 0
                 and np.isfinite(se) and np.isfinite(mean))
    if valid:
        ps = float(stats.t.sf((mean - margin) / se, df))
        pl = float(stats.t.sf((mean + equivalence) / se, df))
        pu = float(stats.t.cdf((mean - equivalence) / se, df))
        critical = float(stats.t.ppf(1 - alpha / tests, df))
        ordinary = float(stats.t.ppf(.975, df))
        ci95 = [mean - ordinary * se, mean + ordinary * se]
        decision_interval = [mean - critical * se, mean + critical * se]
    else:
        ps = pl = pu = 1.; ci95 = decision_interval = [None, None]
    return {"mean": float(mean), "se": float(se), "df": int(df), "n": n,
            "eligible": bool(eligible), "valid_inference": valid,
            "margin": margin, "equivalence_half_width": equivalence,
            "p_superiority": ps, "p_equivalence": max(pl, pu),
            "p_tost_lower": pl, "p_tost_upper": pu, "ci95": ci95,
            "bonferroni_equivalence_interval": decision_interval,
            "bonferroni_superiority_lower": decision_interval[0]}


def family_decisions(family_contrasts, alpha=.04, margin=.01, equivalence=.005):
    """Two outcomes × two groups × superiority/equivalence: one eight-test family."""
    summaries, p = {}, {}
    tests = 4 * len(family_contrasts)
    for group, entry in family_contrasts.items():
        for endpoint in ("pred", "ctx"):
            eligible = entry.get("eligible", True) if endpoint == "ctx" else True
            key = f"{group}.{endpoint}"
            summaries[key] = contrast_summary(entry[endpoint], margin=margin, equivalence=equivalence,
                                               eligible=eligible, alpha=alpha, tests=tests)
            for claim in ("superiority", "equivalence"):
                p[f"{key}.{claim}"] = summaries[key][f"p_{claim}"]
    decisions = holm(p, alpha)
    for key, summary in summaries.items():
        superior = decisions[f"{key}.superiority"]["reject"]
        equivalent = decisions[f"{key}.equivalence"]["reject"]
        summary["status"] = ("uninformative" if not summary["eligible"] else
                             "meaningful_superiority" if superior else
                             "practical_equivalence" if equivalent else "inconclusive")
    return {"method": "family_t_TOST_Holm", "alpha": alpha, "test_count": tests,
            "contrasts": summaries, "tests": decisions,
            "both_groups_both_endpoints_superior": all(
                v["status"] == "meaningful_superiority" for v in summaries.values())}


def crossed_summary(cells, **kwargs):
    a = np.asarray(cells, float)
    if a.ndim != 2 or min(a.shape) < 2 or not np.isfinite(a).all():
        raise ValueError("Crossed inference needs a finite balanced family×orbit table")
    f, o = a.shape
    variance = np.var(a.mean(1), ddof=1) / f + np.var(a.mean(0), ddof=1) / o
    result = contrast_summary(mean=float(a.mean()), se=float(np.sqrt(variance)),
                              df=min(f-1, o-1), n=f*o, **kwargs)
    result.update({"family_clusters": f, "orbit_clusters": o,
                   "assumptions": "balanced independent crossed random effects; approximate t reference"})
    return result


def crossed_decisions(cell_contrasts, alpha=.01, margin=.01, equivalence=.005, eligible=True):
    summaries = {k: crossed_summary(v, alpha=alpha, tests=4, margin=margin,
                                    equivalence=equivalence, eligible=eligible if k == "ctx" else True)
                 for k, v in cell_contrasts.items()}
    p = {f"{k}.{claim}": v[f"p_{claim}"] for k, v in summaries.items()
         for claim in ("superiority", "equivalence")}
    decisions = holm(p, alpha)
    for k, v in summaries.items():
        v["status"] = ("uninformative" if not v["eligible"] else
                       "meaningful_superiority" if decisions[f"{k}.superiority"]["reject"] else
                       "practical_equivalence" if decisions[f"{k}.equivalence"]["reject"] else "inconclusive")
    return {"method": "crossed_conservative_SE_t_TOST_Holm", "alpha": alpha,
            "calibration_required": True, "contrasts": summaries, "tests": decisions}


def bootstrap_sensitivity(values, replicates=10000, seed=2027101600, crossed=False):
    a = np.asarray(values, float); rng = np.random.default_rng(seed)
    samples = np.empty(int(replicates))
    if crossed:
        if a.ndim != 2: raise ValueError("Crossed table required")
        f, o = a.shape
        for i in range(len(samples)):
            samples[i] = a[rng.integers(f, size=f)[:, None], rng.integers(o, size=o)[None, :]].mean()
    else:
        if a.ndim != 1: raise ValueError("One contrast per family required")
        for lo in range(0, len(samples), 256):
            count = min(256, len(samples)-lo)
            samples[lo:lo+count] = a[rng.integers(len(a), size=(count, len(a)))].mean(1)
    return {"sensitivity_only": True, "replicates": len(samples), "seed": int(seed),
            "crossed": crossed, "mean": float(a.mean()),
            "percentile_ci95": np.quantile(samples, [.025, .975]).tolist()}


def binomial_interval(successes, trials, level=.95):
    k, n = int(successes), int(trials)
    if n <= 0: return {"rate": None, "interval": [None, None], "upper_one_sided": None}
    tail = (1-level)/2
    lo = 0. if k == 0 else stats.beta.ppf(tail, k, n-k+1)
    hi = 1. if k == n else stats.beta.ppf(1-tail, k+1, n-k)
    upper = 1. if k == n else stats.beta.ppf(level, k+1, n-k)
    return {"rate": k/n, "successes": k, "trials": n, "interval": [float(lo), float(hi)],
            "upper_one_sided": float(upper), "mcse": float(np.sqrt(k/n*(1-k/n)/n))}


def _vector_rejections(data, alpha, margin=.01, equivalence=.005):
    """Vectorized EXACT same t/TOST/Holm rule as family_decisions (no bootstrap)."""
    n = data.shape[1]
    means, se = data.mean(1), data.std(1, ddof=1)/np.sqrt(n)
    with np.errstate(divide="ignore", invalid="ignore"):
        ps = stats.t.sf((means-margin)/se, n-1)
        pe = np.maximum(stats.t.sf((means+equivalence)/se, n-1),
                        stats.t.cdf((means-equivalence)/se, n-1))
    p = np.concatenate([ps, pe], axis=1)
    p[~np.isfinite(p) | np.concatenate([se <= 0, se <= 0], 1)] = 1.
    order = np.argsort(p, axis=1, kind="stable")
    sortedp = np.take_along_axis(p, order, 1)
    adj = np.minimum(1., np.maximum.accumulate(sortedp*np.arange(p.shape[1], 0, -1), axis=1))
    result = np.empty_like(adj)
    np.put_along_axis(result, order, adj, axis=1)
    return result <= alpha


def run_power(config, output, *, pilot_contrasts=None, limit=None):
    from .io import prepare_run, atomic_json, identity, read_json
    from .estimators import numeric_code_identity
    cfg = config.get("power", config.get("statistics", {}).get("power", {}))
    nrep = int(limit or cfg.get("replicates", 5000)); seed = int(cfg.get("seed", 2127101600))
    sizes = cfg.get("test_families", [32, 48, 64]); rng = np.random.default_rng(seed)
    pilot = pilot_contrasts if pilot_contrasts is not None else cfg.get("pilot_contrasts")
    if isinstance(pilot, str): pilot = np.load(pilot)
    if pilot is None:
        sigma = np.asarray(cfg.get("assumed_sd", [.05]*4), float)
        source = "explicit_planning_assumptions_not_measured_pilot"
    else:
        a = np.asarray(pilot, float)
        if a.ndim != 2 or a.shape[1] != 4: raise ValueError("Pilot needs family×four-contrast array")
        sigma = a.std(0, ddof=1); source = "excluded_pilot_variation"
    prepare_run(output, config, "power", provenance={"pilot_values_identity":identity(pilot),
                "replicates":nrep,"numeric_code_identity":numeric_code_identity()})
    alpha = float(config.get("statistics", {}).get("main_alpha", .04))
    margin = float(config.get("statistics", {}).get("margin", .01)); alt = float(cfg.get("alternative", .02))
    results = []; start = time.monotonic()
    for n in sizes:
        for correlation in cfg.get("correlations", [0., .8]):
            cov = np.full((4, 4), float(correlation)); np.fill_diagonal(cov, 1.)
            chol = np.linalg.cholesky(cov)
            stress_names={"pilot":"normal","student_t5":"t5","high_variance_mixture":"mixture"}
            for requested_stress in cfg.get("stress", ["normal", "t5", "mixture"]):
                stress=stress_names.get(requested_stress,requested_stress)
                if stress not in {"normal","t5","mixture"}:raise ValueError(f"Unknown power stress {requested_stress}")
                scenario={"families":int(n),"correlation":correlation,"stress":stress,"replicates":nrep}
                target=Path(output)/"scenarios"/(identity(scenario)[:20]+".json")
                if target.exists():results.append(read_json(target));continue
                rng=np.random.default_rng(np.random.SeedSequence([seed,int(n),int(correlation*10000),["normal","t5","mixture"].index(stress)]))
                x = rng.normal(size=(nrep, int(n), 4)) @ chol.T
                if stress == "t5": x *= np.sqrt(3/rng.chisquare(5, size=(nrep,int(n),1)))
                if stress == "mixture": x *= np.where(rng.random((nrep,int(n),1)) < .1, 3., 1.)
                x = x * sigma * np.sqrt(float(cfg.get("variance_inflation",2.)))
                reject = _vector_rejections(x+alt, alpha, margin)
                marginal = [binomial_interval(np.sum(reject[:,i]), nrep) for i in range(4)]
                joint = binomial_interval(np.all(reject[:,:4],1).sum(), nrep)
                eq = _vector_rejections(x, alpha, margin)[:,4:]
                result={"families": int(n), "correlation": correlation, "stress": stress,
                                "marginal_superiority": marginal, "joint_superiority": joint,
                                "requested_stress":requested_stress,
                                "equivalence_at_zero": [binomial_interval(eq[:,i].sum(),nrep) for i in range(4)]}
                atomic_json(target,result);results.append(result)
    passing = [n for n in sizes if all(
        all(v["interval"][0] >= float(cfg.get("marginal_target",.9)) for v in r["marginal_superiority"]) and r["joint_superiority"]["interval"][0] >= float(cfg.get("joint_target",.8))
        for r in results if r["families"] == n and r["stress"] == "normal")]
    result = {"source": source, "variance_inflation": 2., "seed": seed, "replicates": nrep,
              "alternative": alt, "margin": margin, "scenarios": results,
              "selected_families": min(passing) if passing else None,
              "status": "power_target_met" if passing else "limited_precision_do_not_claim_powered",
              "cpu_wall_seconds": time.monotonic()-start}
    atomic_json(Path(output)/"power.json", result)
    return result


def run_crossed_calibration(config, output, limit=None):
    from .io import prepare_run, atomic_json, read_json, identity
    from .estimators import numeric_code_identity
    cfg = config.get("synthetic", {}); reps = int(limit or cfg.get("crossed_replicates", 2000))
    summary_only = limit == 0
    worker=int(cfg.get("worker_index",0));workers=int(cfg.get("worker_count",1))
    if not 0<=worker<workers:raise ValueError("Invalid crossed calibration shard")
    clean=dict(config);clean["synthetic"]={k:v for k,v in cfg.items() if k not in {"worker_index","worker_count"}}
    marker=prepare_run(output, clean, "crossed_calibration",provenance={"numeric_code_identity":numeric_code_identity(),"replicates":reps})
    seedbase=int(cfg.get("crossed_seed",2327101600));scenario_index=0
    out = []
    for f in cfg.get("crossed_families", [32,48,64]):
        for o in cfg.get("crossed_orbits", [16,24]):
            for ratio in cfg.get("crossed_variances", [[1,0,1],[0,1,1],[1,1,1],[1,4,1],[4,1,1]]):
                for heavy in (False, True):
                    for corr in cfg.get("crossed_correlations", [0., .8]):
                        corr=float(corr)
                        for hetero in cfg.get("crossed_heteroscedastic",[False,True]):
                            scenario={"families":f,"orbits":o,"variance_ratio":ratio,"heavy_t5":heavy,"correlation":corr,"heteroscedastic":hetero}
                            sid=identity(scenario)[:16]
                            for rep in range(reps):
                                if summary_only:break
                                if (scenario_index*reps+rep)%workers!=worker:continue
                                target=Path(output)/"seeds"/sid/f"rep_{rep:05d}.json"
                                if target.exists():
                                    saved=read_json(target)
                                    if saved["run_identity"]!=marker["identity"]:raise ValueError("Crossed seed identity mismatch")
                                    continue
                                seed=seedbase+100000*scenario_index+rep;rng=np.random.default_rng(seed)
                                def draw(shape):
                                    common = rng.normal(size=shape[:-1]+(1,))
                                    x = np.sqrt(corr)*common+np.sqrt(1-corr)*rng.normal(size=shape)
                                    return x*np.sqrt(3/rng.chisquare(5,size=shape[:-1]+(1,))) if heavy else x
                                a=draw((f,1,2))*np.sqrt(ratio[0]); b=draw((1,o,2))*np.sqrt(ratio[1]); e=draw((f,o,2))*np.sqrt(ratio[2])
                                if hetero:
                                    row_scale=np.linspace(.5,1.5,f)[:,None,None];column_scale=np.linspace(.5,1.5,o)[None,:,None]
                                    a*=row_scale;e*=row_scale*column_scale
                                x=(a+b+e)*.02
                                d=crossed_decisions({"pred":x[:,:,0]+.01,"ctx":x[:,:,1]+.01})
                                rejected=any(v["reject"] for k,v in d["tests"].items() if k.endswith("superiority"))
                                covered=all(v["ci95"][0] is not None and v["ci95"][0] <= .01 <= v["ci95"][1] for v in d["contrasts"].values())
                                positive=crossed_decisions({"pred":x[:,:,0]+.02,"ctx":x[:,:,1]+.02})
                                powered=all(v["status"]=="meaningful_superiority" for v in positive["contrasts"].values())
                                # Equivalence is a different composite null. Audit its
                                # boundary too, rather than only testing superiority.
                                equiv=crossed_decisions({"pred":x[:,:,0]+.005,"ctx":x[:,:,1]+.005})
                                false_equiv=any(v["reject"] for k,v in equiv["tests"].items() if k.endswith("equivalence"))
                                atomic_json(target,{"run_identity":marker["identity"],"seed":seed,"replicate":rep,
                                                    "rejected":rejected,"covered":covered,"powered":powered,"false_equivalence":false_equiv})
                            records=[read_json(p) for p in sorted((Path(output)/"seeds"/sid).glob("rep_*.json"))]
                            count=len(records)
                            out.append({**scenario,"completed_replicates":count,"expected_replicates":reps,
                                        "null_boundary_fwer":binomial_interval(sum(r["rejected"] for r in records),count),
                                        "equivalence_boundary_fwer":binomial_interval(sum(r["false_equivalence"] for r in records),count),
                                        "joint_ci95_coverage":binomial_interval(sum(r["covered"] for r in records),count),
                                        "joint_power":binomial_interval(sum(r["powered"] for r in records),count)})
                            scenario_index+=1
    missing=np.zeros((4,4));missing[0,0]=np.nan
    try:
        crossed_summary(missing);missing_rejected=False
    except ValueError:missing_rejected=True
    result={"method":"crossed_actual_primary_inference", "no_inner_bootstrap":True,"scenarios":out,
            "missing_cell_integrity_check":{"unsupported_missing_cells_rejected":missing_rejected,"no_inference_reported_on_missing_table":True}}
    atomic_json(Path(output)/f"crossed_calibration_worker_{worker:03d}.json",result);return result


def _psd(matrix):
    value=(np.asarray(matrix)+np.asarray(matrix).T)/2
    eig,vec=np.linalg.eigh(value)
    return (vec*np.maximum(eig,0.))@vec.T


def crossed_variance_components(pilot):
    """Method-of-moments crossed covariance, PSD truncation explicitly recorded."""
    x=np.asarray(pilot,float)
    if x.ndim!=3 or x.shape[-1]!=2 or min(x.shape[:2])<3 or not np.isfinite(x).all():
        raise ValueError("S3 pilot must be finite family×orbit×2, at least three clusters per axis")
    f,o,_=x.shape;grand=x.mean((0,1));row=x.mean(1);column=x.mean(0)
    residual=x-row[:,None,:]-column[None,:,:]+grand
    ce=residual.reshape(-1,2).T@residual.reshape(-1,2)/((f-1)*(o-1))
    ca=np.cov(row,rowvar=False,ddof=1)-ce/o
    cb=np.cov(column,rowvar=False,ddof=1)-ce/f
    return {"family":_psd(ca),"orbit":_psd(cb),"interaction":_psd(ce)}, {
        "pilot_family_clusters":f,"pilot_orbit_clusters":o,
        "raw_family_covariance":ca.tolist(),"raw_orbit_covariance":cb.tolist(),
        "raw_interaction_covariance":ce.tolist(),"negative_eigenvalues_truncated":True,
        "scope":"excluded pilot only; limited precision with eight pilot orbits"}


def _crossed_vector_rejections(x,alpha=.01,margin=.01,equivalence=.005):
    f,o=x.shape[1:3];means=x.mean((1,2))
    se=np.sqrt(x.mean(2).var(1,ddof=1)/f+x.mean(1).var(1,ddof=1)/o)
    df=min(f-1,o-1)
    with np.errstate(divide="ignore",invalid="ignore"):
        ps=stats.t.sf((means-margin)/se,df)
        pe=np.maximum(stats.t.sf((means+equivalence)/se,df),stats.t.cdf((means-equivalence)/se,df))
    p=np.concatenate([ps,pe],1);p[~np.isfinite(p)|np.concatenate([se<=0,se<=0],1)]=1.
    order=np.argsort(p,1,kind="stable");sort=np.take_along_axis(p,order,1)
    adjusted=np.minimum(1.,np.maximum.accumulate(sort*np.arange(4,0,-1),axis=1))
    original=np.empty_like(adjusted);np.put_along_axis(original,order,adjusted,axis=1)
    return original<=alpha


def run_s3_power(config,output,*,pilot_crossed=None,limit=None):
    """Prospective power of the actual crossed SE/df/Holm four-test family."""
    from .io import prepare_run,atomic_json,read_json,identity
    from .estimators import numeric_code_identity
    cfg=config.get("power",{});pilot_crossed=pilot_crossed if pilot_crossed is not None else cfg.get("pilot_crossed")
    if pilot_crossed is None:raise ValueError("S3 prospective power requires excluded pilot_crossed_contrasts.npz")
    energy_eligible=True
    if isinstance(pilot_crossed,(str,Path)):
        with np.load(pilot_crossed,allow_pickle=False) as loaded:
            pilot=np.stack([loaded["pred"],loaded["ctx"]],-1)
            if "eligible" in loaded:energy_eligible=bool(loaded["eligible"])
    elif isinstance(pilot_crossed,dict):pilot=np.stack([pilot_crossed["pred"],pilot_crossed["ctx"]],-1)
    else:pilot=np.asarray(pilot_crossed,float)
    components,diagnostics=crossed_variance_components(pilot)
    reps=int(limit or cfg.get("replicates",5000));seed=int(cfg.get("s3_seed",2527101600))
    prepare_run(output,config,"s3_power",provenance={"pilot_identity":identity(pilot),"pilot_energy_eligible":energy_eligible,"replicates":reps,"numeric_code_identity":numeric_code_identity()})
    if not energy_eligible:
        result={"status":"pilot_context_ineligible","selected_families":None,"selected_orbits":None,
                "pilot_diagnostics":diagnostics,"scenarios":[],"not_a_software_failure":True}
        atomic_json(Path(output)/"s3_power.json",result);return result
    inflation=float(cfg.get("variance_inflation",2.));alpha=config.get("statistics",{}).get("s3_alpha",.01)
    margin=config.get("statistics",{}).get("margin",.01);equiv=config.get("statistics",{}).get("equivalence",.005)
    alt=float(cfg.get("alternative",.02));scenarios=[]
    families=cfg.get("test_families",[32,48,64]);orbits=cfg.get("test_orbits",[16,24])
    aliases={"pilot":"normal","student_t5":"t5","high_variance_mixture":"mixture"}
    for f in families:
        for o in orbits:
            for correlation in cfg.get("s3_correlations",["estimated",0.,.8]):
                for requested in cfg.get("stress",["normal","t5","mixture"]):
                    stress=aliases.get(requested,requested)
                    if stress not in {"normal","t5","mixture"}:raise ValueError("Unknown crossed power stress")
                    desc={"families":f,"orbits":o,"correlation":correlation,"stress":stress}
                    sid=identity(desc);target=Path(output)/"scenarios"/(sid[:20]+".json")
                    if target.exists():scenarios.append(read_json(target));continue
                    rng=np.random.default_rng(np.random.SeedSequence([seed,int(sid[:8],16)]))
                    factors={}
                    for name,cov in components.items():
                        cov=cov*inflation
                        if correlation!="estimated":
                            sd=np.sqrt(np.diag(cov));cov=np.diag(sd)@np.array([[1,float(correlation)],[float(correlation),1]])@np.diag(sd)
                        e,v=np.linalg.eigh(_psd(cov));factors[name]=v@np.diag(np.sqrt(np.maximum(e,0)))
                    marginal=np.zeros(2,int);joint=0;equivalent=np.zeros(2,int)
                    for lo in range(0,reps,128):
                        count=min(128,reps-lo)
                        def draw(shape,name):
                            value=rng.normal(size=shape)@factors[name].T
                            if stress=="t5":value*=np.sqrt(3/rng.chisquare(5,size=shape[:-1]+(1,)))
                            return value
                        a=draw((count,int(f),1,2),"family");b=draw((count,1,int(o),2),"orbit");e=draw((count,int(f),int(o),2),"interaction")
                        if stress=="mixture":
                            scale=np.where(rng.random((count,int(f),1,1))<.1,3.,1.);a*=scale;e*=scale
                        x=a+b+e
                        reject=_crossed_vector_rejections(x+alt,alpha,margin,equiv)
                        marginal+=reject[:,:2].sum(0);joint+=np.all(reject[:,:2],1).sum()
                        equivalent+=_crossed_vector_rejections(x,alpha,margin,equiv)[:,2:].sum(0)
                    record={**desc,"replicates":reps,"df":min(f-1,o-1),"variance_inflation":inflation,
                            "marginal_superiority":[binomial_interval(k,reps) for k in marginal],
                            "joint_superiority":binomial_interval(joint,reps),
                            "equivalence_at_zero":[binomial_interval(k,reps) for k in equivalent]}
                    atomic_json(target,record);scenarios.append(record)
    passing=[]
    for f in families:
        for o in orbits:
            primary=[r for r in scenarios if r["families"]==f and r["orbits"]==o and r["stress"]=="normal" and r["correlation"]=="estimated"]
            if primary and all(all(v["interval"][0]>=cfg.get("marginal_target",.9) for v in r["marginal_superiority"]) and
                               r["joint_superiority"]["interval"][0]>=cfg.get("joint_target",.8) for r in primary):passing.append((f,o))
    selected=min(passing,key=lambda v:(v[0]*v[1],v[0],v[1])) if passing else (None,None)
    result={"method":"prospective_crossed_SE_df_Holm_power","alpha":alpha,"margin":margin,"alternative":alt,
            "components":{k:v.tolist() for k,v in components.items()},"pilot_diagnostics":diagnostics,"scenarios":scenarios,
            "selected_families":selected[0],"selected_orbits":selected[1],
            "status":"power_target_met" if passing else "limited_precision_do_not_claim_powered",
            "test_pool_must_be_sealed_before_pilot":True,"no_inner_bootstrap":True}
    atomic_json(Path(output)/"s3_power.json",result);return result
