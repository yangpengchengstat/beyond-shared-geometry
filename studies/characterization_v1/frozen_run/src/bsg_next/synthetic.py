"""Known-truth synthetic calibration, sharded identity-bound atomic seed resume."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import itertools
import time
import tracemalloc
import numpy as np
from .groups import get_group
from .structural import analyze_cell
from .statistics import family_decisions, binomial_interval, bootstrap_sensitivity, run_power, run_crossed_calibration
from .io import prepare_run, atomic_json, read_json, identity

REGIMES=("Y1","Y2","Y3","Y4","Y5","Y6")


def generate_dataset(group="C7", regime="Y4", noise=.1, condition=1., seed=42,
                     train_families=48, validation_families=16, test_families=32, dim=16):
    group=get_group(group) if isinstance(group,str) else group
    k=group.state_count; d=int(dim);regime=str(regime).split("_")[0]
    if regime not in REGIMES:raise ValueError("Unknown known-truth regime")
    if d<2*k:raise ValueError("This block construction requires d >= twice the state count")
    if noise<0 or condition<1:raise ValueError("Invalid noise/conditioning")
    rng=np.random.default_rng(seed)
    u,_=np.linalg.qr(rng.normal(size=(d,d)));v,_=np.linalg.qr(rng.normal(size=(d,d)))
    a=(u*np.geomspace(1.,float(condition),d))@v.T
    mu=np.zeros((k,d));mu[:,:k]=np.eye(k)-np.ones((k,k))/k
    beta=np.zeros((k,d))
    if regime in {"Y4","Y5"}:beta[:,k:2*k]=np.eye(k)
    if regime=="Y2":beta[:,k]=1.
    # Fix interaction energy to half category energy BEFORE the coordinate map.
    category_energy=float(np.mean(np.sum(mu**2,1)))
    if np.any(beta):beta*=np.sqrt(.5*category_energy/np.mean(np.sum(beta**2,1)))
    state_noise=.5*category_energy/d if regime=="Y3" else 0.
    transformed_mu=mu@a;transformed_beta=beta@a
    cov_context=np.zeros((k,k,d,d))
    for x in range(k):
        for y in range(k):
            if regime in {"Y2","Y4"}:cov_context[x,y]=np.outer(transformed_beta[x],transformed_beta[y])
            elif regime=="Y3" and x==y:cov_context[x,y]=state_noise*(a.T@a)
            elif regime=="Y5" and x==y:
                raw=np.zeros((d,d));raw[k:2*k,k:2*k]=np.eye(k)*float(np.sum(beta[x]**2))/k
                cov_context[x,y]=a.T@raw@a
    clean_energy=float(np.mean(np.sum(transformed_mu**2,1))+np.mean([np.trace(cov_context[x,x]) for x in range(k)]))
    noise_var=float(noise)*clean_energy/d
    rows=[];zs=[];family_index=0
    for split,count in (("train",train_families),("validation",validation_families),("test",test_families)):
        for _ in range(int(count)):
            family=f"{split}_{family_index:04d}";family_index+=1
            shared=rng.normal()*np.sqrt(.5)
            family_beta=beta.copy()
            if regime=="Y5":
                q,qr=np.linalg.qr(rng.normal(size=(k,k)));q=q*np.where(np.diag(qr)>=0,1.,-1.)
                family_beta[:,k:2*k]=beta[:,k:2*k]@q
            for paraphrase in range(2):
                context=shared+rng.normal()*np.sqrt(.5)
                for x in range(k):
                    clean=mu[x]+context*family_beta[x]
                    if regime=="Y3":clean=clean+rng.normal(size=d)*np.sqrt(state_noise)
                    z=clean@a+rng.normal(size=d)*np.sqrt(noise_var)
                    row={"row_id":f"{family}_{paraphrase}_{x}","family_id":family,"paraphrase_id":paraphrase,
                         "template_id":f"{family}_{paraphrase}","state_id":x,"orbit_id":"single","split":split}
                    rows.append(row);zs.append(z)
    true_ops={}
    ainv=np.linalg.inv(a)
    for g in group.elements:
        p=np.eye(k)[[group.apply(g,x) for x in range(k)]]
        r=np.eye(d);r[:k,:k]=p
        if regime=="Y4":r[k:2*k,k:2*k]=p
        if regime=="Y6":r[2*k:,2*k:]=rng.normal(size=(d-2*k,d-2*k))*10
        true_ops[g]=ainv@r@a
    return {"z":np.asarray(zs),"rows":rows,"group":group,"regime":regime,
            "mu":transformed_mu,"context_cov":cov_context,"observation_noise_variance":noise_var,
            "pair_context_factor":2. if regime=="Y3" else 1.,"true_operators":true_ops,
            "coordinate_condition":float(np.linalg.cond(a)),"noise_energy_ratio":float(noise),"seed":int(seed)}


def expected_linear_loss(mx,my,sxx,syy,sxy,coef,bias):
    """E||Y-XB-b||² from known row-vector first/second moments."""
    mean=my-mx@coef-bias
    value=np.dot(mean,mean)+np.trace(syy)+np.trace(coef.T@sxx@coef)-2*np.trace(coef.T@sxy)
    return float(max(0.,value))


def known_estimands(dataset, analysis):
    """Conditional on fitted train/validation estimators; no test outcomes used."""
    fitted=analysis["_fitted"];group=dataset["group"];k=group.state_count;d=dataset["z"].shape[1]
    eye=np.eye(d);noise=dataset["observation_noise_variance"];mu=dataset["mu"];cov=dataset["context_cov"]
    pred=[];ctx=[];baseline=fitted["baseline"];centroids=fitted["centroids"]
    for g in group.elements:
        for x in range(k):
            y=group.apply(g,x)
            if x==y:continue
            sxx=cov[x,x]+noise*eye;syy=cov[y,y]+noise*eye;sxy=cov[x,y]
            if baseline=="centroid":bcoef=np.zeros((d,d));bias=centroids[y]
            else:
                alpha=1. if baseline=="displacement" else float(baseline.split("_")[1])
                bcoef=alpha*eye;bias=centroids[y]-alpha*centroids[x]
            op=fitted["operators"][g]
            pred.append((expected_linear_loss(mu[x],mu[y],sxx,syy,sxy,bcoef,bias)-
                         expected_linear_loss(mu[x],mu[y],sxx,syy,sxy,op.coef,op.intercept))/analysis["training_scale"])
            factor=dataset["pair_context_factor"];xx=factor*cov[x,x]+2*noise*eye;yy=factor*cov[y,y]+2*noise*eye;xy=factor*cov[x,y]
            qm=fitted["difference_operators"][g]
            ctx.append((expected_linear_loss(np.zeros(d),np.zeros(d),xx,yy,xy,fitted["context_alpha"]*eye,np.zeros(d))-
                        expected_linear_loss(np.zeros(d),np.zeros(d),xx,yy,xy,qm.coef,qm.intercept))/max(analysis["context_energy"],1e-12))
    return {"pred":float(np.mean(pred)),"ctx":float(np.mean(ctx)),
            "truth_scope":"conditional fitted-estimator population moments, not semantic ground truth"}


def _as_family(analysis):
    p=analysis["family_orbit_prediction"];q=analysis["family_orbit_context"];keys=sorted(p)
    return {"pred":[p[k] for k in keys],"ctx":[q[k] for k in keys],"eligible":analysis["eligible"]}


def simulate_seed(config,cell,replicate,seed,companion_positive=False):
    cfg=config.get("synthetic",{});regime,designated,noise,condition=cell
    summaries={};contrasts={};truth={};start=time.perf_counter()
    for gi,group in enumerate(("C7","D4")):
        use_regime="Y4" if companion_positive and group!=designated else regime
        ds=generate_dataset(group,use_regime,noise,condition,
                            seed=np.random.SeedSequence([int(seed),gi]).generate_state(1)[0],
                            train_families=cfg.get("train_families",48),validation_families=cfg.get("validation_families",16),
                            test_families=cfg.get("test_families",32),dim=cfg.get("dimension",16))
        analysis=analyze_cell(ds["z"],ds["rows"],ds["group"],{"secondary":[],"lambda_grid":cfg.get("lambda_grid",LAMBDA_GRID),
                              "context_energy_floor":config.get("statistics",{}).get("context_energy_floor",.05)},return_models=True)
        truth[group]=known_estimands(ds,analysis);analysis.pop("_fitted")
        contrasts[group]=_as_family(analysis)
        summaries[group]={k:analysis[k] for k in ("mean_losses","primary_baseline","selection","context_selection","eligible","training_scale","context_energy_ratio")}
    decisions=family_decisions(contrasts,alpha=config.get("statistics",{}).get("main_alpha",.04))
    true_null_rejections=[]
    for group in contrasts:
        for endpoint in ("pred","ctx"):
            val=truth[group][endpoint]
            for claim,is_null in (("superiority",val<=.01),("equivalence",abs(val)>=.005)):
                # An ineligible context endpoint is never declared significant.
                key=f"{group}.{endpoint}.{claim}"
                if is_null and decisions["tests"][key]["reject"]:true_null_rejections.append(key)
    key="D4" if designated.upper().startswith("D4") else "C7"
    beyond=any(decisions["tests"][f"{key}.{ep}.superiority"]["reject"] for ep in ("pred","ctx"))
    return {"cell":list(cell),"replicate":replicate,"seed":int(seed),"companion_positive":companion_positive,
            "truth":truth,"summaries":summaries,"decisions":decisions,"family_contrasts":contrasts,
            "statistical_type_I":bool(true_null_rejections),"false_null_rejections":true_null_rejections,
            "category_identification_counterexample":bool(beyond and regime in {"Y1","Y2","Y3"}),
            "seconds":time.perf_counter()-start,"no_inner_bootstrap":True}


def _cells(mode,cfg=None):
    cfg=cfg or {};groups=cfg.get("groups",["C7","D4"]);regimes=cfg.get("regimes",REGIMES)
    if mode=="positive":return list(itertools.product(["Y4"],groups,(0.,.1),(1.,)))
    if mode in {"negative","bootstrap"}:return list(itertools.product([r for r in regimes if r in REGIMES[:3]],groups,(.1,),cfg.get("conditions",[1.,30.])))
    if mode=="timing":return list(itertools.product(regimes,groups,(.1,),(1.,)))
    return list(itertools.product(regimes,groups,cfg.get("noise_ratios",[0.,.1,.3]),cfg.get("conditions",[1.,30.])))


def run_synthetic(config,output,mode="timing",limit=None,worker_index=None,worker_count=None,**kwargs):
    if mode=="power":return run_power(config,output,limit=limit)
    if mode in {"crossed","crossed_calibration","crossed_bootstrap"}:
        local=dict(config);local["synthetic"]=dict(config.get("synthetic",{}))
        if worker_index is not None:local["synthetic"]["worker_index"]=worker_index
        if worker_count is not None:local["synthetic"]["worker_count"]=worker_count
        if mode!="crossed_bootstrap":return run_crossed_calibration(local,output,limit=limit)
    if mode=="crossed_bootstrap":
        from .gates import run_crossed_bootstrap
        return run_crossed_bootstrap(local,output,limit=limit)
    cfg=config.get("synthetic",{});mode={"full":"calibration","formal":"negative","smoke":"timing"}.get(mode,mode)
    if mode not in {"timing","positive","calibration","negative","bootstrap"}:raise ValueError("Unknown synthetic stage")
    worker_index=int(cfg.get("worker_index",0) if worker_index is None else worker_index)
    worker_count=int(cfg.get("worker_count",1) if worker_count is None else worker_count)
    if not 0<=worker_index<worker_count:raise ValueError("Invalid worker shard")
    # Worker assignment is NOT part of scientific/config identity.
    clean={k:v for k,v in config.items() if k not in {"worker_index","worker_count"}}
    clean["synthetic"]={k:v for k,v in cfg.items() if k not in {"worker_index","worker_count"}}
    from .estimators import numeric_code_identity
    marker=prepare_run(output,clean,f"synthetic_{mode}",provenance={"numeric_code_identity":numeric_code_identity()})
    repetitions=int(cfg.get({"timing":"timing_replicates","positive":"calibration_replicates","calibration":"calibration_replicates","negative":"negative_replicates","bootstrap":"bootstrap_datasets"}[mode],
                           {"timing":10,"positive":200,"calibration":200,"negative":5000,"bootstrap":50}[mode]))
    if mode in {"positive","calibration"}:repetitions=int(cfg.get("replicates_per_cell",cfg.get("replicates",repetitions)))
    if mode=="bootstrap":repetitions=int(cfg.get("bootstrap_datasets_per_cell",repetitions))
    bases={"timing":1927101600,"positive":2027101600,"calibration":2027101600,"negative":2227101600,"bootstrap":2427101600}
    base=int(cfg.get(("calibration" if mode=="positive" else mode)+"_seed",bases[mode]));total=0;done=0;start=time.perf_counter()
    # Python allocation tracing can dominate these small numerical fits. It is
    # an optional profiler, not part of the estimator or scientific decisions.
    trace_allocations=bool(config.get("execution",{}).get("trace_python_allocations",False))
    own_tracer=trace_allocations and not tracemalloc.is_tracing()
    if own_tracer:tracemalloc.start()
    cells=_cells(mode,cfg)
    # Sentinel cells retain the original calibration cell indices and seeds.
    # A subset is an early stop check, never a substitute for full calibration.
    full_cells=_cells("calibration",cfg) if mode=="positive" else cells
    for ci,cell in enumerate(cells):
        source_ci=full_cells.index(cell)
        for ri in range(repetitions):
            index=ci*repetitions+ri
            if index%worker_count!=worker_index:continue
            if limit is not None and done>=int(limit):break
            seed=base+100000*source_ci+ri
            target=Path(output)/"seeds"/f"cell_{source_ci:03d}_rep_{ri:05d}.json"
            if target.exists():
                old=read_json(target)
                if old.get("seed")!=seed or old.get("run_identity")!=marker["identity"]:raise ValueError("Seed record identity mismatch")
                done+=1;continue
            try:
                record=simulate_seed(config,cell,ri,seed,companion_positive=mode=="negative" and ri%2==1)
                if mode=="bootstrap":
                    record["bootstrap_sensitivity"]={g:{ep:bootstrap_sensitivity(e[ep],replicates=999,seed=seed)
                                                          for ep in ("pred","ctx")} for g,e in record["family_contrasts"].items()}
                record.update({"status":"complete","run_identity":marker["identity"]})
            except Exception as exc:
                atomic_json(target.with_suffix(".failed.json"),{"status":"failed","seed":seed,"run_identity":marker["identity"],"error":repr(exc)})
                if own_tracer:tracemalloc.stop()
                raise
            atomic_json(target,record);done+=1
            if done%50==0:
                atomic_json(Path(output)/f"progress_worker_{worker_index:03d}.json",
                            {"stage":mode,"worker_index":worker_index,"worker_count":worker_count,
                             "completed_worker_datasets":done,"wall_seconds":time.perf_counter()-start,
                             "expected_total_datasets":len(cells)*repetitions})
                print(f"PROGRESS {mode} worker={worker_index} completed={done} total_all_workers={len(cells)*repetitions}",flush=True)
        if limit is not None and done>=int(limit):break
    groups=defaultdict(list)
    for p in sorted((Path(output)/"seeds").glob("cell_*_rep_*.json")):
        if p.name.endswith(".failed.json"):continue
        r=read_json(p);groups[tuple(r["cell"])].append(r)
    summaries=[];timings=[]
    for cell,rs in groups.items():
        timings.extend(r["seconds"] for r in rs)
        summaries.append({"cell":cell,"datasets":len(rs),"statistical_type_I":binomial_interval(sum(r["statistical_type_I"] for r in rs),len(rs)),
                          "category_identification_counterexamples":binomial_interval(sum(r["category_identification_counterexample"] for r in rs),len(rs)),
                          "complete_null":binomial_interval(sum(r["statistical_type_I"] for r in rs if not r["companion_positive"]),sum(not r["companion_positive"] for r in rs)),
                          "partial_null":binomial_interval(sum(r["statistical_type_I"] for r in rs if r["companion_positive"]),sum(r["companion_positive"] for r in rs))})
        if cell[0]=="Y4" and cell[2] in (0.,.1) and cell[3]==1.:
            summaries[-1]["positive_sensitivity"]=binomial_interval(sum(r["decisions"]["both_groups_both_endpoints_superior"] for r in rs),len(rs))
    peak=tracemalloc.get_traced_memory()[1] if trace_allocations else None
    if own_tracer:tracemalloc.stop()
    result={"stage":mode,"expected_datasets":len(cells)*repetitions,"completed_datasets":sum(len(v) for v in groups.values()),
            "worker_index":worker_index,"worker_count":worker_count,"cells":summaries,"peak_traced_memory_bytes":peak,
            "wall_seconds":time.perf_counter()-start,"median_dataset_seconds":float(np.median(timings)) if timings else None,
            "p95_dataset_seconds":float(np.quantile(timings,.95)) if timings else None,
            "python_allocation_tracing_enabled":trace_allocations,
            "memory_measurement_scope":"Python traced allocations only" if trace_allocations else "not measured; allocation tracing disabled",
            "no_inner_bootstrap":mode!="bootstrap","all_complete":sum(len(v) for v in groups.values())==len(cells)*repetitions}
    if timings:result["formal_60000_cpu_hour_projection_with_2x_reserve"]=float(np.quantile(timings,.95)*60000*2/3600)
    complete=result["all_complete"]
    result["negative_calibration_gate"]=("pass" if complete and mode=="negative" and all(c["statistical_type_I"]["upper_one_sided"]<=.06 for c in summaries) else "fail" if complete and mode=="negative" else "pending_not_a_complete_negative_audit")
    positives=[c["positive_sensitivity"] for c in summaries if "positive_sensitivity" in c]
    result["positive_sensitivity_gate"]=("pass" if complete and mode in {"positive","calibration"} and positives and all(p["rate"]>=.9 for p in positives) else "fail" if complete and mode in {"positive","calibration"} else "pending_not_a_complete_calibration")
    atomic_json(Path(output)/f"summary_worker_{worker_index:03d}.json",result)
    return result


# Export the registered grid for CLI planning without executing a single fit.
def synthetic_plan():
    return {"timing":120,"calibration":14400,"negative":60000,
            "bootstrap_dataset_subset":600,"bootstrap_per_subset_dataset":999,
            "formal_audit_inner_bootstrap":False}


from .estimators import LAMBDA_GRID
