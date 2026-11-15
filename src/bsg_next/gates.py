"""Protocol-specific completeness and calibration checks, never universal guarantees."""
from __future__ import annotations

from itertools import product
from pathlib import Path
import hashlib
import numpy as np
from .io import read_json,atomic_json,identity,prepare_run,file_sha256,canonical_json
from .statistics import binomial_interval,bootstrap_sensitivity,holm
from .estimators import numeric_code_identity

RATIOS=((1,0,1),(0,1,1),(1,1,1),(1,4,1),(4,1,1))


class CalibrationValidationError(RuntimeError):
    pass


def _require(condition,message):
    if not condition:raise CalibrationValidationError(message)


def _config_without_workers(config):
    result={k:v for k,v in config.items() if k not in {"worker_index","worker_count"}}
    result["synthetic"]={k:v for k,v in config.get("synthetic",{}).items() if k not in {"worker_index","worker_count"}}
    return result


def _marker(path,stage,config):
    path=Path(path);record=read_json(path/"run_identity.json")
    _require(record.get("stage")==stage,f"{path}: expected stage {stage}")
    _require(record.get("config")==_config_without_workers(config),f"{path}: calibration configuration differs from the confirmation configuration")
    descriptor={k:record[k] for k in ("schema_version","stage","config","provenance")}
    _require(identity(descriptor)==record.get("identity"),f"{path}: modified run identity")
    _require(record.get("provenance",{}).get("numeric_code_identity")==numeric_code_identity(),f"{path}: numerical code changed since calibration")
    return record


def _seed_records(path,marker,expected):
    path=Path(path);actual={};hashes=[]
    for file in sorted((path/"seeds").glob("cell_*_rep_*.json")):
        if file.name.endswith(".failed.json"):continue
        raw=file.read_bytes();r=__import__("json").loads(raw)
        _require(r.get("status")=="complete",f"{file}: incomplete seed")
        _require(r.get("run_identity")==marker["identity"],f"{file}: seed belongs to another run")
        key=(tuple(r["cell"]),int(r["replicate"]))
        _require(key in expected,f"{file}: unregistered cell/replicate")
        _require(key not in actual,f"{file}: duplicate scientific seed")
        _require(r["seed"]==expected[key],f"{file}: seed namespace mismatch")
        # Recompute Holm rejects from recorded raw p-values, not a claimed PASS.
        tests=r["decisions"]["tests"]
        _require(len(tests)==8,f"{file}: primary multiplicity family is incomplete")
        checked=holm({k:v["p"] for k,v in tests.items()},.04)
        _require(all(checked[k]["reject"]==v["reject"] for k,v in tests.items()),f"{file}: inconsistent Holm decision")
        joint_superior=all(v["reject"] for k,v in checked.items() if k.endswith(".superiority"))
        _require(joint_superior==r["decisions"]["both_groups_both_endpoints_superior"],f"{file}: positive sensitivity flag disagrees with component tests")
        false=[]
        for group,truth in r["truth"].items():
            for endpoint in ("pred","ctx"):
                value=truth[endpoint];_require(value is not None and np.isfinite(value),f"{file}: unknown fitted-estimator truth")
                for claim,null in (("superiority",value<=.01),("equivalence",abs(value)>=.005)):
                    keytest=f"{group}.{endpoint}.{claim}"
                    if null and tests[keytest]["reject"]:false.append(keytest)
        _require(set(false)==set(r["false_null_rejections"]) and bool(false)==bool(r["statistical_type_I"]),f"{file}: true-null error flag disagrees with estimand/decision")
        actual[key]=r;hashes.append((str(file.relative_to(path)),hashlib.sha256(raw).hexdigest()))
    missing=set(expected)-set(actual)
    _require(not missing,f"{path}: {len(missing)} registered datasets are missing (expected {len(expected)}, found {len(actual)})")
    return actual,identity(hashes)


def _expected(config,mode):
    from .synthetic import _cells
    cfg=config.get("synthetic",{});cells=_cells(mode,cfg)
    expected_cells=(set(product(("Y4",),("C7","D4"),(0.,.1),(1.,))) if mode=="positive" else
                    set(product(("Y1","Y2","Y3") if mode=="negative" else tuple(f"Y{i}" for i in range(1,7)),
                                ("C7","D4"),(.1,) if mode=="negative" else (0.,.1,.3),(1.,30.))))
    _require(set(cells)==expected_cells,f"{mode}: reduced/altered cell grid cannot qualify as the registered calibration")
    repetitions=int(cfg.get("negative_replicates",5000) if mode=="negative" else cfg.get("replicates_per_cell",cfg.get("replicates",200)))
    required=5000 if mode=="negative" else 200
    _require(repetitions==required,f"{mode}: requires {required} independent datasets/cell, not {repetitions}")
    base=int(cfg.get(("calibration" if mode=="positive" else mode)+"_seed",2227101600 if mode=="negative" else 2027101600))
    full_cells=_cells("calibration",cfg) if mode=="positive" else cells
    return {(tuple(cell),r):base+100000*full_cells.index(cell)+r for cell in cells for r in range(required)}


def _positive_results(records):
    results=[];failures=[]
    for cell in sorted({key[0] for key in records if key[0][0]=="Y4" and key[0][2] in (0.,.1) and key[0][3]==1.}):
        rs=[r for (c,_),r in records.items() if c==cell]
        detection=binomial_interval(sum(r["decisions"]["both_groups_both_endpoints_superior"] for r in rs),len(rs))
        results.append({"cell":cell,"sensitivity":detection})
        if detection["rate"]<.9:failures.append(f"Positive cell {cell}: sensitivity {detection['rate']:.5f} below .90")
    _require(len(results)==4,"Expected all four registered positive cells")
    _require(not failures,"; ".join(failures))
    return results


def validate_positive_calibration(config,directory,*,sentinel=True):
    """Early fail-fast check; success never replaces the complete scientific gate."""
    _require(not config.get("technical_test",False),"Technical outputs cannot pass a production positive gate")
    stage="synthetic_positive" if sentinel else "synthetic_calibration"
    marker=_marker(directory,stage,config)
    records,digest=_seed_records(directory,marker,_expected(config,"positive" if sentinel else "calibration"))
    return {"status":"passed_positive_sentinel_only" if sentinel else "passed_positive_checks_only",
            "positive":_positive_results(records),"seed_manifest_identity":digest,
            "full_calibration_passed":False,"confirmation_authorized":False}


def validate_calibration(config,calibration_dir,negative_dir,crossed_dir):
    """Strict seed-level gate for the registered 72/12/120 grids.

    S3 threshold .02 is a preregistered approximate-calibration tolerance around
    nominal .01, not a theorem granting arbitrary-population Type-I control.
    """
    _require(not config.get("technical_test",False),"Technical smoke runs cannot pass scientific calibration gates")
    st=config.get("statistics",{})
    _require(st.get("main_alpha",.04)==.04 and st.get("s3_alpha",.01)==.01,"Changed multiplicity allocations require a new calibration protocol")
    _require(st.get("margin",.01)==.01 and st.get("equivalence",.005)==.005,"Changed estimand margins require a new calibration protocol")
    _require(int(config.get("synthetic",{}).get("dimension",16))==16,"Primary calibration dimension must be 16")
    calmarker=_marker(calibration_dir,"synthetic_calibration",config)
    negmarker=_marker(negative_dir,"synthetic_negative",config)
    cal,calhash=_seed_records(calibration_dir,calmarker,_expected(config,"calibration"))
    neg,neghash=_seed_records(negative_dir,negmarker,_expected(config,"negative"))
    negative=[]
    for cell in sorted({key[0] for key in neg}):
        rs=[r for (c,_),r in neg.items() if c==cell]
        interval=binomial_interval(sum(r["statistical_type_I"] for r in rs),len(rs))
        _require(interval["upper_one_sided"]<=.06,f"Negative cell {cell}: true-null error upper95={interval['upper_one_sided']:.5f} exceeds .06")
        negative.append({"cell":cell,"statistical_type_I":interval})
    positive=_positive_results(cal)
    crossmarker=_marker(crossed_dir,"crossed_calibration",config)
    _require(crossmarker.get("provenance",{}).get("replicates")==2000,"Crossed calibration requires 2000 replicates/scenario")
    cross=[];crosshash=[];cfg=config.get("synthetic",{})
    _require(set(cfg.get("crossed_families",[32,48,64]))=={32,48,64},"Crossed family grid was reduced")
    _require(set(cfg.get("crossed_orbits",[16,24]))=={16,24},"Crossed orbit grid was reduced")
    _require({tuple(x) for x in cfg.get("crossed_variances",RATIOS)}==set(RATIOS),"Crossed variance-ratio grid was reduced")
    _require(set(cfg.get("crossed_correlations",[0.,.8]))=={0.,.8},"Crossed correlation grid was reduced")
    primary=list(product((32,48,64),(16,24),RATIOS,(False,True),(0.,.8)))
    _require(len(primary)==120,"Internal crossed primary grid error")
    tolerance=float(cfg.get("crossed_gate_upper_tolerance",.02))
    _require(tolerance==.02,"Changed crossed gate tolerance requires protocol revision")
    scenario_positions={}
    complete_grid=product(cfg.get("crossed_families",[32,48,64]),cfg.get("crossed_orbits",[16,24]),
                          cfg.get("crossed_variances",RATIOS),(False,True),cfg.get("crossed_correlations",[0.,.8]),
                          cfg.get("crossed_heteroscedastic",[False,True]))
    for i,(ff,oo,rr,hh,cc,hetero) in enumerate(complete_grid):
        desc={"families":ff,"orbits":oo,"variance_ratio":list(rr),"heavy_t5":hh,"correlation":float(cc),"heteroscedastic":hetero}
        scenario_positions[identity(desc)[:16]]=i
    for f,o,ratio,heavy,corr in primary:
        scenario={"families":f,"orbits":o,"variance_ratio":list(ratio),"heavy_t5":heavy,"correlation":corr,"heteroscedastic":False}
        sid=identity(scenario)[:16];directory=Path(crossed_dir)/"seeds"/sid;records=[]
        _require(sid in scenario_positions,"Homoscedastic primary scenario omitted from crossed calibration")
        for rep in range(2000):
            file=directory/f"rep_{rep:05d}.json"
            _require(file.is_file(),f"Crossed scenario {scenario}: missing replicate {rep}")
            raw=file.read_bytes();r=__import__("json").loads(raw)
            _require(r.get("run_identity")==crossmarker["identity"] and r.get("replicate")==rep,f"{file}: crossed seed identity mismatch")
            expected_seed=int(cfg.get("crossed_seed",2327101600))+100000*scenario_positions[sid]+rep
            _require(r.get("seed")==expected_seed,f"{file}: crossed seed namespace mismatch")
            records.append(r);crosshash.append((str(file.relative_to(crossed_dir)),hashlib.sha256(raw).hexdigest()))
        for field in ("rejected","false_equivalence"):
            rate=binomial_interval(sum(r[field] for r in records),len(records))
            _require(rate["upper_one_sided"]<=tolerance,f"Crossed {scenario}/{field}: approximate FWER upper95 exceeds {tolerance}")
        cross.append({"scenario":scenario,"superiority_boundary_fwer":binomial_interval(sum(r["rejected"] for r in records),2000),
                      "equivalence_boundary_fwer":binomial_interval(sum(r["false_equivalence"] for r in records),2000)})
    return {"status":"passed_registered_calibration","calibration_datasets":len(cal),"negative_datasets":len(neg),
            "crossed_primary_scenarios":len(cross),"crossed_datasets":240000,
            "seed_manifest_identities":{"calibration":calhash,"negative":neghash,"crossed":identity(crosshash)},
            "negative":negative,"positive":positive,"crossed":cross,
            "rules":{"negative_one_sided_upper95_max":.06,"positive_empirical_sensitivity_min":.9,
                     "S3_nominal_alpha":.01,"S3_approximate_one_sided_upper95_tolerance":tolerance},
            "scope":"specified generators and approximate crossed model only; no universal Type-I guarantee"}


def run_crossed_bootstrap(config,output,limit=None):
    """The prespecified five-case ×50-seed ×999-resample S3 sensitivity subset."""
    cfg=config.get("synthetic",{});repetitions=int(cfg.get("crossed_bootstrap_datasets",50));resamples=int(cfg.get("bootstrap_replicates",999))
    marker=prepare_run(output,_config_without_workers(config),"crossed_bootstrap_sensitivity",
                       provenance={"numeric_code_identity":numeric_code_identity(),"gate_code_sha256":file_sha256(__file__)})
    base=int(cfg.get("crossed_bootstrap_seed",2627101600));done=0;records=[]
    worker=int(cfg.get("worker_index",0));workers=int(cfg.get("worker_count",1))
    _require(0<=worker<workers,"Invalid crossed bootstrap worker shard")
    for ci,ratio in enumerate(RATIOS):
        for rep in range(repetitions):
            if (ci*repetitions+rep)%workers!=worker:continue
            if limit is not None and done>=limit:break
            target=Path(output)/"seeds"/f"case_{ci:02d}_rep_{rep:04d}.json"
            if target.exists():
                old=read_json(target);_require(old["run_identity"]==marker["identity"],"Crossed bootstrap seed identity mismatch")
                records.append(old);done+=1;continue
            seed=base+100000*ci+rep;rng=np.random.default_rng(seed)
            x=(rng.normal(size=(32,1,2))*np.sqrt(ratio[0])+rng.normal(size=(1,16,2))*np.sqrt(ratio[1])+
               rng.normal(size=(32,16,2))*np.sqrt(ratio[2]))*.02
            sensitivity={ep:bootstrap_sensitivity(x[:,:,j],replicates=resamples,seed=seed,crossed=True) for j,ep in enumerate(("pred","ctx"))}
            record={"run_identity":marker["identity"],"seed":seed,"case":ci,"replicate":rep,"variance_ratio":ratio,
                    "families":32,"orbits":16,"normal_effects":True,"sensitivity":sensitivity,
                    "coverage":{k:v["percentile_ci95"][0]<=0<=v["percentile_ci95"][1] for k,v in sensitivity.items()}}
            atomic_json(target,record);records.append(record);done+=1
        if limit is not None and done>=limit:break
    allrecords=[read_json(p) for p in sorted((Path(output)/"seeds").glob("case_*.json"))]
    cases=[]
    for ci,ratio in enumerate(RATIOS):
        subset=[r for r in allrecords if r["case"]==ci]
        cases.append({"variance_ratio":ratio,"datasets":len(subset),"coverage":{ep:binomial_interval(sum(r["coverage"][ep] for r in subset),len(subset)) for ep in ("pred","ctx")}})
    result={"sensitivity_only":True,"expected_datasets":5*repetitions,"completed_datasets":len(allrecords),"resamples_per_dataset":resamples,
            "all_complete":len(allrecords)==5*repetitions,"cases":cases,"not_a_rare_tail_coverage_guarantee":True}
    atomic_json(Path(output)/f"summary_worker_{worker:03d}.json",result);return result
