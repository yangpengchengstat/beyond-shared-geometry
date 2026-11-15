"""Read-only archived FULL replay and prospective retrospective controls.

ZIP members are read in memory; nothing is extracted into a source tree. Saved
coordinates and historical operators are replayed separately from new refits.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import io
import json
import zipfile
from types import SimpleNamespace
import numpy as np
from .io import prepare_run, atomic_json, read_json, identity, file_sha256
from .structural import analyze_cell, fit_train_pca
from .groups import get_group


def load_archive_cell(archive,index_entry):
    """Return saved arrays and member hashes without executing pickled objects."""
    owns=not isinstance(archive,zipfile.ZipFile);zf=zipfile.ZipFile(archive) if owns else archive
    try:
        result={"fields":index_entry["fields"],"split":index_entry["split"],"members":{}}
        for label,key in (("base","base_member"),("joint","joint_member")):
            name=index_entry.get(key)
            if not name:continue
            raw=zf.read(name)
            with np.load(io.BytesIO(raw),allow_pickle=False) as f:
                arrays={k:f[k] for k in f.files}
            result[label]=arrays;result["members"][label]={"path":name,"sha256":hashlib.sha256(raw).hexdigest(),"bytes":len(raw)}
        return result
    finally:
        if owns:zf.close()


def historical_replay(cell):
    """The unchanged legacy metric functions evaluate saved operators, no refit."""
    from bsg.experiments.main_experiment import _group_specs
    from bsg.experiments.canonical_units import evaluate_operators
    from bsg.representations.pca import PCASubspace
    spec=_group_specs()[cell["fields"]["group"]];frame=spec.frame
    split=cell["split"]
    partition=SimpleNamespace(seed=split["seed"],train_ids=tuple(split["train_ids"]),
                              validation_ids=tuple(split["validation_ids"]),test_ids=tuple(split["test_ids"]))
    for part in ("train","validation","test"):
        setattr(partition,part+"_mask",frame.template_id.isin(split[part+"_ids"]).to_numpy())
    rows=[]
    for label,method in (("base","independent_ridge"),("joint","group_consistent_joint")):
        if label not in cell:continue
        data=cell[label];pca=PCASubspace(int(cell["fields"]["latent_dim"]))
        pca.components_=data["pca_components"];pca.mean_=data["pca_mean"]
        pca.explained_variance_ratio_=data["pca_explained"]
        operators=dict(zip(spec.elements,data["operators"]))
        rows.extend(evaluate_operators(spec,{"pca":pca,"z":data["z"],"operators":operators},partition,method,
                                       optimization_seed=42 if label=="joint" else None,random_baseline_seed=None))
    for row in rows:
        row.update({k:v for k,v in cell["fields"].items() if k not in {"optimization_seed","random_baseline_seed"}})
    return rows


def verify_historical_replay(rows,reference):
    import pandas as pd
    keys=["model","group","layer","latent_dim","split_seed","data_split","method","template_id"]
    lookup={tuple(str(r[k]) for k in keys):r for r in reference.to_dict("records")}
    checked=0;maximum=0.
    for row in rows:
        key=tuple(str(row[k]) for k in keys)
        if key not in lookup:raise ValueError(f"Archived replay row missing from source CSV: {key}")
        target=lookup[key]
        for metric,value in row.items():
            if not (metric.startswith("E_") or metric=="pca_variance") or metric not in target:continue
            actual=float(target[metric])
            if not np.isfinite(actual) and not np.isfinite(value):continue
            error=abs(float(value)-actual);maximum=max(maximum,error);checked+=1
            if not np.isclose(value,actual,rtol=1e-8,atol=1e-10):
                raise ValueError(f"Historical metric replay discrepancy {key}/{metric}: {value} vs {actual}")
    return {"status":"matched_source_CSV","checked_values":checked,"max_absolute_difference":maximum,
            "relative_tolerance":1e-8,"absolute_tolerance":1e-10}


def retrospective_rows(fields,split):
    """Predeclared pairing of old templates, only for exploratory difference checks."""
    from bsg.data.c7 import build_dataset as c7
    from bsg.data.d4 import build_dataset as d4
    frame=c7() if fields["group"]=="C7" else d4()
    assignment={}
    for name in ("train","validation","test"):
        ids=sorted(split[name+"_ids"])
        if len(ids)%2:raise ValueError("Retrospective pairing needs an even template count")
        for j,t in enumerate(ids):assignment[t]=(name,f"old_{name}_{j//2:03d}",j%2)
    rows=[]
    for original in frame.to_dict("records"):
        name,family,paraphrase=assignment[int(original["template_id"])]
        rows.append({**original,"row_id":f"old_{original['template_id']}_{original['state_id']}",
                     "original_template_id":original["template_id"],"family_id":family,
                     "paraphrase_id":paraphrase,"orbit_id":"single","split":name})
    return rows


def _ledger_hashes(zf):
    candidates=[n for n in zf.namelist() if n.endswith("run_ledger.json")]
    if not candidates:return {}
    output={}
    def visit(value):
        if isinstance(value,dict):
            path=value.get("path",value.get("artifact_path"));digest=value.get("sha256",value.get("artifact_sha256"))
            if path and digest:output[str(path)]=str(digest)
            for v in value.values():visit(v)
        elif isinstance(value,list):
            for v in value:visit(v)
    for name in candidates:visit(json.loads(zf.read(name)))
    return output


def run_offline(config,output,archive=None,index=None,mode="pilot",limit=None,**kwargs):
    cfg=config.get("offline",{});archive=Path(archive or cfg.get("archive",""))
    if not archive.is_file():raise FileNotFoundError("Provide the immutable historical FULL ZIP via offline.archive")
    default=Path(__file__).resolve().parents[2]/"docs/prospective/icml2027_identification_v1/evidence/CELL_INPUT_INDEX.json"
    index=Path(index or cfg.get("index",default));entries=read_json(index)["cells"]
    chosen=[]
    for entry in entries:
        f=entry["fields"]
        if f.get("latent_dim")!=16 or f.get("depth_slot") not in (.2,.4,.6,.8):continue
        if mode in {"pilot","smoke"} and not (f["model_key"] in {"llama_1b","qwen_1p5b","qwen_1_5b","qwen_1.5b"}
                  and f["depth_slot"] in (.4,.8) and f["split_seed"] in (101,303)):continue
        chosen.append(entry)
    if mode in {"pilot","smoke"} and not chosen:
        # Match pinned model IDs too, avoiding an undocumented key spelling dependency.
        chosen=[e for e in entries if e["fields"].get("latent_dim")==16 and e["fields"].get("depth_slot") in (.4,.8)
                and e["fields"].get("split_seed") in (101,303) and e["fields"].get("model") in
                {"meta-llama/Llama-3.2-1B","Qwen/Qwen2.5-1.5B"}]
    if limit is not None:chosen=chosen[:int(limit)]
    with zipfile.ZipFile(archive) as zf:
        directory=[(i.filename,i.CRC,i.file_size,i.compress_size) for i in zf.infolist()]
        provenance={"archive_bytes":archive.stat().st_size,"zip_directory_identity":identity(directory),"index_sha256":file_sha256(index)}
        if cfg.get("verify_archive_sha256",False):
            provenance["archive_sha256"]=file_sha256(archive)
            expected=cfg.get("archive_expected_sha256",cfg.get("expected_archive_sha256"))
            if expected and provenance["archive_sha256"]!=expected:raise ValueError("Historical archive hash mismatch")
        from .estimators import numeric_code_identity
        provenance["numeric_code_identity"]=numeric_code_identity()
        prepare_run(output,config,f"offline_{mode}",provenance=provenance)
        ledger=_ledger_hashes(zf);summaries=[]
        import pandas as pd
        reference_names=[n for n in zf.namelist() if n.endswith("/structural/template_metrics.csv")]
        if len(reference_names)!=1:raise ValueError("Historical replay requires one authoritative structural/template_metrics.csv")
        reference=pd.read_csv(io.BytesIO(zf.read(reference_names[0])))
        for entry in chosen:
            key=identity(entry)[:20];target=Path(output)/"cells"/(key+".json")
            if target.exists():summaries.append(read_json(target));continue
            cell=load_archive_cell(zf,entry)
            checks=[]
            for label,member in cell["members"].items():
                matching=[digest for path,digest in ledger.items() if member["path"].endswith(path) or path.endswith(member["path"])]
                if matching and any(d!=member["sha256"] for d in matching):raise ValueError("Ledger checkpoint hash mismatch")
                checks.append({"label":label,"ledger_hash_matched":bool(matching)})
            replay=historical_replay(cell)
            replay_check=verify_historical_replay(replay,reference)
            rows=retrospective_rows(cell["fields"],cell["split"]);z=cell["base"]["z"]
            if cfg.get("refit_raw_pca",False):
                raw=zf.read(entry["raw_cache_member"])
                with np.load(io.BytesIO(raw),allow_pickle=False) as cache:
                    keys=[k for k in cache.files if np.asarray(cache[k]).ndim==2 and np.asarray(cache[k]).shape[0]==len(rows)]
                    if len(keys)!=1:raise ValueError("Ambiguous raw hidden-state array")
                    h=cache[keys[0]]
                center,basis,pca=fit_train_pca(h,np.asarray([r["split"]=="train" for r in rows]),cfg.get("q",16));z=(h-center)@basis
            else:pca={"source":"saved_training_PCA","q":z.shape[1]}
            controls=dict(config.get("structural",{}));controls.update(cfg.get("controls",{}))
            controls["secondary"]=list(dict.fromkeys([*controls.get("secondary",["affine","procrustes","residual","generator"]),"local"]))
            derived=analyze_cell(z,rows,get_group(cell["fields"]["group"]),controls)
            result={"fields":cell["fields"],"members":cell["members"],"ledger_checks":checks,
                    "historical_replay":replay,"historical_replay_check":replay_check,"retrospective_controls":derived,"coordinates":pca,
                    "inference_status":"exploratory_reused_old_data; paired_templates_not_new_families",
                    "source_unchanged":True}
            atomic_json(target,result);summaries.append(result)
    summary={"mode":mode,"cells":len(summaries),"expected_pilot_cells":16,"expected_full_cells":240,
             "archive_read_only":True,"historical_replay_separate_from_refits":True,
             "formal_confirmatory_decisions":False,"cell_outputs":[identity(e)[:20] for e in chosen]}
    atomic_json(Path(output)/"summary.json",summary);return summary
