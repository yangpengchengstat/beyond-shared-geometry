"""Complete-result reports, private collection, and provenance-bound TeX slots."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import re
import zipfile

from .io import atomic_json,read_json,file_sha256,identity

SECRET_KEYS={"hf_token","access_token","refresh_token","authorization","api_key","password","passwd","secret","credentials"}
SECRET_PATTERN=re.compile(r"(?:hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{20,})")
STAGE_GROUPS=(
    ("Calibration and error control",("synthetic_calibration","synthetic_negative","synthetic_crossed","synthetic_bootstrap","synthetic_crossed_bootstrap","calibration_gate")),
    ("Historical refit diagnostics",("offline_pilot","offline_full")),
    ("Representation sensitivity",("embedding_floor","structural_confirmation","grammar_sensitivity")),
    ("Operation/pathway/sham tests",("native_confirmation",)),
    ("Peer anchors/common tasks",("peer_anchors","native_confirmation")),
    ("Factorial ANOVA/relation edits",("peer_factorial",)),
    ("Model/word/action transport",("crossmodel_transport","crossmodel_structural")),
    ("Product/interface/lineage",("extensions",)),
)


def _load(path):
    return read_json(path) if path.is_file() else None


def _public(value):
    """Retain scientific content while removing host-specific path identities."""
    if isinstance(value,dict):return {k:("[redacted]" if str(k).lower() in SECRET_KEYS else _public(v)) for k,v in value.items()}
    if isinstance(value,list):return [_public(v) for v in value]
    if isinstance(value,str):
        value=SECRET_PATTERN.sub("[redacted credential]",value)
        if value.startswith("/"):return "external/"+Path(value).name
        return re.sub(r"(?<![A-Za-z0-9:/])/(?:Users|home|private|workspace|tmp|mnt|root|var)/[^\s,;\"']+","[external path]",value)
    return value


def _run_identity(run,cfg,required=False):
    marker=_load(run/"run_identity.json")
    if marker is None:
        if required:raise ValueError("publication import requires the original run_identity.json")
        return None
    descriptor={k:v for k,v in marker.items() if k!="identity"}
    if marker.get("identity")!=identity(descriptor) or identity(marker.get("config",{}))!=identity(cfg):
        raise ValueError("run configuration/source identity mismatch")
    return marker["identity"]


def _expected_stages(cfg):
    names={"datasets","synthetic_timing","synthetic_calibration","synthetic_negative","synthetic_crossed",
           "synthetic_bootstrap","synthetic_crossed_bootstrap","extract_development","embedding_floor",
           "structural_pilot","pilot_context_eligibility","structural_power","extract_confirmation","structural_confirmation"}
    if _native_enabled(cfg):names.update(("native_prepare","native_confirmation"))
    if cfg.get("data",{}).get("grammar_sensitivity",True):names.add("grammar_sensitivity")
    if cfg.get("peers",{}).get("reproduction_anchors",True):names.add("peer_anchors")
    if cfg.get("peers",{}).get("factorial_anova",True):names.add("peer_factorial")
    if not cfg.get("technical_test"):names.add("calibration_gate")
    if cfg.get("execution",{}).get("require_historical_archive",True):names.update(("offline_pilot","offline_full"))
    if any(str(g).upper().startswith("S3") for g in cfg.get("data",{}).get("groups",[])):names.add("s3_power")
    if cfg.get("crossmodel",{}).get("enabled",False):names.update(("crossmodel_extraction","crossmodel_transport","crossmodel_structural"))
    return sorted(names)


def _native_enabled(cfg):
    native=cfg.get("native",{})
    return bool(native.get("enabled",True) and native.get("models",[]))


def _group_stages(label,names,cfg,states):
    selected=[]
    for name in names:
        if name=="grammar_sensitivity" and not cfg.get("data",{}).get("grammar_sensitivity",True):continue
        if name.startswith("native_") and not _native_enabled(cfg):continue
        if name=="peer_anchors" and not cfg.get("peers",{}).get("reproduction_anchors",True):continue
        if name=="peer_factorial" and not cfg.get("peers",{}).get("factorial_anova",True):continue
        if name.startswith("crossmodel_") and not cfg.get("crossmodel",{}).get("enabled",False):continue
        if name.startswith("offline_") and not cfg.get("execution",{}).get("require_historical_archive",True) and name not in states:continue
        if name=="extensions" and not any(cfg.get("extensions",{}).get(k,False) for k in ("product_actions","interface_training","third_lineage","external_language_banks")):continue
        if label=="Operation/pathway/sham tests" and not any(cfg.get("native",{}).get(k,False) for k in ("include_operations","include_joint_variables","include_pathway","include_shams","include_rebinding")):continue
        selected.append(name)
    return selected


def _execution_status(names,states):
    if not names:return "N/A (disabled)"
    statuses=[states.get(name,{}).get("status","pending") for name in names]
    if "failed" in statuses:return "Execution failed"
    if "gate_stopped" in statuses:return "Gate stopped"
    if all(s=="complete" for s in statuses):return "Execution complete"
    if "running" in statuses:return "Running"
    if any(s=="complete" for s in statuses):return "Partial"
    return "Pending"


def build_report(run,output):
    run,output=Path(run),Path(output)
    cfg=read_json(run/"study_config.json")
    verified=_run_identity(run,cfg)
    states={p.stem:read_json(p) for p in sorted((run/"stage_state").glob("*.json"))}
    structural=_load(run/"outputs/structural_confirmation/summary.json")
    native=_load(run/"outputs/native_study/native_results.json")
    crossmodel=_load(run/"outputs/crossmodel_transport/crossmodel.json")
    expected=_expected_stages(cfg)
    technical=bool(cfg.get("technical_test",False) or (native or {}).get("technical_run",False)
                   or (crossmodel or {}).get("technical_run",False) or (structural or {}).get("technical_run",False))
    result={"study_id":cfg["study_id"],"technical_test":technical,
            "configuration_sha256":identity(cfg),"stages":states,
            "structural":structural,"native":native,"crossmodel":crossmodel,
            "unexecuted_results_are_not_zero":True,
            "run_identity_verified":verified is not None,"expected_core_stages":expected,
            "all_stages_complete":all(states.get(name,{}).get("status")=="complete" for name in expected),
            "execution_status_does_not_mean_scientific_success":True,
            "all_recorded_stages_complete":bool(states) and all(v.get("status")=="complete" for v in states.values())}
    result=_public(result)
    output.mkdir(parents=True,exist_ok=True)
    atomic_json(output/"study_report.json",result)
    rows=[]
    if structural:
        for key,value in (structural.get("decisions") or {}).get("contrasts",{}).items():
            rows.append({"study":"structural","condition":key,"mean":value.get("mean"),
                "ci_lower":value.get("ci95",[None,None])[0],"ci_upper":value.get("ci95",[None,None])[1],
                "status":value.get("status"),"units":value.get("n"),"unit_type":"scenario_families",
                "interval_level":.95,"interval_method":"ordinary_family_t_descriptive_not_Holm_decision"})
        for key,value in ((structural.get("S3") or {}).get("decisions") or {}).get("contrasts",{}).items():
            rows.append({"study":"S3_joint_transfer","condition":key,"mean":value.get("mean"),
                "ci_lower":value.get("ci95",[None,None])[0],"ci_upper":value.get("ci95",[None,None])[1],
                "status":value.get("status"),"units":value.get("n"),"unit_type":"crossed_family_orbit_cells_not_independent_rows",
                "interval_level":.95,"interval_method":"ordinary_crossed_t_descriptive_not_Holm_decision"})
    if native:
        for key,value in native.get("decisions",{}).get("cells",native.get("decisions",{}).get("contrasts",{})).items():
            rows.append({"study":"native","condition":key,"mean":value.get("mean",value.get("effect")),
                "ci_lower":(value.get("interval") or value.get("ci95") or [None,None])[0],"ci_upper":(value.get("interval") or value.get("ci95") or [None,None])[1],
                "status":value.get("status"),"units":value.get("families",value.get("n")),"unit_type":"scenario_families",
                "interval_level":value.get("interval_level",.95),"interval_method":"Bonferroni_family_t"})
    with (output/"paired_effects.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=["study","condition","mean","ci_lower","ci_upper","interval_level","interval_method","status","units","unit_type"])
        writer.writeheader();writer.writerows(rows)
    lines=["# Prospective study report","",f"Study: {cfg['study_id']}","",
        "Technical test; not eligible for manuscript results." if result["technical_test"] else "Production configuration; execution completeness and inference are reported separately.","",
        "| Stage | Status |","|---|---|"]
    lines += [f"| {name} | {entry.get('status','unknown')} |" for name,entry in states.items()]
    lines += ["","Unexecuted results remain pending. Negative, infeasible and inconclusive outcomes are retained.",""]
    (output/"README.md").write_text("\n".join(lines),encoding="utf-8")
    # This diagnostic plot uses real paired effects only and is never populated from a simulation fixture.
    if rows and not result["technical_test"]:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            valid=[r for r in rows if r["mean"] is not None]
            if valid:
                fig,ax=plt.subplots(figsize=(8,max(3,.35*len(valid))))
                for i,row in enumerate(valid):
                    if row["ci_lower"] is not None and row["ci_upper"] is not None:
                        ax.plot([row["ci_lower"],row["ci_upper"]],[i,i],color="0.35")
                    ax.scatter([row["mean"]],[i],color="#24648c",s=20)
                ax.axvline(0,color="0.6",lw=.7);ax.set_yticks(range(len(valid)),[r["condition"] for r in valid])
                ax.set_xlabel("Paired improvement; task-specific scale (see protocol)")
                fig.tight_layout();fig.savefig(output/"paired_effects.pdf");plt.close(fig)
        except ImportError:pass
    return {"report":"study_report.json","paired_effects":"paired_effects.csv",
            "technical_test":result["technical_test"],"all_stages_complete":result["all_stages_complete"]}


def _format(value):
    return "Pending" if value is None else f"{float(value):.4f}"


def _entry_format(value):
    if not value:return "Pending"
    return "Not estimable" if value.get("mean") is None else _format(value["mean"])


def export_paper(run,project,allow_partial=False):
    run,project=Path(run),Path(project)
    cfg=read_json(run/"study_config.json")
    from .config import validate_config
    validate_config(cfg)
    if cfg.get("technical_test") or cfg.get("native",{}).get("technical_run"):
        raise ValueError("technical/smoke outputs cannot populate the manuscript")
    run_identity=_run_identity(run,cfg,required=True)
    required=["structural_confirmation"]+(["native_confirmation"] if _native_enabled(cfg) else [])
    states={p.stem:read_json(p) for p in (run/"stage_state").glob("*.json")}
    if any(v.get("identity")!=run_identity for v in states.values()):raise ValueError("stage identity does not match the source run")
    missing=[name for name in required if states.get(name,{}).get("status")!="complete"]
    if missing and not allow_partial:
        raise ValueError("publication import needs complete primary stages: "+", ".join(missing))
    if not (run/"confirmation_freeze.json").is_file() and not allow_partial:
        raise ValueError("no sealed confirmation decision found")
    frozen=_load(run/"confirmation_freeze.json")
    if frozen and frozen.get("configuration") not in (None,run_identity):raise ValueError("confirmation freeze does not match the source run")
    structural=_load(run/"outputs/structural_confirmation/summary.json") if "structural_confirmation" not in missing else None
    native=_load(run/"outputs/native_study/native_results.json") if _native_enabled(cfg) and "native_confirmation" not in missing else None
    if native and native.get("technical_run"):
        raise ValueError("native runtime marks this result technical")
    if structural is None and "structural_confirmation" not in missing:missing.append("structural_results")
    if _native_enabled(cfg) and native is None and "native_confirmation" not in missing:missing.append("native_results")
    if missing and not allow_partial:raise ValueError("publication import needs complete primary results: "+", ".join(missing))
    evidence=project/"results/derived"/(cfg["study_id"]+"-prospective")
    if evidence.exists():
        old=_load(evidence/"provenance.json")
        if old and old.get("configuration_sha256")!=identity(cfg):
            raise ValueError("a different result already occupies this publication evidence namespace")
    evidence.mkdir(parents=True,exist_ok=True)
    build_report(run,evidence)
    provenance={"source_run":cfg["study_id"],"configuration_sha256":identity(cfg),
        "runtime_source_sha256":cfg.get("runtime_source_sha256"),
        "confirmation_freeze_sha256":file_sha256(run/"confirmation_freeze.json") if frozen else None,
        "structural_summary_sha256":file_sha256(run/"outputs/structural_confirmation/summary.json") if structural else None,
        "native_summary_sha256":file_sha256(run/"outputs/native_study/native_results.json") if native else None,
        "partial":bool(missing),"missing_primary_stages":missing}
    atomic_json(evidence/"provenance.json",provenance)
    # Keep numeric exports explicit: no synthetic or retrospective metric can fill a prospective row.
    contrasts=((structural or {}).get("decisions") or {}).get("contrasts",{})
    s3=(((structural or {}).get("S3") or {}).get("decisions") or {}).get("contrasts",{})
    table=["% Provenance-bound production results; unexecuted conditions stay Pending.",
        r"\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{3pt}",
        r"\caption{Prospective paired improvements. Native contrasts use length-six-to-eight continuations; intervals are multiplicity-adjusted. Unexecuted conditions remain Pending. All inferential decisions and failures accompany the source report.}",
        r"\label{tab:prospective}",r"\begin{tabular}{lcc}\toprule",
        r"Structural task & Prediction gain & Context gain \\\midrule"]
    for label,key in [(r"$C_7$","C7"),(r"$D_4$ corners","D4_corner")]:
        table.append(label+" & "+_entry_format(contrasts.get(key+".pred",{}))+" & "+_entry_format(contrasts.get(key+".ctx",{}))+r" \\")
    def endpoint(ep):
        return next((v for k,v in s3.items() if k==ep or k.endswith("."+ep)),{})
    table.append(r"$S_3$ joint new & "+_entry_format(endpoint("pred"))+" & "+_entry_format(endpoint("ctx"))+r" \\")
    table += [r"\midrule Native task & Accuracy gain & Adjusted CI \\"]
    ncontrasts=((native or {}).get("decisions") or {}).get("contrasts",{})
    for mi,name in enumerate(("Llama", "Qwen")):
        for domain,label in (("C7",r"$C_7$"),("D4_free8",r"$D_4$ free")):
            value=ncontrasts.get(f"m{mi}.{domain}",{})
            bounds=value.get("interval")
            interval=("N/A" if not _native_enabled(cfg) else "Pending" if not value else "Not estimable" if not bounds or any(v is None for v in bounds)
                      else f"[{_format(bounds[0])}, {_format(bounds[1])}]")
            estimate=_entry_format(value) if _native_enabled(cfg) else "N/A"
            table.append(name+" "+label+" & "+estimate+" & "+interval+r" \\")
    table += [r"\bottomrule\end{tabular}\end{table}",
              "% Full native and peer results, intervals, stage status and inference are in the bound evidence report."]
    tex=project/"paper/generated/prospective_results.tex"
    tex.parent.mkdir(parents=True,exist_ok=True)
    tex.write_text("\n".join(table)+"\n",encoding="utf-8")
    status_tex=project/"paper/generated/prospective_status.tex"
    status=[r"% Execution states bound to the source-run provenance; not hypothesis outcomes.",
            r"\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{3pt}",
            r"\caption{Prospective execution status. Execution completion does not imply a positive scientific finding; negative, infeasible and inconclusive outcomes remain in the bound report.}\label{tab:prospective-stages}",
            r"\begin{tabular}{ll}\toprule Study & Execution \\\midrule"]
    status += [label+" & "+_execution_status(_group_stages(label,names,cfg,states),states)+r" \\" for label,names in STAGE_GROUPS]
    status += [r"\bottomrule\end{tabular}\end{table}"]
    status_tex.write_text("\n".join(status)+"\n",encoding="utf-8")
    manifest_path=project/"release/artifact_manifest.json"
    manifest=read_json(manifest_path)
    new=[p for p in evidence.rglob("*") if p.is_file()]+[tex,status_tex]
    paths={str(p.relative_to(project)) for p in new}
    manifest["artifacts"]=[e for e in manifest["artifacts"] if e["path"] not in paths]
    for path in new:
        manifest["artifacts"].append({"path":str(path.relative_to(project)),"sha256":file_sha256(path),
            "source_run":cfg["study_id"],"source_path":"prospective_result_export/"+path.name,
            "role":"prospective production result; includes negative and uninformative outcomes"})
    atomic_json(manifest_path,manifest)
    qa=project/"release/pdf_qa.json"
    if qa.exists():qa.unlink()  # a source edit explicitly invalidates the previous PDF binding
    return {"evidence":str(evidence.relative_to(project)),"tex":str(tex.relative_to(project)),"status_tex":str(status_tex.relative_to(project)),
            "requires_paper_rebuild":True,"partial":bool(missing)}


def collect_results(run,output,include_caches=False):
    import sys
    run,output=Path(run).resolve(),Path(output).resolve()
    if not (run/"study_config.json").is_file():raise ValueError("not a prospective run directory")
    if output.exists():raise ValueError("result bundle already exists; choose a new output")
    if output.is_relative_to(run):raise ValueError("result archive must be outside the run tree")
    forbidden={"hf_cache","hf",".cache",".git",".venv","__pycache__","secrets","credentials"}
    partial=output.with_suffix(output.suffix+".partial")
    if partial.exists():raise ValueError("unfinished collection exists; preserve it and choose a fresh output name")
    files=[]
    print("COLLECT scanning result files",file=sys.stderr,flush=True)
    for scanned,path in enumerate(sorted(run.rglob("*")),1):
        if scanned%10000==0:print(f"COLLECT scanned={scanned} selected={len(files)}",file=sys.stderr,flush=True)
        rel=path.relative_to(run)
        if any(part in forbidden or part.startswith(".") for part in rel.parts) or path.name.endswith(".partial"):continue
        if re.search(r"(?:^|[_.-])(?:secrets?|credentials?|tokens?|authorization|password|passwd|private_key)(?:[_.-]|$)",path.name,re.I):continue
        if path.is_symlink():raise ValueError("result symlink cannot be collected")
        if not path.is_file():continue
        if path.suffix in {".safetensors",".bin",".pt",".pth",".gguf",".pyc",".pem",".key"}:continue
        if not include_caches and path.suffix in {".npy",".npz"}:continue
        if path.suffix.lower() in {".json",".jsonl",".yaml",".yml",".txt",".log",".csv"}:
            # Refuse rather than silently distributing an accidental credential
            # in an otherwise required scientific record. No environment is read.
            if SECRET_PATTERN.search(path.read_text(encoding="utf-8",errors="ignore")):
                raise ValueError("secret-like credential content in result file: "+str(rel))
        files.append(path)
    records=[]
    for index,p in enumerate(files,1):
        records.append({"path":str(p.relative_to(run)),"bytes":p.stat().st_size,"sha256":file_sha256(p)})
        if index%10000==0:print(f"COLLECT checksummed={index}/{len(files)}",file=sys.stderr,flush=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(partial,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for index,path in enumerate(files,1):
            archive.write(path,"run/"+str(path.relative_to(run)))
            if index%10000==0:print(f"COLLECT archived={index}/{len(files)}",file=sys.stderr,flush=True)
        archive.writestr("COLLECTION_MANIFEST.json",json.dumps({"files":records,"include_caches":include_caches,
            "model_weights_included":False,"may_be_incomplete_run":True},indent=2))
    partial.replace(output)
    digest=file_sha256(output)
    output.with_suffix(output.suffix+".sha256").write_text(digest+"  "+output.name+"\n")
    print("COLLECT complete; archive and checksum are ready",file=sys.stderr,flush=True)
    return {"archive":output.name,"sha256":digest,"files":len(files),"bytes":output.stat().st_size,
            "cached_activations_included":include_caches}
