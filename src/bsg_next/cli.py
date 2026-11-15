"""Command-line entry point for the prospective RunPod study."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

from . import __version__
from .config import load_config,smoke_config
from .io import jsonable,atomic_json,source_identity,file_sha256

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_CONFIG=ROOT/"configs/prospective/bf16_revision.json"


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--version",action="version",version="bsg-next " + __version__)
    sub=p.add_subparsers(dest="command",required=True)
    for name in ("validate","plan","preflight","datasets","synthetic","offline","extract","structural",
                 "power","native","peers","crossmodel","extensions","model-smoke","diagnose","run","report","export-paper","collect","pin-model"):
        q=sub.add_parser(name)
        q.add_argument("--config",type=Path,default=DEFAULT_CONFIG)
        q.add_argument("--set",dest="overrides",action="append",default=[],help="dotted.key=JSON_value")
        q.add_argument("--output",type=Path)
        q.add_argument("--technical",action="store_true",help="technical tests cannot be promoted to the paper")
        if name in {"extract","native","peers","extensions","model-smoke","run"}:
            q.add_argument("--execute-models",action="store_true",help="explicitly allow loading/downloading model weights")
        if name=="preflight":
            q.add_argument("--require-cuda",action="store_true");q.add_argument("--check-access",action="store_true")
        if name=="synthetic":
            q.add_argument("--mode",choices=["timing","positive","calibration","negative","crossed","bootstrap","crossed_bootstrap","power"],default="timing")
            q.add_argument("--limit",type=int);q.add_argument("--worker-index",type=int,default=0);q.add_argument("--worker-count",type=int,default=1)
        if name=="offline":
            q.add_argument("--archive",type=Path,required=True);q.add_argument("--index",type=Path)
            q.add_argument("--mode",choices=["pilot","full"],default="pilot");q.add_argument("--limit",type=int)
        if name=="diagnose":
            q.add_argument("--replicates",type=int,default=12)
            q.add_argument("--max-seconds",type=float,default=180.)
        if name=="extract":
            q.add_argument("--datasets",type=Path,required=True)
            q.add_argument("--phase",choices=["development","confirmation"],default="development")
            q.add_argument("--freeze",type=Path);q.add_argument("--development-cache",type=Path)
        if name=="structural":
            q.add_argument("--cache-root",type=Path);q.add_argument("--mode",choices=["all","primary","pilot","actions"],default="all")
        if name=="power":
            q.add_argument("--pilot-contrasts",type=Path);q.add_argument("--s3-pilot",type=Path);q.add_argument("--limit",type=int)
        if name=="native":q.add_argument("--mode",choices=["prepare","pilot","confirm","full"],default="prepare")
        if name=="peers":q.add_argument("--mode",choices=["anchor","factorial"],default="anchor")
        if name=="crossmodel":q.add_argument("--cache-root",type=Path);q.add_argument("--extract",action="store_true");q.add_argument("--execute-models",action="store_true")
        if name=="extensions":q.add_argument("--cache-root",type=Path)
        if name=="run":
            q.add_argument("--phase",choices=["cpu","retrospective","pilot","confirm","full","peers","crossmodel","extensions"],default="full")
            q.add_argument("--archive",type=Path);q.add_argument("--index",type=Path);q.add_argument("--limit",type=int)
            q.add_argument("--include-retrospective",action="store_true",help="explicitly rerun expensive historical diagnostics")
        if name in {"report","export-paper","collect"}:q.add_argument("--run",type=Path,required=True)
        if name=="export-paper":q.add_argument("--project",type=Path,default=ROOT);q.add_argument("--allow-partial",action="store_true")
        if name=="collect":q.add_argument("--include-caches",action="store_true")
        if name=="pin-model":
            q.add_argument("--model-id",required=True);q.add_argument("--revision",default="main");q.add_argument("--lineage",required=True)
            q.add_argument("--native-only",action="store_true")
    return p


def _models_allowed(args):
    if not getattr(args,"execute_models",False):
        raise ValueError("this command loads model weights; pass --execute-models when intentionally running research inference")


def _output(args):
    if args.output is None:raise ValueError("--output requires a fresh or identity-matching run directory")
    return args.output


def main(argv=None):
    args=parser().parse_args(argv)
    try:
        config=load_config(args.config,args.overrides)
        if args.technical:config=smoke_config(config)
        config["runtime_source_sha256"]=source_identity(Path(__file__).parent)
        from importlib.metadata import version,PackageNotFoundError
        dependencies={}
        for package in ("numpy","scipy","torch","transformers","tokenizers","huggingface-hub"):
            try:dependencies[package]=version(package).split("+",1)[0]
            except PackageNotFoundError:dependencies[package]=None
        config["protocol_dependency_versions"]=dependencies
        assets={}
        references={"peer_anchors":config.get("peers",{}).get("original_model_config"),
                    "word_bank":config.get("crossmodel",{}).get("word_file"),
                    "historical_cell_index":config.get("offline",{}).get("index")}
        references.update({"external_bank_"+str(k):v for k,v in config.get("data",{}).get("external_manifests",{}).items()})
        for name,reference in references.items():
            if not reference:continue
            asset=Path(reference)
            if not asset.is_file() and not asset.is_absolute():asset=ROOT/asset
            if not asset.is_file():raise ValueError("missing scientific input asset: "+name)
            assets[name]=file_sha256(asset)
        config["protocol_asset_sha256"]=assets
        if args.command=="validate":
            result={"valid":True,"study_id":config["study_id"],"models":len(config["models"]),"model_weights_loaded":False}
        elif args.command=="plan":
            from .synthetic import synthetic_plan
            result={"study_id":config["study_id"],"technical_test":config.get("technical_test",False),
                "synthetic":synthetic_plan(),"primary_structural_prompts_default":344448,
                "primary_float32_bytes_default":15049162752,"native_test_recipients_default":3072,
                "native_legacy_condition_evaluations":69120,
                "peer_and_secondary_conditions_require_actual_ledger":True,
                "sequence":["seal maximum prompt/orbit pool","CPU calibration and retrospective diagnostics",
                    "development extraction and excluded pilot","power and configuration freeze",
                    "selected confirmation extraction","structural/native/peer comparisons","cross-model and conditional extensions","verified result export"],
                "models_loaded":False,"GPU_hours":"measure the pilot; not inferred from row counts"}
        elif args.command=="preflight":
            from .preflight import preflight
            result=preflight(config,require_cuda=args.require_cuda,check_access=args.check_access,output=args.output)
            if args.output:atomic_json(args.output,result)
            print(json.dumps(jsonable(result),indent=2));return 0 if result["ok"] else 2
        elif args.command=="datasets":
            from .pipeline import seal_study
            result=seal_study(config,_output(args))
        elif args.command=="model-smoke":
            from .smoke import run_model_smoke
            _models_allowed(args)
            result=run_model_smoke(config,_output(args))
        elif args.command=="synthetic":
            from .synthetic import run_synthetic
            if args.limit is not None and not args.technical:raise ValueError("a truncated synthetic run must be marked --technical")
            result=run_synthetic(config,_output(args),mode=args.mode,limit=args.limit,worker_index=args.worker_index,worker_count=args.worker_count)
        elif args.command=="diagnose":
            from .diagnostics import diagnose_context
            result=diagnose_context(config,_output(args),replicates=args.replicates,max_seconds=args.max_seconds)
        elif args.command=="offline":
            from .offline import run_offline
            if args.limit is not None and not args.technical:raise ValueError("a truncated archive check must be marked --technical")
            result=run_offline(config,_output(args),archive=args.archive,index=args.index,mode=args.mode,limit=args.limit)
        elif args.command=="extract":
            from .pipeline import extract_study
            _models_allowed(args)
            f=32;o=16
            if args.phase=="confirmation":
                if not args.freeze:raise ValueError("confirmation extraction requires --freeze from power-selected protocol")
                frozen=json.loads(args.freeze.read_text());f=int(frozen["test_families"]);o=int(frozen.get("test_orbits",16))
            result=extract_study(config,_output(args),args.datasets,phase=args.phase,test_families=f,test_orbits=o,development_cache=args.development_cache,confirmation_freeze=args.freeze)
        elif args.command=="structural":
            from .structural import run_structural
            result=run_structural(config,_output(args),cache_root=args.cache_root,mode=args.mode)
        elif args.command=="power":
            from .statistics import run_power,run_s3_power
            if args.limit is not None and not args.technical:raise ValueError("a truncated power run must be marked --technical")
            result=run_s3_power(config,_output(args),pilot_crossed=str(args.s3_pilot),limit=args.limit) if args.s3_pilot else run_power(config,_output(args),pilot_contrasts=str(args.pilot_contrasts) if args.pilot_contrasts else None,limit=args.limit)
        elif args.command=="native":
            from .native import run_native
            _models_allowed(args);result=run_native(config,_output(args),mode=args.mode)
        elif args.command=="peers":
            from .peers import run_reproduction_anchor,run_factorial_stage
            _models_allowed(args)
            result=(run_reproduction_anchor if args.mode=="anchor" else run_factorial_stage)(config,_output(args))
        elif args.command=="crossmodel":
            if args.extract:
                from .wordfields import collect_word_fields
                _models_allowed(args);result=collect_word_fields(config,_output(args))
            elif args.cache_root:
                from .wordfields import analyze_word_fields
                result=analyze_word_fields(config,_output(args),args.cache_root)
            else:
                from .crossmodel import run_crossmodel
                result=run_crossmodel(config,_output(args))
        elif args.command=="extensions":
            from .extensions import run_extensions
            _models_allowed(args)
            config.setdefault("execution",{})["allowed_model_execution"]=True
            result=run_extensions(config,_output(args),cache_root=args.cache_root)
        elif args.command=="run":
            from .pipeline import StudyPipeline
            if args.phase not in {"cpu","retrospective"}:_models_allowed(args)
            if args.limit is not None and not args.technical:raise ValueError("a limited suite must be --technical")
            result=StudyPipeline(config,_output(args)).run(args.phase,archive=args.archive,index=args.index,limit=args.limit,
                                                         include_retrospective=args.include_retrospective)
        elif args.command=="report":
            from .reporting import build_report
            result=build_report(args.run,args.output or args.run/"report")
        elif args.command=="export-paper":
            from .reporting import export_paper
            result=export_paper(args.run,args.project,allow_partial=args.allow_partial)
        elif args.command=="collect":
            from .reporting import collect_results
            result=collect_results(args.run,_output(args),include_caches=args.include_caches)
        elif args.command=="pin-model":
            from huggingface_hub import HfApi,hf_hub_download
            info=HfApi().model_info(args.model_id,revision=args.revision)
            architecture=json.loads(Path(hf_hub_download(args.model_id,"config.json",revision=info.sha)).read_text())
            layers=int(architecture["num_hidden_layers"])
            model={"id":args.model_id,"revision":info.sha,"tokenizer_revision":info.sha,"lineage":args.lineage,
                "structural":not args.native_only,"native":args.native_only,
                "layers":[max(1,round(layers*d)) for d in (.2,.4,.6,.8)],
                "hidden_dimension":int(architecture["hidden_size"]),
                "metadata_source":"pinned Hugging Face config.json; no weight file downloaded"}
            atomic_json(_output(args),model);result={"metadata_only":True,"model":model,"weights_downloaded":False}
        else:raise ValueError("unknown command")
        print(json.dumps(jsonable(result),indent=2,sort_keys=True))
        return 0
    except (ValueError,RuntimeError,FileNotFoundError,KeyError) as exc:
        print(f"bsg-next: {type(exc).__name__}: {exc}",file=sys.stderr)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
