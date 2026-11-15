"""Ordered, resumable experiment stages with a sealed-test boundary."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import time
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

from .io import atomic_json, read_json, identity, prepare_run, file_sha256
from .config import model_key


class ScientificGateError(RuntimeError):
    pass


def _synthetic_worker(config, output, mode, worker_index, worker_count):
    """Spawn-safe entry point; the seed namespace never depends on shard count."""
    from .synthetic import run_synthetic
    return run_synthetic(config, output, mode=mode, worker_index=worker_index,
                         worker_count=worker_count)


def run_synthetic_workers(config, output, mode, limit=None):
    """Execute disjoint atomic seed shards, then read one aggregate summary.

    A technical ``limit`` always stays serial so that it is a total budget, not
    a budget multiplied by the worker count. Initial/final limit=0 calls never
    fit a seed, and the parent creates the immutable run marker before spawn.
    """
    from .synthetic import run_synthetic
    workers = int(config.get("execution", {}).get("cpu_workers", 1))
    if workers < 1:
        raise ValueError("execution.cpu_workers must be a positive integer")
    if workers == 1 or limit is not None:
        return run_synthetic(config, output, mode=mode, limit=limit)
    run_synthetic(config, output, mode=mode, limit=0, worker_index=0, worker_count=1)
    # Spawned workers inherit single-threaded BLAS to avoid workers x BLAS
    # threads oversubscription. Restore the caller's environment afterward.
    names = ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            os.environ[name] = "1"
        with ProcessPoolExecutor(max_workers=workers,
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(_synthetic_worker, config, str(output), mode, index, workers)
                       for index in range(workers)]
            for future in futures:
                future.result()
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    result = run_synthetic(config, output, mode=mode, limit=0, worker_index=0, worker_count=1)
    result["execution_workers"] = workers
    atomic_json(Path(output)/"aggregate_summary.json", result)
    return result


def seal_study(config, output):
    from .data import build_study_records, build_support_records, seal_dataset, import_dataset,load_sealed_dataset
    output = Path(output)
    prepare_run(output,config,"sealed_maximum_prompt_pool")
    paths = {}
    cfg = config.get("data",{})
    for domain in cfg.get("groups",["C7","D4","S3"]):
        leaf = output/domain
        if domain in cfg.get("external_manifests",{}):
            import_dataset(cfg["external_manifests"][domain],leaf,
                           metadata={"scope":"externally authored, family/grammar identities supplied"})
        else:
            rows = build_study_records(domain,seed=config["seed"],
                max_test_families=cfg.get("max_test_families",64),max_test_orbits=cfg.get("max_test_orbits",24),include_pilot=True)
            if domain.upper().startswith("S3") and config.get("structural",{}).get("equal_support",False):
                rows += build_support_records(domain,seed=config["seed"],test_orbits=cfg.get("max_test_orbits",24))
            seal_dataset(rows,leaf,metadata={"sampling_scope":cfg.get("sampling_scope"),
                "maximum_candidate_pool":True,"test_outcomes_accessed":False})
        paths[domain] = str(leaf)
        if not config.get("technical_test",False):
            sealed=load_sealed_dataset(leaf)
            candidate_families={r["family_id"] for r in sealed if r.get("context_split",r.get("split"))=="test"}
            if len(candidate_families)<int(cfg.get("max_test_families",64)):
                raise ValueError(f"{domain}: sealed bank lacks the maximum candidate test-family pool")
            if domain.upper().startswith("S3"):
                candidate_orbits={r["orbit_id"] for r in sealed if r.get("entity_split")=="test"}
                if len(candidate_orbits)<int(cfg.get("max_test_orbits",24)):
                    raise ValueError("S3 sealed bank lacks the maximum candidate held-out orbit pool")
    result = {"datasets":paths,"seal_before_pilot_outcomes":True}
    atomic_json(output/"datasets.json",result)
    return result


def extract_study(config, output, dataset_root, *, phase="development",test_families=32,test_orbits=16,
                  development_cache=None,runtime_factory=None):
    from .data import load_sealed_dataset, select_phase_records
    from .hf import runtime_from_config
    from .extraction import extract_features,merge_feature_caches
    output = Path(output)
    descriptor = {**config,"extraction_phase":phase,"selected_test_families":test_families,"selected_test_orbits":test_orbits}
    data_paths = read_json(Path(dataset_root)/"datasets.json")["datasets"]
    prepare_run(output,descriptor,"structural_extraction",
                provenance={k:file_sha256(Path(v)/"manifest.json") for k,v in data_paths.items()})
    caches = []
    for model in config["models"]:
        if model.get("structural",True) is False:continue
        runtime = runtime_factory(model,config.get("runtime",{})) if runtime_factory else runtime_from_config(model,config.get("runtime",{}))
        try:
            for domain,path in data_paths.items():
                rows = load_sealed_dataset(path)
                selected = select_phase_records(rows,phase,test_families,test_orbits)
                if not selected:continue
                if phase=="confirmation" and not config.get("technical_test",False):
                    families={r["family_id"] for r in selected if r.get("context_split",r.get("split"))=="test"}
                    if len(families)!=int(test_families):raise ValueError("Selected confirmation family count disagrees with power freeze")
                    if domain.upper().startswith("S3"):
                        orbits={r["orbit_id"] for r in selected if r.get("entity_split")=="test"}
                        if len(orbits)!=int(test_orbits):raise ValueError("Selected confirmation orbit count disagrees with power freeze")
                readout = "final_token" if domain.upper().startswith("S3") else "source_span"
                leaf = output/model_key(model)/domain
                extract_features(runtime,selected,model["layers"],readout,leaf,
                                 potential_features=domain.upper().startswith("S3"))
                caches.append(str(leaf))
        finally:
            if hasattr(runtime,"close"):runtime.close()
    if phase=="confirmation":
        if not development_cache:raise ValueError("confirmation must merge its sealed development cache")
        merged = []
        for model in config["models"]:
            if model.get("structural",True) is False:continue
            for domain in data_paths:
                old = Path(development_cache)/model_key(model)/domain
                new = output/model_key(model)/domain
                leaf = output/"merged"/model_key(model)/domain
                merge_feature_caches([old,new],leaf)
                merged.append(str(leaf))
        caches = merged
    result = {"phase":phase,"cache_dirs":caches,"test_families":test_families,"test_orbits":test_orbits}
    atomic_json(output/"caches.json",result)
    return result


def calibration_gate(config,calibration_output,negative_output,crossed_output,output=None):
    """Seed-level completeness, positive sensitivity and true-null calibration."""
    from .gates import validate_calibration,CalibrationValidationError
    try:result=validate_calibration(config,calibration_output,negative_output,crossed_output)
    except (CalibrationValidationError,FileNotFoundError,KeyError) as exc:
        raise ScientificGateError(f"calibration failed/incomplete: {exc}; preserve outputs and revise before confirmation") from exc
    if output:atomic_json(Path(output)/"calibration_gate.json",result)
    return result


class StudyPipeline:
    def __init__(self,config,output):
        self.config=deepcopy(config)
        self.output=Path(output)
        self.descriptor=prepare_run(self.output,self.config,"prospective_study")
        self.state_dir=self.output/"stage_state"
        self.state_dir.mkdir(exist_ok=True)
        atomic_json(self.output/"study_config.json",self.config)

    def stage(self,name,call,*,resume=True):
        state=self.state_dir/(name+".json")
        if state.exists():
            previous=read_json(state)
            if previous.get("identity")!=self.descriptor["identity"]:
                raise ValueError("stage identity mismatch")
            if previous.get("status")=="complete" and resume:
                print(f"RESUME {name}",flush=True)
                return previous["result"]
        start=time.monotonic()
        print(f"START {name}",flush=True)
        atomic_json(state,{"status":"running","identity":self.descriptor["identity"],"stage":name})
        try:
            result=call(self.output/"outputs"/name)
        except Exception as exc:
            atomic_json(state,{"status":"gate_stopped" if isinstance(exc,ScientificGateError) else "failed",
                "stage":name,"identity":self.descriptor["identity"],"error_type":type(exc).__name__,
                "error":str(exc),"seconds":time.monotonic()-start})
            raise
        record={"status":"complete","stage":name,"identity":self.descriptor["identity"],
                "result":result,"seconds":time.monotonic()-start}
        atomic_json(state,record)
        print(f"DONE {name} ({record['seconds']:.1f}s)",flush=True)
        return result

    def run(self,phase="full",*,archive=None,index=None,limit=None,include_retrospective=False):
        from .synthetic import run_synthetic
        from .offline import run_offline
        from .structural import run_structural
        from .statistics import run_power,run_s3_power
        cfg=self.config
        if include_retrospective and phase not in {"cpu","full"}:
            raise ValueError("Retrospective work belongs to the separate retrospective phase or explicit CPU run")
        if phase not in {"cpu","retrospective","pilot","confirm","full","peers","crossmodel","extensions"}:
            raise ValueError("unknown suite phase")
        datasets=self.stage("datasets",lambda out:seal_study(cfg,out))
        dataset_root=self.output/"outputs/datasets"
        if phase in {"cpu","full"}:
            # Check the four positive cells first, before expensive negative,
            # bootstrap and retrospective work. Keep the final full gate intact.
            self.stage("synthetic_positive",lambda out:run_synthetic_workers(cfg,out,mode="positive",limit=limit))
            if not cfg.get("technical_test",False):
                from .gates import validate_positive_calibration,CalibrationValidationError
                def positive_check(out,directory,sentinel):
                    try:result=validate_positive_calibration(cfg,directory,sentinel=sentinel)
                    except (CalibrationValidationError,FileNotFoundError,KeyError) as exc:
                        raise ScientificGateError(f"positive calibration stopped early: {exc}; preserve outputs") from exc
                    atomic_json(out/"positive_gate.json",result)
                    return result
                self.stage("positive_sentinel_gate",lambda out:positive_check(out,self.output/"outputs/synthetic_positive",True),resume=False)
            for mode in ("timing","calibration","negative","crossed","bootstrap","crossed_bootstrap"):
                self.stage("synthetic_"+mode,lambda out,mode=mode:run_synthetic_workers(cfg,out,mode=mode,limit=limit))
                if mode=="calibration" and not cfg.get("technical_test",False):
                    self.stage("positive_calibration_gate",lambda out:positive_check(out,self.output/"outputs/synthetic_calibration",False),resume=False)
            if not cfg.get("technical_test",False):
                self.stage("calibration_gate",lambda out:calibration_gate(cfg,self.output/"outputs/synthetic_calibration",
                    self.output/"outputs/synthetic_negative",self.output/"outputs/synthetic_crossed",output=out),resume=False)
        if phase in {"full","retrospective"} or include_retrospective:
            if not archive:raise ValueError("Explicit retrospective diagnostics require --archive inputs/full.zip")
            self.stage("offline_pilot",lambda out:run_offline(cfg,out,archive=archive,index=index,mode="pilot",limit=limit))
            self.stage("offline_full",lambda out:run_offline(cfg,out,archive=archive,index=index,mode="full",limit=limit))
        if phase in {"pilot","full"}:
            if not cfg.get("technical_test",False):
                self.stage("calibration_gate",lambda out:calibration_gate(cfg,self.output/"outputs/synthetic_calibration",
                    self.output/"outputs/synthetic_negative",self.output/"outputs/synthetic_crossed",output=out),resume=False)
            development=self.stage("extract_development",lambda out:extract_study(cfg,out,dataset_root,phase="development"))
            local=deepcopy(cfg);local["structural"]["cache_manifests"]=development["cache_dirs"]
            local["structural"]["equal_support"]=False
            self.stage("embedding_floor",lambda out:self._embedding(out,self.output/"outputs/extract_development"))
            pilot=self.stage("structural_pilot",lambda out:run_structural(local,out,mode="pilot"))
            self.stage("pilot_context_eligibility",lambda out:self._pilot_eligibility(pilot,out))
            pilot_path=self.output/"outputs/structural_pilot/pilot_contrasts.npy"
            if not pilot_path.is_file():
                raise ScientificGateError("pilot family contrasts are missing or tasks lack aligned independent family IDs")
            power=self.stage("structural_power",lambda out:run_power(cfg,out,pilot_contrasts=str(pilot_path),limit=limit))
            if self._has_s3():
                s3_pilot=self.output/"outputs/structural_pilot/pilot_crossed_contrasts.npz"
                if not s3_pilot.is_file():raise ScientificGateError("S3 excluded crossed pilot table is missing; no orbit power selection is possible")
                self.stage("s3_power",lambda out:run_s3_power(cfg,out,pilot_crossed=s3_pilot,limit=limit))
            if self._native_enabled():self.stage("native_prepare",lambda out:self._native(out,"prepare"))
        if phase in {"confirm","full"}:
            if not cfg.get("technical_test",False):
                self.stage("calibration_gate",lambda out:calibration_gate(cfg,self.output/"outputs/synthetic_calibration",
                    self.output/"outputs/synthetic_negative",self.output/"outputs/synthetic_crossed",output=out),resume=False)
            pstate=self.state_dir/"structural_power.json"
            if not pstate.is_file():raise ScientificGateError("run pilot and freeze power before collecting test activations")
            power=read_json(pstate)["result"]
            size=power.get("selected_families")
            if size is None:
                if not cfg.get("execution",{}).get("allow_confirmatory_underpowered",False):
                    raise ScientificGateError("pilot power is insufficient at64families; register a larger design or explicitly label a limited-precision study")
                size=64
            orbit_size=int(cfg.get("data",{}).get("entity_split",[24,8,16])[2]);s3_power_hash=None
            if self._has_s3():
                s3state=self.state_dir/"s3_power.json"
                if not s3state.is_file():raise ScientificGateError("run S3 excluded-pilot power before collecting held-out entity outcomes")
                s3power=read_json(s3state)["result"]
                sf,so=s3power.get("selected_families"),s3power.get("selected_orbits")
                if sf is None or so is None:
                    if not cfg.get("execution",{}).get("allow_confirmatory_underpowered",False):
                        raise ScientificGateError("S3 power is insufficient within the sealed F/O grid; no test outcomes collected")
                    sf,so=64,24
                size=max(int(size),int(sf));orbit_size=int(so)
                s3_power_hash=file_sha256(self.output/"outputs/s3_power/s3_power.json")
            if size>cfg.get("data",{}).get("max_test_families",64) or orbit_size>cfg.get("data",{}).get("max_test_orbits",24):
                raise ScientificGateError("Selected power design exceeds the pool sealed before pilot")
            if not cfg.get("technical_test",False) and int(size)<int(cfg.get("synthetic",{}).get("test_families",32)):
                raise ScientificGateError("Confirmation family count is smaller than the validated calibration design")
            frozen={"test_families":int(size),"power_sha256":file_sha256(self.output/"outputs/structural_power/power.json"),
                    "test_orbits":orbit_size,"s3_power_sha256":s3_power_hash,
                    "maximum_dataset_seal_sha256":file_sha256(dataset_root/"datasets.json"),"configuration":self.descriptor["identity"]}
            freeze=self.output/"confirmation_freeze.json"
            if freeze.exists() and read_json(freeze)!=frozen:raise ValueError("confirmation decision cannot change")
            atomic_json(freeze,frozen)
            final=self.stage("extract_confirmation",lambda out:extract_study(cfg,out,dataset_root,
                phase="confirmation",test_families=int(size),test_orbits=orbit_size,development_cache=self.output/"outputs/extract_development"))
            local=deepcopy(cfg);local["structural"]["cache_manifests"]=final["cache_dirs"]
            self.stage("structural_confirmation",lambda out:run_structural(local,out,mode="all"))
            if cfg.get("data",{}).get("grammar_sensitivity",True):
                from .grammar import run_grammar_sensitivity
                self.stage("grammar_sensitivity",lambda out:run_grammar_sensitivity(cfg,out,
                    cache_root=self.output/"outputs/extract_confirmation/merged",
                    summary_root=self.output/"outputs/structural_confirmation"))
            if self._native_enabled():self.stage("native_confirmation",lambda out:self._native(out,"confirm"))
        if phase in {"peers","full"}:
            from .peers import run_factorial_stage
            if cfg.get("peers",{}).get("reproduction_anchors",True):self.stage("peer_anchors",lambda out:self._peer_anchors(out))
            if cfg.get("peers",{}).get("factorial_anova",True):self.stage("peer_factorial",lambda out:run_factorial_stage(cfg,out))
        if phase in {"crossmodel","full"} and cfg.get("crossmodel",{}).get("enabled",False):
            from .wordfields import collect_word_fields,analyze_word_fields
            self.stage("crossmodel_extraction",lambda out:collect_word_fields(cfg,out))
            self.stage("crossmodel_transport",lambda out:analyze_word_fields(cfg,out,self.output/"outputs/crossmodel_extraction"))
            self.stage("crossmodel_structural",lambda out:self._structural_crossmodel(out))
        if phase=="extensions":
            from .extensions import run_extensions
            authorized=deepcopy(cfg);authorized.setdefault("execution",{})["allowed_model_execution"]=True
            self.stage("extensions",lambda out:run_extensions(authorized,out,cache_root=self.output/"outputs/extract_confirmation/merged"))
        result={"study_id":cfg["study_id"],"phase":phase,"technical_test":cfg.get("technical_test",False),
                "stages":{p.stem:read_json(p)["status"] for p in sorted(self.state_dir.glob("*.json"))}}
        atomic_json(self.output/"suite_status.json",result)
        return result

    def _native(self,out,mode):
        from .native import run_native
        # prepare and confirm share the exact frozen native output namespace.
        native_root=self.output/"outputs/native_study"
        result=run_native(self.config,native_root,mode=mode)
        if result.get("status") in {"clean_competence_failed","power_gate_not_met_no_test_outcomes"}:
            raise ScientificGateError(f"native scientific gate: {result['status']}; raw outputs preserved at {native_root}")
        return result

    def _has_s3(self):
        return any(str(g).upper().startswith("S3") for g in self.config.get("data",{}).get("groups",[]))

    def _native_enabled(self):
        native=self.config.get("native",{})
        return bool(native.get("enabled",True) and native.get("models",[]))

    def _pilot_eligibility(self,pilot,out):
        flags={g:bool(v.get("eligible",False)) for g,v in pilot.get("family_contrasts",{}).items()}
        if self._has_s3():flags["S3_role"]=bool(pilot.get("S3",{}).get("eligible",False))
        result={"eligible_by_domain":flags,"ineligible_cells":pilot.get("ineligible_cells",[]),
                "status":"eligible_for_registered_power" if flags and all(flags.values()) else "ineligible_context_energy",
                "not_a_software_failure":True,"test_outcomes_collected":False}
        atomic_json(Path(out)/"eligibility.json",result)
        if result["status"]!="eligible_for_registered_power" and not self.config.get("technical_test",False):
            raise ScientificGateError("excluded pilot has insufficient context energy for the registered all-endpoint confirmation; outputs preserved; independent phases remain available")
        return result

    def _embedding(self,out,cache_root):
        from .embedding import run_embedding_floor
        return run_embedding_floor(self.config,out,cache_root=cache_root)

    def _peer_anchors(self,out):
        from .peers import run_reproduction_anchor
        cfg=self.config;configured=cfg.get("peers",{}).get("original_model_config")
        if not configured:raise ValueError("Peer anchors require an explicit original_model_config; no study-model fallback")
        path=Path(configured)
        if not path.is_absolute():path=Path(__file__).resolve().parents[2]/path
        bank=read_json(path);records=bank.get("anchors",[])
        if not records:raise ValueError("Original-model peer anchor configuration is empty")
        results=[]
        for record in records:
            local=deepcopy(cfg);peers=local.setdefault("peers",{})
            spec=record.get("spec",record.get("model"))
            if isinstance(spec,str):
                matching=[m for m in cfg.get("models",[]) if m["id"]==spec]
                if not matching:raise ValueError("Original anchor model needs a pinned full spec")
                spec=matching[0]
            if not isinstance(spec,dict) or not spec.get("id") or not spec.get("revision"):
                raise ValueError("Original anchor model needs an ID and exact revision")
            peers.update({k:v for k,v in record.items() if k not in {"name","spec","model"}})
            peers["model"]=spec
            if "layer" in record and "layers" not in record:peers["layers"]=[int(record["layer"])]
            name=record.get("name",record.get("anchor","anchor"))
            result=run_reproduction_anchor(local,Path(out)/name)
            results.append({"name":name,"model":spec,"result":result})
        summary={"anchors":results,"original_model_config_sha256":file_sha256(path),"study_model_fallback":False}
        atomic_json(Path(out)/"summary.json",summary);return summary

    def _structural_crossmodel(self,out):
        cache=self.output/"outputs/extract_confirmation/merged"
        if not cache.exists():raise ScientificGateError("Structural cross-model transport requires power-frozen confirmation caches")
        from .wordfields import analyze_structural_transport
        return analyze_structural_transport(self.config,out,cache_root=cache)
