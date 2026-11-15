"""Read-only host checks; model access checks fetch metadata only when requested."""
from __future__ import annotations

import importlib.metadata
import platform
from pathlib import Path
import shutil
import sys

from .io import file_sha256
from .config import validate_config


def preflight(config, *, require_cuda=False, check_access=False, output=None):
    validate_config(config)
    packages = {}
    for package in ("numpy","scipy","pandas","PyYAML","torch","transformers","accelerate","huggingface-hub"):
        try:packages[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:packages[package]=None
    errors=[]
    for name in ("numpy","scipy","pandas","PyYAML"):
        if packages[name] is None:errors.append("missing dependency: "+name)
    gpu={"checked":False}
    if require_cuda:
        if not packages["torch"]:errors.append("GPU runtime requires torch")
        else:
            import torch
            gpu={"checked":True,"available":torch.cuda.is_available(),"torch_cuda":torch.version.cuda,"devices":[]}
            if not gpu["available"]:errors.append("CUDA is unavailable; no research model was loaded")
            else:
                for i in range(torch.cuda.device_count()):
                    gpu["devices"].append({"index":i,"name":torch.cuda.get_device_name(i),
                                           "total_bytes":torch.cuda.get_device_properties(i).total_memory})
        for name in ("transformers","accelerate","huggingface-hub"):
            if packages[name] is None:errors.append("missing GPU dependency: "+name)
        if packages["transformers"] != "4.57.1":errors.append("tested Transformers runtime is 4.57.1; install the RunPod constraints")
    root=Path(__file__).resolve().parents[2]
    frozen=root/"release/source_manifest.json"
    legacy=[]
    if frozen.exists():
        import json
        for relative,digest in json.loads(frozen.read_text())["files"].items():
            if file_sha256(root/relative)!=digest:legacy.append(relative)
    if legacy:errors.append("historical locked numerical source differs: "+", ".join(legacy))
    path=Path(output or ".").resolve()
    while not path.exists():path=path.parent
    disk=shutil.disk_usage(path)
    access=[]
    if check_access:
        from huggingface_hub import HfApi
        api=HfApi()
        for model in config["models"]:
            try:
                info=api.model_info(model["id"],revision=model["revision"])
                if info.sha != model["revision"]:raise ValueError("resolved model commit mismatch")
                access.append({"model":model["id"],"commit":info.sha,"metadata_accessible":True,
                               "gated":bool(info.gated),"weights_access_not_tested":True})
            except Exception as exc:
                access.append({"model":model["id"],"metadata_accessible":False,"error_type":type(exc).__name__})
                errors.append("model metadata is inaccessible: "+model["id"])
    return {"ok":not errors,"errors":errors,"python":sys.version.split()[0],"platform":platform.system(),
            "packages":packages,"gpu":gpu,"disk_free_bytes":disk.free,"disk_total_bytes":disk.total,
            "historical_source_mismatches":legacy,"model_metadata":access,
            "model_weights_loaded":False,"secrets_or_environment_values_recorded":False}
