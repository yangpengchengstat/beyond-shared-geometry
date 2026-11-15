"""Construction-only checkpoint/hardware checks, permanently technical.

This bounded check never fits a representation or evaluates held-out outcomes.
It checks the actual registered checkpoints and raw-block hook adapter before a
user commits to the full study. It is not a calibration or power pilot.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import time

import numpy as np

from .config import model_key
from .data import build_structural_dataset, build_native_dataset, dataset_identity
from .extraction import extract_features, read_cache, load_cache, _writer_lock
from .hf import PatchSpec, dependency_versions, runtime_from_config
from .io import atomic_json, file_sha256, identity, prepare_run, read_json, source_identity


def _construction_inputs(seed):
    rows = build_structural_dataset("C7", seed=seed, splits=("train",))[:8]
    # Explicit spans make the intended source readout independently inspectable.
    for row in rows:
        begin = row["text"].index(row["target"])
        row["source_span"] = [begin, begin + len(row["target"])]
    native = build_native_dataset("D4_free8", seed=seed, splits=("train",))[0]
    if len(rows) != 8 or any(r["split"] != "train" for r in rows + [native]):
        raise ValueError("Model smoke inputs must be construction-only")
    return rows, native


def _completed(output, descriptor):
    marker = output / "completion.json"
    if not marker.exists():
        return None
    complete = read_json(marker)
    if complete.get("identity") != descriptor["identity"] or complete.get("technical_test") is not True:
        raise ValueError("Model smoke completion identity differs")
    files = complete.get("files", {})
    if "summary.json" not in files:
        raise ValueError("Model smoke completion lacks a summary")
    for relative, digest in files.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or (output / path).is_symlink():
            raise ValueError("Unsafe model smoke completion path")
        if file_sha256(output / path) != digest:
            raise ValueError("Model smoke completion checksum differs")
    result = read_json(output / "summary.json")
    if result.get("technical_test") is not True or result.get("paper_eligible") is not False:
        raise ValueError("Model smoke cannot be promoted to a scientific result")
    for cell in result["models"]:
        for relative in cell["cache_dirs"]:
            load_cache(output / relative)
    return {**result, "resumed": True}


def run_model_smoke(config, output, runtime_factory=None):
    """Check eight source spans and one carrier sham per structural checkpoint.

    Actual inference requires explicit authorization by the calling CLI. The
    runtime factory receives ``(model_config, runtime_config)`` and permits tiny
    local or mock tests. A completed, checksummed run resumes without creating a
    runtime. An interrupted scoring pair can be repeated on restart.
    """
    output = Path(output)
    models = [m for m in config.get("models", []) if m.get("structural", True)]
    if not models:
        raise ValueError("Model smoke requires at least one registered structural checkpoint")
    names = [model_key(model) for model in models]
    if len(names) != len(set(names)):
        raise ValueError("Model smoke checkpoint names collide")
    rows, native = _construction_inputs(int(config.get("seed", 2027101700)))
    technical = deepcopy(config)
    technical.update({"technical_test": True, "paper_eligible": False,
                      "execution_purpose": "construction_only_checkpoint_hardware_smoke"})
    provenance = {"source_sha256": source_identity(Path(__file__).parent),
                  "dependency_versions": dependency_versions(),
                  "construction_rows_sha256": dataset_identity(rows),
                  "native_construction_row_sha256": identity(native),
                  "structural_rows_per_model": 8, "native_rows_per_model": 1,
                  "max_new_tokens": 2, "validation_pilot_test_outcomes_read": False,
                  "operators_fitted": False}
    descriptor = prepare_run(output, technical, "model_smoke", provenance=provenance)
    factory = runtime_factory or runtime_from_config
    with _writer_lock(output):
        complete = _completed(output, descriptor)
        if complete is not None:
            return complete
        cells = []
        for model in models:
            started = time.perf_counter()
            runtime = factory(model, config.get("runtime", {}))
            try:
                layers = sorted(set(map(int, model.get("layers", []))))
                if not layers or min(layers) < 1:
                    raise ValueError("Smoke requires registered non-embedding block layers")
                leaf = output / "caches" / model_key(model)
                source_cache = leaf / "source_spans"
                manifest = extract_features(runtime, rows, layers, "span_mean", source_cache)
                cached_rows, arrays, _ = read_cache(source_cache)
                if any(r["split"] != "train" for r in cached_rows):
                    raise ValueError("Smoke cache contains a held-out record")
                dimensions = {array.shape[1] for array in arrays.values()}
                if len(dimensions) != 1:
                    raise ValueError("Registered layers have incompatible hidden dimensions")
                dimension = dimensions.pop()
                expected = model.get("hidden_dimension")
                if expected is not None and int(expected) != dimension:
                    raise ValueError("Checkpoint hidden dimension differs from the registered configuration")
                # The first registered block is an early site (.2 in the main
                # panel); no validation-driven site selection occurs here.
                early_layer = layers[0]
                carrier_cache = leaf / "native_carrier"
                extract_features(runtime, [native], [early_layer], "carrier_span", carrier_cache)
                _, carrier, _ = read_cache(carrier_cache)
                if carrier[early_layer].shape != (1, dimension):
                    raise ValueError("Carrier hook shape differs from source feature shape")
                clean = runtime.score(native["text"], native["candidates"], generation=True, max_new_tokens=2)
                patch = PatchSpec(layer=early_layer, span=native["carrier_span"],
                                  delta=np.zeros(dimension, dtype=np.float32))
                sham = runtime.score(native["text"], native["candidates"], patch=patch,
                                     generation=True, max_new_tokens=2)
                clean_scores = np.asarray(clean["candidate_logprobs"], dtype=np.float64)
                sham_scores = np.asarray(sham["candidate_logprobs"], dtype=np.float64)
                expected_shape = (len(native["candidates"]),)
                if clean_scores.shape != expected_shape or sham_scores.shape != expected_shape or not (
                    np.isfinite(clean_scores).all() and np.isfinite(sham_scores).all()
                ):
                    raise ValueError("Smoke candidate scores are malformed or nonfinite")
                equal_scores = np.array_equal(clean_scores, sham_scores)
                equal_generation = clean.get("generated_text") == sham.get("generated_text")
                if not equal_scores or not equal_generation or clean.get("generated_text") is None:
                    raise RuntimeError("Zero-delta carrier hook changes clean candidate scores or generation")
                cell = {"model": model, "runtime_identity": runtime.identity(),
                        "technical_test": True, "paper_eligible": False,
                        "construction_rows": len(rows), "native_construction_rows": 1,
                        "layers": layers, "hidden_dimension": dimension,
                        "native_domain": "D4_free8", "early_carrier_layer": early_layer,
                        "candidate_count": len(native["candidates"]),
                        "candidate_token_counts": clean.get("candidate_token_counts"),
                        "zero_delta_candidate_scores_exactly_equal": equal_scores,
                        "zero_delta_generation_exactly_equal": equal_generation,
                        "maximum_candidate_logprob_difference": float(np.max(np.abs(clean_scores-sham_scores))),
                        "max_new_tokens": 2, "seconds": time.perf_counter()-started,
                        "source_cache_key": manifest["cache_key"],
                        "cache_dirs": [str(source_cache.relative_to(output)), str(carrier_cache.relative_to(output))],
                        "telemetry": list(getattr(runtime, "telemetry", []))}
                loaded = getattr(runtime, "model", None)
                if loaded is not None:
                    mapping = getattr(loaded, "hf_device_map", None)
                    cell["actual_device_map"] = {str(k):str(v) for k,v in mapping.items()} if mapping else {
                        "input_embeddings": str(loaded.get_input_embeddings().weight.device)}
                cells.append(cell)
                atomic_json(output / "progress.json", {"technical_test":True, "completed_models":cells})
            except Exception as error:
                atomic_json(output / "failure.json", {"technical_test":True, "paper_eligible":False,
                    "model":model["id"], "error_type":type(error).__name__, "error":str(error),
                    "telemetry":list(getattr(runtime,"telemetry",[]))})
                raise
            finally:
                close = getattr(runtime, "close", getattr(runtime, "clear_gpu", None))
                if close is not None:
                    close()
        result = {"status":"passed", "technical_test":True, "paper_eligible":False,
                  "purpose":"checkpoint/hardware/hook validation; no scientific inference",
                  "identity":descriptor["identity"], "models":cells, "resumed":False,
                  "validation_pilot_test_outcomes_read":False, "operators_fitted":False,
                  "runtime_batch_policy_changed":False}
        atomic_json(output / "summary.json", result)
        files = {str(path.relative_to(output)):file_sha256(path) for path in sorted(output.rglob("*"))
                 if path.is_file() and path.name not in (".writer.lock", "completion.json", "failure.json")}
        atomic_json(output / "completion.json", {"identity":descriptor["identity"],
                    "technical_test":True, "files":files})
        return result
