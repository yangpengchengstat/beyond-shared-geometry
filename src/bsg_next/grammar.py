"""Descriptive grammar-root sensitivity of existing confirmation contrasts.

No activations, model inference, operator fits, site choices, or primary
decisions are recomputed here. The generated scenario families are conditional
draws from a fixed grammar bank, not independently authored new grammars.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from .data import dataset_identity
from .groups import get_group
from .io import atomic_json, file_sha256, identity, prepare_run, read_json


def _model_id(value):
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        raise ValueError("Grammar sensitivity lacks a model identity")
    name = value.get("id", value.get("model_id"))
    if not name:
        raise ValueError("Grammar sensitivity lacks a model identifier")
    return str(name)


def _relative_file(root, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or (root/path).is_symlink():
        raise ValueError("Unsafe grammar input reference")
    return root/path


def _cache_metadata(path, family_roots):
    """Verify the row payload bound to the primary cell, without reading H."""
    manifest = read_json(path/"manifest.json")
    if manifest.get("status") != "complete" or manifest.get("schema_version") != 1:
        raise ValueError("Grammar sensitivity requires complete activation cache metadata")
    if "descriptor" in manifest:
        descriptor = manifest["descriptor"]
        if identity(descriptor) != manifest.get("cache_key") or any(
            manifest.get(key) != descriptor.get(key) for key in
            ("dataset_sha256", "model", "layers", "readout", "extraction_code_sha256")
        ):
            raise ValueError("Grammar cache descriptor identity mismatch")
    rows_path = _relative_file(path, manifest["rows_path"])
    if file_sha256(rows_path) != manifest["rows_sha256"]:
        raise ValueError("Grammar row checksum mismatch")
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != manifest["n_rows"] or dataset_identity(rows) != manifest["dataset_sha256"]:
        raise ValueError("Grammar row dataset identity mismatch")
    if not rows or len({r["row_id"] for r in rows}) != len(rows):
        raise ValueError("Grammar rows are empty or duplicated")
    group = get_group(rows[0]["domain"]).name
    expected = set()
    target_split = "test_joint" if group == "S3_role" else "test"
    for row in rows:
        if get_group(row["domain"]).name != group:
            raise ValueError("Grammar cache mixes domains")
        family = str(row["family_id"])
        root = row.get("grammar_root_id")
        if not isinstance(root, str) or not root:
            raise ValueError("Every grammar family requires an explicit grammar_root_id")
        if "grammar_index" in row and root != f"grammar_{int(row['grammar_index']):02d}":
            raise ValueError("Family grammar root disagrees with its generator index")
        context = str(row.get("context_split", row.get("split")))
        mapping = (root, context)
        if family in family_roots and family_roots[family] != mapping:
            raise ValueError("Family grammar root or context partition changes across rows/caches")
        family_roots[family] = mapping
        if row.get("split") != target_split:
            continue
        if context != "test" or (group == "S3_role" and row.get("entity_split") != "test"):
            raise ValueError("Primary grammar contrasts contain construction/validation/pilot or non-test entities")
        expected.add((family, str(row.get("orbit_id", "single"))))
    return {"group":group, "model":_model_id(manifest["model"]),
            "layers":{str(layer) for layer in manifest["layers"]},
            "expected_pairs":expected, "target_split":target_split,
            "digest":file_sha256(path/"manifest.json"),
            "rows_sha256":manifest["rows_sha256"]}


def _contrasts(record, field, expected):
    values = {}
    for name, value in record.get(field, {}).items():
        parts = name.split("|")
        if len(parts) != 2:
            raise ValueError("Malformed family/orbit contrast key")
        key = tuple(parts)
        if key not in expected:
            raise ValueError("Primary grammar contrast references an unknown or non-test family/orbit")
        value = float(value)
        if not np.isfinite(value):
            raise ValueError("Nonfinite primary grammar contrast")
        values[key] = value
    if set(values) != expected:
        raise ValueError("Primary grammar contrast omits held-out family/orbit cells")
    return values


def _effects(table):
    return {endpoint:float(values.mean()) if values.size else None for endpoint,values in table.items()}


def _describe_group(group, averaged, family_roots):
    pairs = sorted(averaged["pred"])
    families = sorted({f for f,_ in pairs})
    orbits = sorted({o for _,o in pairs})
    if set(pairs) != {(f,o) for f in families for o in orbits}:
        raise ValueError("Grammar sensitivity requires a balanced family-by-entity table")
    table = {endpoint:np.asarray([[averaged[endpoint][f,o] for o in orbits] for f in families],float)
             for endpoint in ("pred","ctx")}
    roots = sorted({family_roots[f][0] for f in families})
    per_root, leave_one = {}, {}
    for root in roots:
        mask = np.asarray([family_roots[f][0] == root for f in families])
        selected = {endpoint:values[mask] for endpoint,values in table.items()}
        remaining = {endpoint:values[~mask] for endpoint,values in table.items()}
        per_root[root] = {"family_count":int(mask.sum()), "families":[f for f,m in zip(families,mask) if m],
            "entity_orbit_count":len(orbits), "effects":_effects(selected),
            "entity_effects":{orbit:{endpoint:float(values[:,j].mean()) for endpoint,values in selected.items()}
                              for j,orbit in enumerate(orbits)}}
        leave_one[root] = {"excluded_family_count":int(mask.sum()), "remaining_family_count":int((~mask).sum()),
            "entity_orbit_count":len(orbits), "effects":_effects(remaining),
            "entity_effects":{orbit:{endpoint:float(values[:,j].mean()) if len(values) else None
                                      for endpoint,values in remaining.items()} for j,orbit in enumerate(orbits)},
            "status":"descriptive" if (~mask).any() else "no_remaining_families"}
    return {"group":group, "family_count":len(families), "entity_orbit_count":len(orbits),
            "observed_root_count":len(roots), "effects":_effects(table), "per_root":per_root,
            "leave_one_root_out":leave_one,
            "family_effects":{family:{endpoint:float(values[i].mean()) for endpoint,values in table.items()}
                              for i,family in enumerate(families)},
            "family_orbit_contrasts":{endpoint:{f+"|"+o:value for (f,o),value in values.items()}
                                      for endpoint,values in averaged.items()},
            "averaging":"equal fixed model/depth conditions within family/orbit; equal families and entity orbits afterward",
            "crossing_preserved":group == "S3_role",
            "leave_one_root_out_refits":False}


def run_grammar_sensitivity(config, output, cache_root, summary_root):
    """Summarize primary confirmation effects by the declared grammar root.

    ``cache_root`` contains the merged confirmation cache directories;
    ``summary_root`` contains structural ``summary.json`` and ``cells/*.json``.
    Only q-primary C7/D4 test and S3 joint-test contrasts are included. S3
    entity columns are retained when roots are removed along the family axis.
    All intervals and p-values remain those of the separate primary analysis.
    """
    cache_root, summary_root, output = Path(cache_root), Path(summary_root), Path(output)
    summary = read_json(summary_root/"summary.json")
    if summary.get("mode") == "pilot":
        raise ValueError("Grammar sensitivity consumes confirmation, never excluded-pilot outcomes")
    if not summary.get("panel_complete", False):
        raise ValueError("Grammar sensitivity requires the complete registered fixed model/depth panel")
    cache_paths = [cache_root] if (cache_root/"manifest.json").exists() else sorted(
        {p.parent for p in cache_root.rglob("manifest.json")})
    if not cache_paths:
        raise ValueError("Grammar sensitivity requires merged cache metadata")
    family_roots, caches, cache_provenance = {}, {}, {}
    for path in cache_paths:
        info = _cache_metadata(path, family_roots)
        if info["digest"] in caches:
            raise ValueError("Duplicate grammar cache identity")
        caches[info["digest"]] = info
        cache_provenance[str(path.relative_to(cache_root))] = {
            "manifest_sha256":info["digest"], "rows_sha256":info["rows_sha256"]}
    cell_paths = sorted((summary_root/"cells").glob("*.json"))
    if not cell_paths:
        raise ValueError("Grammar sensitivity requires the primary structural cell records")
    grouped = defaultdict(lambda:{"pred":defaultdict(list), "ctx":defaultdict(list)})
    coverage = defaultdict(set)
    eligibility = defaultdict(list)
    q = int(config.get("structural",{}).get("q_primary",16))
    for path in cell_paths:
        entries = read_json(path)
        if not isinstance(entries,list):
            raise ValueError("Structural cell file must contain a record list")
        for record in entries:
            if not record.get("is_primary_dimension",True) or record.get("status") == "secondary_rank_failure":
                continue
            group = get_group(record["group"]).name
            target_split = "test_joint" if group == "S3_role" else "test"
            if record.get("test_split") != target_split:
                continue
            if int(record.get("q",q)) != q:
                raise ValueError("A primary grammar cell has the wrong coordinate dimension")
            info = caches.get(record.get("cache_identity"))
            if info is None:
                raise ValueError("Grammar cell references a different or absent merged-cache identity")
            model, layer = _model_id(record["model"]), str(record["layer"])
            if info["group"] != group or info["model"] != model or layer not in info["layers"]:
                raise ValueError("Grammar cell domain/model/layer disagrees with its cache")
            condition = (model,layer)
            if condition in coverage[group]:
                raise ValueError("Duplicate model/depth grammar cell")
            coverage[group].add(condition)
            eligibility[group].append(bool(record.get("eligible",False)))
            for endpoint,field in (("pred","family_orbit_prediction"),("ctx","family_orbit_context")):
                for key,value in _contrasts(record,field,info["expected_pairs"]).items():
                    grouped[group][endpoint][key].append(value)
    expected_conditions = {(str(model["id"]),str(layer)) for model in config.get("models",[])
                           if model.get("structural",True) for layer in model.get("layers",[])}
    expected_groups = {get_group(g).name for g in config.get("structural",{}).get("groups",["C7","D4","S3"])}
    if set(grouped) != expected_groups:
        raise ValueError("Grammar sensitivity is missing a registered primary domain")
    reports = {}
    external_bank = bool(config.get("data",{}).get("external_manifests"))
    declared_roots = [f"grammar_{index:02d}" for index in range(12)] if not external_bank else None
    for group,entry in sorted(grouped.items()):
        if expected_conditions and coverage[group] != expected_conditions:
            raise ValueError("Grammar sensitivity has incomplete or extra fixed model/depth conditions")
        count = len(coverage[group])
        if not count or any(len(values) != count for endpoint in entry.values() for values in endpoint.values()):
            raise ValueError("Unequal model/depth coverage within grammar family/orbit")
        averaged = {endpoint:{key:float(np.mean(values)) for key,values in values_by_pair.items()}
                    for endpoint,values_by_pair in entry.items()}
        report = _describe_group(group,averaged,family_roots)
        report["declared_fixed_generator_root_count"] = 12 if not external_bank else None
        report["unobserved_generator_roots"] = sorted(set(declared_roots)-set(report["per_root"])) if declared_roots else None
        report["context_primary_eligible"] = all(eligibility[group])
        report["fixed_model_depth_conditions"] = count
        # Check aggregation against the already published primary summaries.
        if group == "S3_role":
            original = summary.get("S3",{})
            families = sorted({f for f,_ in averaged["pred"]})
            orbits = sorted({o for _,o in averaged["pred"]})
            if original.get("families") != families or original.get("orbits") != orbits:
                raise ValueError("S3 primary summary and grammar table have different crossed units")
            contrasts = (original.get("decisions") or {}).get("contrasts",{})
            for endpoint in ("pred","ctx"):
                if endpoint in contrasts and not np.isclose(contrasts[endpoint]["mean"],report["effects"][endpoint],atol=1e-12,rtol=1e-12):
                    raise ValueError("S3 grammar averaging disagrees with primary summary")
        else:
            original = summary.get("family_contrasts",{}).get(group,{})
            families = sorted(report["family_effects"])
            if original.get("families") != families:
                raise ValueError("Primary summary and grammar table contain different families")
            for endpoint in ("pred","ctx"):
                values = [report["family_effects"][f][endpoint] for f in families]
                if not np.allclose(original.get(endpoint,[]),values,atol=1e-12,rtol=1e-12):
                    raise ValueError("Grammar averaging disagrees with primary family summary")
        reports[group] = report
    provenance = {"module_sha256":file_sha256(Path(__file__)),
        "summary_sha256":file_sha256(summary_root/"summary.json"), "cache_metadata":cache_provenance,
        "cell_sha256":{str(p.relative_to(summary_root)):file_sha256(p) for p in cell_paths}}
    descriptor = prepare_run(output,config,"grammar_root_sensitivity",provenance=provenance)
    result = {"status":"complete", "identity":descriptor["identity"],
        "technical_test":bool(config.get("technical_test",False)), "new_model_forwards":0,
        "activation_payloads_read":False, "operators_refitted":False, "primary_decisions_changed":False,
        "groups":reports, "scope":("descriptive scenario-family sensitivity conditional on the supplied root bank"
            if external_bank else "descriptive scenario-family sensitivity conditional on the fixed 12-root grammar generator"),
        "independently_authored_grammar_population_inference":False,
        "external_bank_supplied":external_bank,
        "external_bank_caveat":"Supplied root labels describe that particular bank; no population of independently authored grammars is inferred.",
        "uncertainty":"No additional p-values or confidence intervals; leave-root-out effects are descriptive and retain fitted models and validation choices.",
        "primary_strata_only":"C7/D4 held-out families; S3 joint held-out families and entity triples; no validation, pilot, support, or training contrast outcomes"}
    atomic_json(output/"summary.json",result)
    return result
