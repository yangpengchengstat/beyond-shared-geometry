#!/usr/bin/env python3
"""Prepared-only offline sensitivity launcher. No analysis without explicit opt-in.

`preflight` reads immutable metadata/headers only and never imports NumPy.
`run --execute-offline-analysis` is a future authorized research computation.
"""
from __future__ import annotations
import argparse
import csv
import datetime
import fcntl
import importlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import traceback
import uuid

from archive_inputs import ARCHIVES, InputError, Study2Inputs, canonical, file_sha, parse_json, safe_name, sha

HERE = Path(__file__).resolve().parent


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path, value):
    """Exclusive write: existing scientific receipts are never overwritten."""
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")


def publish_receipt(path, value):
    """Atomic no-overwrite commit; interrupted attempts never look complete."""
    path = Path(path)
    temporary = path.with_name(".receipt-" + uuid.uuid4().hex + ".tmp")
    with temporary.open("x", encoding="utf-8") as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n"); f.flush(); os.fsync(f.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def source_contract():
    manifest_path = HERE / "source_manifest.json"
    manifest = parse_json(manifest_path.read_bytes())
    if not isinstance(manifest.get("files"), dict) or not manifest["files"]:
        raise InputError("missing source manifest files")
    required = {"run.py", "archive_inputs.py", "protocol.json", "historical_protocol.json",
                "core/__init__.py", "core/spectral_dimension_core.py",
                "core/_vendor/__init__.py", "core/_vendor/gamma_r_frozen.py"}
    if not required <= set(manifest["files"]):
        raise InputError("source manifest omits an executable or protocol")
    for rel, expected in manifest["files"].items():
        safe_name(rel)
        p = HERE / rel
        if p.is_symlink() or not p.is_file() or file_sha(p) != expected:
            raise InputError("source hash mismatch: " + rel)
    protocol = parse_json((HERE / "protocol.json").read_bytes())
    historical = parse_json((HERE / "historical_protocol.json").read_bytes())
    required_values = {"dimensions": [8, 16], "execution_order": [16, 8], "planned_cells": 48,
                       "available_source_dimension": 16, "automatic_scientific_promotion": False,
                       "q32_status": "unavailable_same_study2_cache"}
    if any(protocol.get(k) != v for k, v in required_values.items()):
        raise InputError("prepared dimensions/scope contract changed; new protocol required")
    if protocol.get("expected_runtime") != {"python_major_minor": "3.11", "numpy": "2.2.6"}:
        raise InputError("numerical runtime contract differs")
    if historical.get("rank") != 16 or historical.get("planned_cells") != 48 or historical.get("extra_centering") is not False:
        raise InputError("historical protocol differs")
    return protocol, historical, {"manifest_sha256": file_sha(manifest_path), "files": manifest["files"]}


def output_path(value, repo):
    output = Path(value).expanduser().resolve()
    repo = Path(repo).resolve()
    if output == HERE or output.is_relative_to(HERE) or HERE.is_relative_to(output):
        raise InputError("output must be outside the immutable prepared-code directory")
    for rel, _ in ARCHIVES.values():
        archive = (repo / rel).resolve()
        if output == archive or output.is_relative_to(archive.parent) or archive.is_relative_to(output):
            raise InputError("output overlaps a frozen archive or its directory")
    return output


def fresh(path):
    if path.exists() or path.is_symlink():
        raise InputError("output exists; choose a new directory")
    path.mkdir(parents=True, exist_ok=False)


def snapshot_source(root, source):
    directory = root / "source_snapshot"
    for rel, expected in source["files"].items():
        src = HERE / rel
        if file_sha(src) != expected:
            raise InputError("source changed before snapshot: " + rel)
        dst = directory / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        with dst.open("xb") as f:
            f.write(src.read_bytes())
    with (directory / "source_manifest.json").open("xb") as f:
        f.write((HERE / "source_manifest.json").read_bytes())


def verify_snapshot(root, source):
    directory = root / "source_snapshot"
    if file_sha(directory / "source_manifest.json") != source["manifest_sha256"]:
        raise InputError("saved source manifest differs")
    for rel, expected in source["files"].items():
        p = directory / rel
        if p.is_symlink() or file_sha(p) != expected:
            raise InputError("saved source snapshot differs: " + rel)


def make_plan(inputs):
    return [{"unit": c["cell"] + "--" + image, "cell": c["cell"], "domain": c["domain"],
             "subgroup_id": image, "kind": "canonical_semantic" if image == "semantic_canonical" else "image"}
            for c in inputs.cells
            for image in ["semantic_canonical"] + [g["id"] for g in inputs.alt_plan["subgroups"][c["domain"]]]]


def receipt(root, dimension, unit, binding):
    path = root / "units" / str(dimension) / unit / "receipt.json"
    if not path.exists():
        return None
    if path.is_symlink():
        raise InputError("symlink receipt refused")
    value = parse_json(path.read_bytes())
    if value.get("binding") != binding or value.get("unit") != unit or value.get("dimension") != dimension:
        raise InputError("resume receipt binding differs")
    if value.get("terminal") is not True or not isinstance(value.get("files"), dict) or "summary.json" not in value["files"] or not set(value["files"]) <= {"summary.json", "diagnostics.npz"}:
        raise InputError("resume receipt file/terminal contract differs")
    attempt = safe_name(value["attempt"])
    if not attempt.startswith(f"attempts/{dimension}/{unit}/"):
        raise InputError("resume attempt path differs")
    for rel, expected in value["files"].items():
        safe_name(rel)
        p = root / attempt / rel
        if p.is_symlink() or not p.resolve().is_relative_to(root.resolve()) or file_sha(p) != expected:
            raise InputError("resume artifact hash differs")
    summary = parse_json((root / attempt / "summary.json").read_bytes())
    if summary.get("unit") != unit or summary.get("dimension") != dimension or summary.get("status") != value.get("status") or summary.get("subgroup_id") != unit.rsplit("--", 1)[1]:
        raise InputError("resume summary identity differs")
    return summary


def commit_unit(root, dimension, info, binding, summary, arrays, np):
    rel = f"attempts/{dimension}/{info['unit']}/{uuid.uuid4().hex}"
    attempt = root / rel
    attempt.mkdir(parents=True, exist_ok=False)
    summary.update(unit=info["unit"], cell=info["cell"], kind=info["kind"], dimension=dimension,
                   domain=info["domain"], subgroup_id=info["subgroup_id"])
    write_json(attempt / "summary.json", summary)
    files = {"summary.json": file_sha(attempt / "summary.json")}
    if arrays:
        with (attempt / "diagnostics.npz").open("xb") as f:
            np.savez_compressed(f, **arrays)
        files["diagnostics.npz"] = file_sha(attempt / "diagnostics.npz")
    folder = root / "units" / str(dimension) / info["unit"]
    folder.mkdir(parents=True, exist_ok=True)
    publish_receipt(folder / "receipt.json", {"unit": info["unit"], "dimension": dimension, "binding": binding,
                                             "status": summary["status"], "terminal": True, "attempt": rel, "files": files})


def reports(root, plan, binding, core, anchor_passed, inputs):
    directory = root / "reports" / ("snapshot-" + uuid.uuid4().hex)
    directory.mkdir(parents=True, exist_ok=False)
    unit_rows, cell_rows, registrations = [], [], []
    for dimension in [16, 8]:
        by_cell = {}
        for info in plan:
            summary = receipt(root, dimension, info["unit"], binding)
            if summary is None:
                summary = dict(info, dimension=dimension, status="blocked_anchor_validation" if dimension == 8 and not anchor_passed else "not_completed")
            by_cell.setdefault(info["cell"], []).append(summary)
            metrics = summary.get("metrics", {})
            unit_rows.append(dict(unit=info["unit"], cell=info["cell"], domain=info["domain"], dimension=dimension,
                                  kind=info["kind"], subgroup_id=info["subgroup_id"], status=summary["status"],
                                  reason=summary.get("reason", ""), anchor_passed=summary.get("anchor_check", {}).get("passed", ""),
                                  **{k: metrics.get(k) for k in ["gamma", "eta", "L_lower", "L_upper", "p_minus", "p_plus", "L_identity", "e_triv", "b_triv", "delta_type_normalized"]}))
        for cell, units in by_cell.items():
            semantic = next(u for u in units if u["subgroup_id"] == "semantic_canonical")
            images = [u for u in units if u["subgroup_id"] != "semantic_canonical"]
            registration = core.registration_summary(images, semantic["domain"], dimension)
            registration.update(cell=cell)
            registrations.append(registration)
            metrics = semantic.get("metrics", {})
            cell_rows.append(dict(cell=cell, domain=semantic["domain"], dimension=dimension,
                                  role="original_dimension_validation_only" if dimension == 16 else "new_downward_truncation_analysis",
                                  canonical_status=semantic["status"], registration_status=registration["status"],
                                  image_count_planned=registration["planned"], image_count_evaluable=registration["spectral_evaluable"],
                                  **{k: metrics.get(k) for k in ["gamma", "eta", "L_lower", "L_upper", "p_minus", "p_plus", "L_identity"]},
                                  **{k: registration.get(k) for k in ["classification", "rank_low", "rank_high", "d_top", "d_median"]}))
    for name, rows in [("image_summary.csv", unit_rows), ("cell_summary.csv", cell_rows)]:
        with (directory / name).open("x", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    paired = []
    for cell in inputs.cells:
        old = inputs.semantic_baselines[cell["cell"]]["metrics"]
        old_eta = old["eta_LS"]
        baseline = {"gamma": old["gamma"], "eta": old_eta, "L_lower": old["L_lower"], "L_upper": old["L_upper"],
                    "p_minus": max(0.0, old["L_lower"] - old_eta), "p_plus": old["L_upper"] - old_eta}
        current = next(row for row in cell_rows if row["cell"] == cell["cell"] and row["dimension"] == 8)
        row = {"cell": cell["cell"], "domain": cell["domain"], "q8_status": current["canonical_status"],
               "original_q16_reference": "immutable original canonical summary; not recomputed anchor values",
               "different_metrics_warning": "dimensions define different observation spaces and whitening metrics; differences are sensitivity summaries, not common-target optimization"}
        for key, value in baseline.items():
            row["original_q16_" + key] = value
            row["q8_" + key] = current[key]
            row["q8_minus_original_q16_" + key] = current[key] - value if current["canonical_status"] == "complete" else None
        for key in ["rank_low", "rank_high"]:
            row["original_q16_" + key] = inputs.original_ranks[cell["cell"]][key]
            row["q8_" + key] = current[key]
        paired.append(row)
    with (directory / "paired_dimension_changes.csv").open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(paired[0])); writer.writeheader(); writer.writerows(paired)
    write_json(directory / "registrations.json", {"cells": registrations, "p_values": False, "rank_is_Lstar_order": False})
    domains = []
    for dimension in [16, 8]:
        for domain in ["C7", "D4"]:
            rows = [x for x in cell_rows if x["dimension"] == dimension and x["domain"] == domain]
            complete = [x for x in rows if x["canonical_status"] == "complete" and x["registration_status"] == "complete_descriptive"]
            all_complete = len(rows) == len(complete) == 24
            domain_summary = {"domain": domain, "dimension": dimension, "planned_cells": 24,
                              "fully_evaluable_cells": len(complete), "aggregate_status": "complete_descriptive" if all_complete else "withheld_incomplete_panel",
                              "means": {k: sum(x[k] for x in complete) / 24 if all_complete else None
                                        for k in ["gamma", "eta", "L_lower", "L_upper", "p_minus", "p_plus", "L_identity", "d_top", "d_median"]},
                              "medians": {k: statistics.median(x[k] for x in complete) if all_complete else None
                                          for k in ["gamma", "eta", "L_lower", "L_upper", "p_minus", "p_plus", "L_identity", "d_top", "d_median"]},
                              "ranges": {k: {"min": min(x[k] for x in complete), "max": max(x[k] for x in complete)} if all_complete else None
                                         for k in ["gamma", "eta", "L_lower", "L_upper", "p_minus", "p_plus", "L_identity", "d_top", "d_median"]},
                              "positive_premium_count": sum(x["p_minus"] > inputs.historical_protocol["scalar_atol"] + inputs.historical_protocol["scalar_rtol"] * abs(x["p_minus"]) for x in complete) if all_complete else None,
                              "positive_premium_rule": "p_minus > scalar_atol + scalar_rtol * abs(p_minus), comparison against zero",
                              "positive_premium_tolerance": {k: inputs.historical_protocol[k] for k in ["scalar_atol", "scalar_rtol"]},
                              "classification_counts_status": "complete_descriptive" if all_complete else "incomplete_descriptive_counts_only",
                              "classification_counts": {k: sum(x.get("classification") == k for x in complete) for k in ["unique_leading", "tied_leading", "behind"]},
                              "missing_or_failed_cells": [x["cell"] for x in rows if x not in complete]}
            domains.append(domain_summary)
    coverage = {"status": "completed" if all(x["aggregate_status"] == "complete_descriptive" for x in domains) else "incomplete_or_not_evaluable",
                "anchor_validation_passed": anchor_passed, "planned_spectral_units": 6000,
                "unit_status_counts": {status: sum(r["status"] == status for r in unit_rows) for status in sorted({r["status"] for r in unit_rows})},
                "domain_summaries": domains, "primary_16_results_replaced": False, "automatic_scientific_promotion": False,
                "scope": "separate metrics at dimensions8/16; no causal, population or all-dimensions inference"}
    write_json(directory / "coverage.json", coverage)
    return directory, coverage


def run_analysis(args, protocol, historical, source):
    if not args.execute_offline_analysis:
        raise InputError("Research computation requires --execute-offline-analysis; preparation does not authorize execution")
    if platform.python_version_tuple()[:2] != ("3", "11"):
        raise InputError("Frozen numerical execution requires Python3.11; metadata preflight is version-independent")
    np = importlib.import_module("numpy")
    if np.__version__ != "2.2.6":
        raise InputError("Frozen numerical execution requires NumPy2.2.6; no installation is performed automatically")
    core = importlib.import_module("core.spectral_dimension_core")
    root = output_path(args.output, args.repo)
    with Study2Inputs(args.repo, historical) as inputs:
        metadata = inputs.preflight()
        for domain in ["C7", "D4"]:
            if core.subgroup_specs(domain) != inputs.alt_plan["subgroups"][domain]:
                raise InputError("canonical image enumeration differs from original plan")
        plan = make_plan(inputs)
        binding = {"source_manifest_sha256": source["manifest_sha256"], "protocol_sha256": sha(canonical(protocol)),
                   "input_manifest_sha256": metadata["input_manifest_sha256"], "archive_sha256": metadata["archive_sha256"],
                   "plan_sha256": sha(canonical(plan)), "numpy": np.__version__, "python": platform.python_version()}
        if args.resume:
            if not root.is_dir() or root.is_symlink():
                raise InputError("resume requires an existing ordinary run directory")
            prior = parse_json((root / "run_manifest.json").read_bytes())
            if prior.get("binding") != binding:
                raise InputError("resume source/input/runtime/protocol binding differs")
            if parse_json((root / "plan.json").read_bytes()) != {"dimensions": [16, 8], "units_per_dimension": plan}:
                raise InputError("saved plan differs from the immutable planned grid")
            verify_snapshot(root, source)
        else:
            fresh(root)
            write_json(root / "run_manifest.json", {"created_at": now(), "binding": binding, "protocol": protocol,
                                                    "source": source, "status": "execution_authorized_by_explicit_command",
                                                    "primary_16_results_replaced": False})
            write_json(root / "input_preflight.json", metadata)
            write_json(root / "plan.json", {"dimensions": [16, 8], "units_per_dimension": plan})
            snapshot_source(root, source)
        with (root / ".controller.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            anchor_passed = True
            for dimension in [16, 8]:
                cache_cell, bank = None, None
                for index, info in enumerate(plan):
                    prior = receipt(root, dimension, info["unit"], binding)
                    if prior is not None:
                        if dimension == 16 and not prior.get("anchor_check", {}).get("passed", False):
                            anchor_passed = False
                        continue
                    if cache_cell != info["cell"]:
                        bank = inputs.load_test_bank(info["cell"]); cache_cell = info["cell"]
                    meta = next(m for m in metadata["cells"] if m["cell"] == info["cell"])
                    try:
                        if info["kind"] == "canonical_semantic":
                            summary, arrays = core.analyze_semantic(bank, info["domain"], dimension, meta)
                            old = inputs.semantic_baselines[info["cell"]]
                        else:
                            summary, arrays = core.analyze_image(bank, info["domain"], dimension, info["subgroup_id"], meta)
                            old = inputs.baselines[info["unit"]]
                        if dimension == 16:
                            summary["anchor_check"] = core.validate_anchor(summary, old)
                            if not summary["anchor_check"]["passed"]:
                                anchor_passed = False
                                summary["status"] = "anchor_validation_failed"
                    except Exception as exc:
                        summary, arrays = {"status": "technical_failure", "reason": type(exc).__name__ + ": " + str(exc), "traceback": traceback.format_exc()}, {}
                        if dimension == 16:
                            anchor_passed = False
                    commit_unit(root, dimension, info, binding, summary, arrays, np)
                    if (index + 1) % 25 == 0:
                        print(f"PROGRESS q={dimension} {index+1}/{len(plan)}", flush=True)
                if dimension == 16:
                    rank_checks = []
                    for cell in inputs.cells:
                        image_units = [receipt(root, 16, unit["unit"], binding) for unit in plan
                                       if unit["cell"] == cell["cell"] and unit["kind"] == "image"]
                        current = core.registration_summary(image_units, cell["domain"], 16)
                        original = inputs.original_ranks[cell["cell"]]
                        passed = all(current.get(k) == original.get(k) for k in ["status", "planned", "spectral_evaluable", "rank_low", "rank_high"])
                        rank_checks.append({"cell": cell["cell"], "passed": passed, "current": current, "original": original})
                        anchor_passed = anchor_passed and passed
                    check_path = root / ("registration_anchor_check-" + uuid.uuid4().hex + ".json")
                    write_json(check_path, {"checks": rank_checks, "passed": all(x["passed"] for x in rank_checks)})
                    directory, coverage = reports(root, plan, binding, core, anchor_passed, inputs)
                    if not anchor_passed:
                        print("STOP: dimension16 anchor mismatch; dimension8 not executed. " + str(directory), flush=True)
                        return 2
            directory, coverage = reports(root, plan, binding, core, anchor_passed, inputs)
            print("RESULTS " + str(directory), flush=True)
            return 0 if coverage["status"] == "completed" else 3


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["preflight", "run"]:
        p = sub.add_parser(name)
        p.add_argument("--repo", required=True, help="Original repository containing the two immutable archives")
        p.add_argument("--output", required=True, help="New private derived-output directory, outside the code package")
        if name == "run":
            p.add_argument("--execute-offline-analysis", action="store_true")
            p.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    protocol, historical, source = source_contract()
    if args.command == "run":
        return run_analysis(args, protocol, historical, source)
    root = output_path(args.output, args.repo)
    with Study2Inputs(args.repo, historical) as inputs:
        metadata = inputs.preflight()
    fresh(root)
    write_json(root / "preflight.json", dict(metadata, code_source=source, protocol=protocol, checked_at=now()))
    print("PREPARED_METADATA_ONLY: 48 banks and 2952 original image receipts verified; no numeric analysis")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (InputError, OSError, ValueError, KeyError) as exc:
        print("STOP: " + type(exc).__name__ + ": " + str(exc), file=sys.stderr)
        raise SystemExit(2)
