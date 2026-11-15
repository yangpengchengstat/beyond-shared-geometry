"""Summarize existing committed results only; no analysis/refit/resampling.

Run: python3 summarize_spectral_extensions.py ARCHIVE [OUTPUT_DIR]
This reads ZIP members directly, verifies stage receipts and committed unit
bindings/output hashes, checks aggregate values against committed summaries,
and writes compact tables plus provenance. It never invokes study numerics.
"""
import csv
import hashlib
import io
import json
from pathlib import Path
import statistics
import sys
import zipfile
from collections import Counter, defaultdict


ARCHIVE = Path(sys.argv[1])
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).parent / "spectral_evidence"
OUT.mkdir(parents=True, exist_ok=True)
SPEC = "spectral-results-436df4be8558423f8e84aea0e4632408"


def stat(vals):
    vals = list(vals)
    return dict(n=len(vals), min=min(vals), median=statistics.median(vals),
                mean=statistics.mean(vals), max=max(vals))


def write_csv(name, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def tol(a, b):
    return 1e-8 + 1e-7 * max(abs(a), abs(b))


def rank(vals, semantic, descending=True):
    better = sum((v > semantic + tol(v, semantic)) if descending else
                 (v < semantic - tol(v, semantic)) for v in vals)
    tied = sum(abs(v - semantic) <= tol(v, semantic) for v in vals)
    return better + 1, better + tied


with zipfile.ZipFile(ARCHIVE) as z:
    names = z.namelist()
    assert len(names) == len(set(names)), "duplicate ZIP members"
    digest_cache = {}
    sources = {}

    def read(path):
        b = z.read(path)
        digest_cache[path] = hashlib.sha256(b).hexdigest()
        return b

    def js(path):
        return json.loads(read(path))

    def digest(path):
        if path not in digest_cache:
            h = hashlib.sha256()
            with z.open(path) as f:
                for b in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(b)
            digest_cache[path] = h.hexdigest()
        return digest_cache[path]

    def sourced(path):
        b = read(path)
        sources[path] = digest_cache[path]
        return b

    def csvread(path):
        return list(csv.DictReader(io.StringIO(sourced(path).decode())))

    stage_verification = {}
    # Stage receipts include saved reports, inputs, source snapshots and every
    # unit output. Verify without extracting the large per-unit diagnostics.
    for stage, root in [("spectral_analyze", SPEC), ("witness", "witness"),
                        ("simulate", "simulate"), ("alternatives", "alternatives")]:
        receipt_path = next(n for n in names if n.startswith("receipts/" + stage + "-"))
        receipt = json.loads(sourced(receipt_path))
        for rel, expected in receipt["files"].items():
            assert digest(root + "/" + rel) == expected, (stage, rel)
        stage_verification[stage] = dict(receipt=receipt_path, receipt_sha256=digest(receipt_path),
                                        verified_files=len(receipt["files"]),
                                        finished=receipt.get("finished"))

    spectral = csvread(SPEC + "/cell_summary.csv")
    spectral_types = csvread(SPEC + "/type_spectra.csv")
    (OUT / "original_spectral_cell_summary.csv").write_bytes(read(SPEC + "/cell_summary.csv"))
    (OUT / "original_spectral_type_spectra.csv").write_bytes(read(SPEC + "/type_spectra.csv"))
    assert len(spectral) == 48 and all(r["status"] == "complete" for r in spectral)
    core_checks = Counter()
    core_max_residuals = defaultdict(float)
    core_cells = []
    for row in spectral:
        path = SPEC + "/cells/" + row["cell"] + "/summary.json"
        s = js(path)
        for key, value in s["metrics"].items():
            assert float(row[key]) == value
        for key, check in s["checks"].items():
            core_checks["passed" if check["passed"] else "failed"] += 1
            if "residual" in check:
                core_max_residuals[key] = max(core_max_residuals[key], check["residual"])
        t = dict(cell=row["cell"], domain=row["domain"], model_id=row["model_id"], layer=int(row["layer"]),
                 **s["metrics"], sigma_condition_number=s["diagnostics"]["sigma_condition_number"],
                 used_dimension=s["budget"]["used_dimension"],
                 budget_tie_within_tolerance=s["budget"].get("tie_within_tolerance"),
                 source_summary=path, source_summary_sha256=digest(path))
        for typ in s["types"]:
            for field in ["e", "b", "selected_copies", "selected_dimension"]:
                t[typ["name"] + "_" + field] = typ[field]
                if typ["name"] == "chi_pp":
                    t["trivial_" + field] = typ[field]
        core_cells.append(t)
    write_csv("spectral_cells.csv", core_cells)

    stage_rows = {}
    unit_verification = {}
    unit_provenance = []
    endpoint_checks = {}
    for stage in ["witness", "simulate", "alternatives"]:
        state = json.loads(sourced(stage + "/state.json"))
        plan = js(stage + "/plan.json")
        manifest = js(stage + "/run_manifest.json")
        binding = dict(plan_sha256=digest(stage + "/plan.json"),
                       run_manifest_sha256=digest(stage + "/run_manifest.json"))
        report_path = stage + "/" + state["report"] + "/unit_summary.csv"
        report = csvread(report_path)
        (OUT / ("original_" + stage + "_unit_summary.csv")).write_bytes(read(report_path))
        by_unit = {r["unit"]: r for r in report}
        assert len(by_unit) == len(report) == len(plan["units"])
        rows = []
        counts, checks = Counter(), Counter()
        max_residuals = defaultdict(float)
        committed_attempts = set()
        for unit in plan["units"]:
            uid = unit["unit"]
            receipt_path = stage + "/units/" + uid + "/receipt.json"
            receipt = js(receipt_path)
            assert receipt["binding"] == binding and receipt["unit"] == uid and receipt["terminal"]
            assert receipt["attempt"].startswith("attempts/" + uid + "/")
            attempt = stage + "/" + receipt["attempt"]
            committed_attempts.add(attempt)
            for rel, expected in receipt["files"].items():
                assert digest(attempt + "/" + rel) == expected
            s = js(attempt + "/summary.json")
            assert receipt["status"] == s["status"] == by_unit[uid]["status"]
            counts[s["status"]] += 1
            row = {k: s[k] for k in ["unit", "cell", "domain", "mechanism", "noise", "seed", "subgroup_id",
                                    "semantic_registration", "status", "witness_status", "test_spectral_status"] if k in s}
            row.update(s.get("metrics", {}))
            if "test_bank_spectral" in s:
                row["test_bank_gamma"] = s["test_bank_spectral"].get("metrics", {}).get("gamma")
            for key, value in row.items():
                if key in by_unit[uid] and isinstance(value, (int, float)) and not isinstance(value, bool):
                    assert float(by_unit[uid][key]) == value, (uid, key)
            if stage == "simulate":
                row["shared_action_before_noise"] = s["construction"]["shared_action_before_noise"]
                row["mechanism_is_valid_negative_null"] = s["construction"]["mechanism_is_valid_negative_null"]
            witness = s.get("witness", s)
            row["train_bound_passed"] = witness.get("checks", {}).get("witness_training_bound", {}).get("passed")
            row["witness_minus_ols"] = row["test_witness_risk"] - row["test_ols_risk"]
            row["witness_minus_ridge"] = row["test_witness_risk"] - row["test_ridge_risk"]
            row["test_minus_train_witness"] = row["test_witness_risk"] - row["train_witness_risk"]
            row["train_witness_minus_lower"] = row["train_witness_risk"] - row["train_L_lower"]
            row["upper_minus_train_witness"] = row["train_L_upper"] - row["train_witness_risk"]
            row["source_summary"] = attempt + "/summary.json"
            row["source_summary_sha256"] = digest(attempt + "/summary.json")
            for component, component_data in [("witness", witness), ("train_spectral", witness.get("train_spectral", {})),
                                               ("test_spectral", s.get("test_bank_spectral", {}))]:
                for key, check in component_data.get("checks", {}).items():
                    checks[component + ("_passed" if check["passed"] else "_failed")] += 1
                    if check.get("residual") is not None:
                        max_residuals[component + "/" + key] = max(max_residuals[component + "/" + key], check["residual"])
            rows.append(row)
            unit_provenance.append(dict(stage=stage, unit=uid, receipt=receipt_path,
                                        receipt_sha256=digest(receipt_path), summary=attempt + "/summary.json",
                                        summary_sha256=digest(attempt + "/summary.json")))
        attempts = {n.rsplit("/", 1)[0] for n in names if n.startswith(stage + "/attempts/") and n.endswith("/started.json")}
        unit_verification[stage] = dict(planned=len(plan["units"]), committed=len(rows),
                                       statuses=dict(counts), uncommitted_attempts=sorted(attempts - committed_attempts),
                                       not_evaluable_units=sum(v for k, v in counts.items() if k != "complete"),
                                       state=state, report=report_path, binding=binding)
        endpoint_checks[stage] = dict(counts=dict(checks), max_residuals=dict(max_residuals))
        stage_rows[stage] = rows
        write_csv(stage + "_units.csv", rows)

    # Explicit equality cross-check: semantic alternative is the same endpoint.
    wmap = {r["cell"]: r for r in stage_rows["witness"]}
    smap = {r["cell"]: r for r in core_cells}
    alternatives = defaultdict(list)
    for row in stage_rows["alternatives"]:
        alternatives[row["cell"]].append(row)
        if row["semantic_registration"]:
            for key in ["train_gamma", "train_witness_risk", "test_witness_risk", "test_ols_risk", "test_ridge_risk"]:
                assert abs(row[key] - wmap[row["cell"]][key]) < 1e-12
            assert abs(row["test_bank_gamma"] - smap[row["cell"]]["gamma"]) < 1e-12
    rank_path = "alternatives/" + unit_verification["alternatives"]["state"]["report"] + "/descriptive_ranks.json"
    ranks = json.loads(sourced(rank_path))["cells"]
    (OUT / "original_descriptive_ranks.json").write_bytes(read(rank_path))
    alternative_cells = []
    for cell, rows in sorted(alternatives.items()):
        sem = next(r for r in rows if r["semantic_registration"])
        gs = [r["test_bank_gamma"] for r in rows]
        lo, hi = rank(gs, sem["test_bank_gamma"])
        assert (lo, hi) == (ranks[cell]["rank_low"], ranks[cell]["rank_high"])
        ws = [r["test_witness_risk"] for r in rows]
        wlo, whi = rank(ws, sem["test_witness_risk"], False)
        alternative_cells.append(dict(cell=cell, domain=sem["domain"], planned=len(rows),
            semantic_gamma=sem["test_bank_gamma"], rank_low=lo, rank_high=hi,
            gamma_min=min(gs), gamma_max=max(gs), gamma_spread=max(gs)-min(gs),
            best_gamma_minus_semantic=max(gs)-sem["test_bank_gamma"],
            semantic_witness_risk=sem["test_witness_risk"], witness_risk_rank_low=wlo,
            witness_risk_rank_high=whi, witness_risk_min=min(ws), witness_risk_max=max(ws)))
    write_csv("alternative_cells.csv", alternative_cells)

    metrics = dict(aggregation="unweighted descriptive summaries across prespecified cells; no sampling inference",
                   spectral={}, witness={}, simulations={}, alternatives={})
    for domain in ["C7", "D4"]:
        ss = [r for r in core_cells if r["domain"] == domain]
        metrics["spectral"][domain] = {k: stat(r[k] for r in ss) for k in
            ["gamma", "L_lower", "L_upper", "eta_LS", "delta_type", "delta_type_normalized", "sigma_condition_number", "trivial_e", "trivial_b"]}
        metrics["spectral"][domain]["eta_identity_max_abs_difference"] = max(abs(r["eta_spec"]-r["eta_LS"]) for r in ss)
        metrics["spectral"][domain]["selected_dimensions"] = dict(Counter(r["used_dimension"] for r in ss))
        metrics["spectral"][domain]["resolved_type_corrections"] = [{"cell":r["cell"],"delta_type":r["delta_type"],"delta_type_normalized":r["delta_type_normalized"]} for r in ss if abs(r["delta_type"]) > tol(r["delta_type"], 0)]
        ws = [r for r in stage_rows["witness"] if r["domain"] == domain]
        metrics["witness"][domain] = {k: stat(r[k] for r in ws) for k in
            ["train_gamma", "train_witness_risk", "test_witness_risk", "test_ols_risk", "test_ridge_risk", "witness_minus_ols", "witness_minus_ridge", "test_minus_train_witness", "train_witness_minus_lower", "upper_minus_train_witness"]}
        metrics["witness"][domain]["witness_better_than_ols_beyond_tolerance"] = sum(r["witness_minus_ols"] < -tol(r["test_witness_risk"],r["test_ols_risk"]) for r in ws)
        metrics["witness"][domain]["witness_worse_than_ols_beyond_tolerance"] = sum(r["witness_minus_ols"] > tol(r["test_witness_risk"],r["test_ols_risk"]) for r in ws)
        aa = [r for r in alternative_cells if r["domain"] == domain]
        metrics["alternatives"][domain] = {k: stat(r[k] for r in aa) for k in
            ["rank_low", "rank_high", "gamma_spread", "best_gamma_minus_semantic"]}
        metrics["alternatives"][domain].update(
            exact_rank_counts=dict(Counter(f'{r["rank_low"]}-{r["rank_high"]}' for r in aa)),
            semantic_top_including_ties=sum(r["rank_low"] == 1 for r in aa),
            semantic_unique_top=sum(r["rank_low"] == r["rank_high"] == 1 for r in aa),
            semantic_bottom_including_ties=sum(r["rank_high"] == r["planned"] for r in aa),
            all_subgroups_tied=sum(r["rank_low"] == 1 and r["rank_high"] == r["planned"] for r in aa))
        for mechanism in ["M1", "M2", "M3"]:
            for noise in [0.0, 0.05]:
                rs = [r for r in stage_rows["simulate"] if r["domain"] == domain and r["mechanism"] == mechanism and r["noise"] == noise]
                metrics["simulations"][f"{domain}_{mechanism}_noise{noise:g}"] = {
                    k: stat(r[k] for r in rs) for k in ["train_gamma", "test_bank_gamma", "test_witness_risk", "test_ols_risk", "test_ridge_risk"]}
    write_csv("spectral_domain_summary.csv", [dict(domain=d, metric=k, **v) for d, vv in metrics["spectral"].items() for k,v in vv.items() if isinstance(v,dict) and "mean" in v])
    write_csv("witness_domain_summary.csv", [dict(domain=d, metric=k, **v) for d,vv in metrics["witness"].items() for k,v in vv.items() if isinstance(v,dict) and "mean" in v])
    write_csv("simulation_summary.csv", [dict(case=d, metric=k, **v) for d,vv in metrics["simulations"].items() for k,v in vv.items()])
    write_csv("committed_unit_provenance.csv", unit_provenance)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    provenance = dict(archive_name=ARCHIVE.name, archive_bytes=ARCHIVE.stat().st_size,
                      summarizer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      transformation="Existing-output extraction, checksum verification, descriptive mean/median/range and paired subtraction only; no fitting, experiment, resampling or changed thresholds.",
                      stage_verification=stage_verification, unit_verification=unit_verification,
                      aggregate_source_sha256=sources, core_checks=dict(core_checks),
                      core_max_residuals=dict(core_max_residuals), endpoint_checks=endpoint_checks,
                      semantic_equality_checks_passed=True,
                      verified_file_digest_count=len(digest_cache))
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    (OUT / "file_hashes.json").write_text(json.dumps({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(OUT.iterdir()) if p.is_file() and p.name != "file_hashes.json"}, indent=2) + "\n")
    print(json.dumps(dict(output=str(OUT), verified_files=len(digest_cache),
                         stages={k:{"planned":v["planned"],"committed":v["committed"],"statuses":v["statuses"],"uncommitted_attempts":len(v["uncommitted_attempts"])} for k,v in unit_verification.items()},
                         core_checks=dict(core_checks)), indent=2))
