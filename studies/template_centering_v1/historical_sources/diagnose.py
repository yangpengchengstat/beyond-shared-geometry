"""Study 2 q=16 within-template diagnostic using archived test coordinates."""
from collections import Counter
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path.cwd()
OUT = Path(__file__).resolve().parent
SNAPSHOT = ROOT / "dimension_rank_revision_20261105/offline_dimension_execution_v1/analysis/source_snapshot"
sys.path.insert(0, str(SNAPSHOT))
from archive_inputs import Study2Inputs
from core import spectral_dimension_core as core

Q = 16
FORMS = ("original", "within_template_centered")
MATRIX_TOL = core.spectral.MATRIX_TOL
RANK_TOL = core.spectral.RANK_RATIO

def dump_json(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")

def dump_csv(name, rows):
    with (OUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def stats(values):
    return {key: float(fn(values)) for key, fn in
            (("min", np.min), ("median", np.median), ("mean", np.mean), ("max", np.max))}

def matrix_checks(z):
    flat = z.reshape(-1, Q)
    sigma = flat.T @ flat / len(flat)
    w = np.linalg.eigvalsh(sigma)
    ratio = float(w[0] / w[-1]) if w[-1] > 0 else 0.0
    rank = int(np.linalg.matrix_rank(flat))
    return {
        "numerical_rank": rank, "sigma_min": float(w[0]), "sigma_max": float(w[-1]),
        "sigma_min_max_ratio": ratio, "rank_gate_passed": bool(rank == Q and ratio > RANK_TOL),
    }

def evaluate(inputs, bank, cell, form, image_rows, anchor_errors):
    domain, key = cell["domain"], cell["cell"]
    row = {"cell": key, "domain": domain, "model": cell["model_id"], "layer": cell["layer"],
           "form": form, "blocks": int(bank.shape[0]), "states": int(bank.shape[1]),
           **matrix_checks(bank), "whitening_residual": None, "whitening_passed": False,
           "max_abs_whitened_template_mean": None, "semantic_gamma": None,
           "semantic_C_r": None, "e_triv": None, "b_triv": None,
           "L_lower": None, "L_upper": None, "eta": None, "L_identity": None,
           "status": "not_evaluable_rank", "planned_images": core.IMAGE_COUNTS[domain],
           "evaluable_images": 0, "rank_low": None, "rank_high": None, "classification": None,
           "d_top": None, "d_median": None, "image_budget_spread": None}
    if not row["rank_gate_passed"]:
        return row
    sem, arrays = core.analyze_semantic(bank, domain, Q)
    if sem["status"] != "complete":
        row["status"] = sem["status"] + ":" + sem.get("reason", "")
        return row
    summary = sem["test_bank_spectral"]
    residual = float(summary["checks"]["whitening_identity"]["residual"])
    row.update(whitening_residual=residual, whitening_passed=bool(residual <= MATRIX_TOL))
    if form == "original":
        anchor_errors.append(core.validate_anchor(sem, inputs.semantic_baselines[key])["passed"])
    else:
        mean = float(np.max(np.abs((bank @ arrays["whitening"]).mean(axis=1))))
        row["max_abs_whitened_template_mean"] = mean
        assert mean <= MATRIX_TOL and abs(sem["metrics"]["e_triv"]) <= MATRIX_TOL
        assert abs(sem["metrics"]["L_identity"] - 2) <= MATRIX_TOL
    units = []
    for spec in core.subgroup_specs(domain):
        image_id = spec["id"]
        unit, _ = core.analyze_image(bank, domain, Q, image_id)
        units.append(unit)
        gamma = unit["metrics"]["gamma"] if unit["status"] == "complete" else None
        image_rows.append({"cell": key, "domain": domain, "form": form, "image": image_id,
                           "prescribed": spec["semantic"], "status": unit["status"], "gamma": gamma})
        if form == "original":
            anchor_errors.append(core.validate_anchor(unit, inputs.baselines[key + "--" + image_id])["passed"])
        elif unit["status"] == "complete":
            assert abs(unit["metrics"]["e_triv"]) <= MATRIX_TOL
    registration = core.registration_summary(units, domain, Q)
    if form == "original":
        old = inputs.original_ranks[key]
        rank_keys = ["status", "planned", "spectral_evaluable", "rank_low", "rank_high"]
        anchor_errors.append(all(registration[k] == old[k] for k in rank_keys))
    row.update(status=registration["status"], evaluable_images=registration["spectral_evaluable"],
               semantic_gamma=sem["metrics"]["gamma"], semantic_C_r=sem["metrics"]["C_r"],
               **{k: sem["metrics"][k] for k in ("e_triv", "b_triv", "L_lower", "L_upper", "eta", "L_identity")},
               **{k: registration.get(k) for k in ("rank_low", "rank_high", "classification", "d_top", "d_median")},
               image_budget_spread=registration.get("full_image_gamma_spread"))
    if row["status"] == "complete_descriptive":
        assert abs(row["semantic_gamma"] - registration["semantic_gamma"]) <= core._tol(row["semantic_gamma"], registration["semantic_gamma"])
    return row

def summarize(rows):
    output = {}
    for domain in ("C7", "D4"):
        domain_rows = [r for r in rows if r["domain"] == domain]
        output[domain] = {}
        for form in FORMS:
            planned = [r for r in domain_rows if r["form"] == form]
            valid = [r for r in planned if r["status"] == "complete_descriptive"]
            output[domain][form] = {
                "planned": len(planned), "complete": len(valid),
                "rank_full": sum(r["rank_gate_passed"] for r in planned),
                "whitening_passed": sum(r["whitening_passed"] for r in planned),
                "ranks": dict(Counter(r["classification"] for r in valid)),
                "stats": {k: stats([r[k] for r in valid]) for k in
                          ("semantic_gamma", "semantic_C_r",
                           "sigma_min_max_ratio", "whitening_residual", "e_triv", "b_triv",
                           "L_lower", "L_upper", "d_top", "image_budget_spread",
                           "rank_low", "rank_high")} if valid else {},
            }
        indexed = {(r["cell"], r["form"]): r for r in domain_rows if r["status"] == "complete_descriptive"}
        pairs = [(r, indexed[(key, FORMS[1])]) for (key, form), r in indexed.items()
                 if form == FORMS[0] and (key, FORMS[1]) in indexed]
        output[domain]["paired"] = {
            "conditions": len(pairs),
            "rank_interval_changed": sum((a["rank_low"],a["rank_high"]) != (b["rank_low"],b["rank_high"]) for a,b in pairs),
            "definite_rank_improvement": sum(b["rank_high"] < a["rank_low"] for a,b in pairs),
            "definite_rank_worsening": sum(b["rank_low"] > a["rank_high"] for a,b in pairs),
            "gamma_change": stats([b["semantic_gamma"]-a["semantic_gamma"] for a,b in pairs]) if pairs else {},
        }
    return output

assert np.__version__ == "2.2.6"
historical = json.loads((SNAPSHOT / "historical_protocol.json").read_text())
kernel = core.verify_kernel_source()
print("Authenticating the original archives, row bindings, and historical results.", flush=True)
all_rows, image_rows, anchors, banks = [], [], [], {}
with Study2Inputs(ROOT, historical) as inputs:
    metadata = inputs.preflight()
    for domain in ("C7", "D4"):
        assert core.subgroup_specs(domain) == inputs.alt_plan["subgroups"][domain]
    for cell in inputs.cells:
        key = cell["cell"]
        banks[key] = inputs.load_test_bank(key)
        all_rows.append(evaluate(inputs, banks[key], cell, FORMS[0], image_rows, anchors))
    assert len(anchors) == 3048 and all(anchors), (len(anchors), sum(anchors))
    print("Original banks matched: 48 canonical, 2,952 image budgets, 48 ranks.", flush=True)
    for cell in inputs.cells:
        z = banks[cell["cell"]]
        centered = z - z.mean(axis=1, keepdims=True)
        assert np.max(np.abs(centered.mean(axis=1))) <= MATRIX_TOL
        all_rows.append(evaluate(inputs, centered, cell, FORMS[1], image_rows, anchors))
    print("Completed within-template diagnostics.", flush=True)
dump_csv("cell_results.csv", all_rows)
dump_csv("image_results.csv", image_rows)
report = {"dimension": Q, "rank_eigenvalue_ratio_threshold": RANK_TOL,
          "whitening_relative_Frobenius_tolerance": MATRIX_TOL,
          "rank_pairwise_atol": core.spectral.SCALAR_ATOL, "rank_pairwise_rtol": core.spectral.SCALAR_RTOL,
          "outer_centering": "Each complete test-template variant, separately, subtracts its mean across all state rows.",
          "whitening": "Separate empirical second moment for each form, in the same saved 16 PCA coordinates.",
          "scope": "Descriptive finite-bank spectral-budget sensitivity, no re-extraction or PCA refit.",
          "archive_sha256": metadata["archive_sha256"], "kernel": kernel, "numpy": np.__version__,
          "baseline_checks_passed": sum(anchors), "domains": summarize(all_rows)}
dump_json("summary.json", report)
print(json.dumps(report, indent=2, allow_nan=False))
