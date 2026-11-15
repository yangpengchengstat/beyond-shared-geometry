"""Drop the trivial block of the original whitened Study 2 orbits."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__import__('os').environ['BSG_CENTERING_OUTPUT'])
SNAPSHOT = ROOT / "studies/spectral_dimension_sensitivity_v1"
ARCHIVED = ROOT / "results/template_centering_v1/rewhitened"
sys.path.insert(0, str(SNAPSHOT))
from retained_inputs import Study2Inputs
from core import spectral_dimension_core as core

Q = 16
FIXED = "within_template_centered_fixed_whitening"
FORMS = ("original", FIXED, "within_template_centered")
TRIVIAL = {"C7": "trivial", "D4": "chi_pp"}
ATOL, RTOL = core.spectral.SCALAR_ATOL, core.spectral.SCALAR_RTOL

def close(a, b):
    return np.allclose(a, b, rtol=RTOL, atol=ATOL)

def archived_csv(name, receipt):
    raw = (ARCHIVED / name).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == receipt["archived_files_sha256"][name]
    return list(csv.DictReader(raw.decode().splitlines()))

def csv_write(name, rows):
    with (OUT/name).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def integer_spectrum(types):
    return [(t["real_dimension"], np.array(t["multiplicity_eigenvalues"], dtype=float)) for t in types]

def orbit_gram(x, spec):
    actions = np.asarray(spec["actions"], dtype=int)
    v = np.stack([x[:, a, :] for a in actions], axis=2).reshape(-1, len(actions)*Q) / np.sqrt(len(actions))
    return v.T @ v / len(v)

def fixed_image(xc, spec, original, bases, residual_trace, check_projector=False, x=None):
    types = original["types"]
    idx = next(i for i,t in enumerate(types) if t["name"] == TRIVIAL[spec["domain"]])
    assert all(t["name"] == expected["name"] for t, expected in zip(types, spec["types"]))
    assert close(Q*(1-types[idx]["e"]), residual_trace)
    before = core.spectral.integer_budget(integer_spectrum(types), Q)["value"]
    assert close(before, original["metrics"]["C_r"])
    kept_types = [t for j,t in enumerate(types) if j != idx]
    spectra = integer_spectrum(kept_types)
    budget = core.spectral.integer_budget(spectra, Q)
    # Independent numerical check: rebuild the Gram matrix from centered rows
    # in the original metric. No call here recomputes or applies whitening.
    kc = orbit_gram(xc, spec)
    assert close(np.trace(kc), residual_trace)
    if check_projector:
        B0 = bases[idx]
        P = np.eye(kc.shape[0]) - B0 @ B0.T
        assert close(kc, P @ orbit_gram(x, spec) @ P)
        assert np.linalg.norm(B0.T @ kc @ B0) <= 1e-8
    direct_spectra = []
    max_spectral_residual = 0.0
    for j,t in enumerate(types):
        if j == idx:
            continue
        B = bases[j]
        d = t["real_dimension"]
        restricted = B.T @ kc @ B
        eigenvalues = np.linalg.eigvalsh((restricted + restricted.T)/2)[::-1].reshape(-1,d)
        assert close(eigenvalues.max(axis=1), eigenvalues.min(axis=1))
        vals = eigenvalues.mean(axis=1)
        old = np.asarray(t["multiplicity_eigenvalues"])
        assert close(vals, old)
        max_spectral_residual = max(max_spectral_residual, float(np.max(np.abs(vals - old))))
        direct_spectra.append((d, vals))
    direct = core.spectral.integer_budget(direct_spectra,Q)["value"]
    value = budget["value"]
    assert close(value, direct)
    assert value >= -ATOL and value <= residual_trace + ATOL
    return value, budget["used_dimension"], max_spectral_residual

def distribution(values):
    return {"mean":float(np.mean(values)), "median":float(np.median(values)),
            "min":float(np.min(values)), "max":float(np.max(values))}

def comparable_cell(row):
    original = row["form"] == "original"
    return {
        "cell":row["cell"], "domain":row["domain"], "model":row["model"], "layer":int(row["layer"]),
        "form":row["form"], "C16":float(row["semantic_C_r"]), "budget_over_16":float(row["semantic_gamma"]),
        "fraction_of_own_total_energy":float(row["semantic_gamma"]),
        "total_energy_in_metric":16.0, "e_triv":float(row["e_triv"]), "b_triv":float(row["b_triv"]),
        "rank_low":int(row["rank_low"]), "rank_high":int(row["rank_high"]), "classification":row["classification"],
        "d_top_over_16":float(row["d_top"]), "image_count":int(row["evaluable_images"]),
        "original_whitening_residual":float(row["whitening_residual"]) if original else None,
        "second_moment_difference_from_I":None, "selected_dimension":None, "max_crosscheck_eigenvalue_residual":None,
    }

assert np.__version__ == "2.2.6"
historical = json.loads((Path(__file__).resolve().parent/"historical_protocol.json").read_text())
kernel = core.verify_kernel_source()
receipt = json.loads((ARCHIVED/"archive_record.json").read_text())
old_cells = archived_csv("cell_results.csv", receipt)
old_images = archived_csv("image_results.csv", receipt)
assert len(old_cells)==96 and len(old_images)==5904
assert all(r["status"]=="complete_descriptive" for r in old_cells)
archived_images = {(r["cell"],r["form"],r["image"]):r for r in old_images}
cell_rows = [comparable_cell(r) for r in old_cells]
image_rows = [{"cell":r["cell"],"domain":r["domain"],"form":r["form"],"image":r["image"],
               "prescribed":r["prescribed"]=="True","C16":Q*float(r["gamma"]),
               "budget_over_16":float(r["gamma"]),"fraction_of_own_total_energy":float(r["gamma"])}
              for r in old_images]
original_rows = {r["cell"]:r for r in cell_rows if r["form"]=="original"}
print("Authenticating Study 2 archives and prior diagnosis.", flush=True)
with Study2Inputs(ROOT, historical) as inputs:
    metadata = inputs.preflight()
    assert metadata["archive_sha256"] == receipt["input_archives_sha256"]
    assert set(inputs.by_cell) == set(original_rows)
    for domain in TRIVIAL:
        assert core.subgroup_specs(domain) == inputs.alt_plan["subgroups"][domain]
    print("All 48 input banks and 2,952 original image records authenticated.", flush=True)
    for c in inputs.cells:
        key, domain = c["cell"], c["domain"]
        z = inputs.load_test_bank(key)
        original, arrays = core.analyze_semantic(z, domain, Q)
        assert core.validate_anchor(original, inputs.semantic_baselines[key])["passed"]
        W = arrays["whitening"]
        x = z @ W
        xc = x - x.mean(axis=1,keepdims=True)
        assert close(xc,(z-z.mean(axis=1,keepdims=True))@W)
        assert np.max(np.abs(xc.mean(axis=1))) <= 1e-8
        residual_trace = float(np.mean(np.sum(xc**2,axis=-1)))
        assert 0 < residual_trace < Q
        assert close(residual_trace, Q*(1-original["metrics"]["e_triv"]))
        flat = xc.reshape(-1,Q)
        sigma = flat.T @ flat / len(flat)
        original_whitening_residual = original["test_bank_spectral"]["checks"]["whitening_identity"]["residual"]
        bases = []
        for j,t in enumerate(original["test_bank_spectral"]["types"]):
            w,v = np.linalg.eigh(arrays[f"type_{j}_projector"])
            bases.append(v[:,w>0.5])
        units, image_info = [], {}
        for entry in core.subgroup_specs(domain):
            iid = entry["id"]
            historical_image = inputs.baselines[key+"--"+iid]
            assert close(historical_image["metrics"]["gamma"],float(archived_images[(key,"original",iid)]["gamma"]))
            value, dim, residual = fixed_image(
                xc, entry["spec"], historical_image, bases, residual_trace, entry["semantic"], x)
            score = value/Q
            units.append({"status":"complete","domain":domain,"dimension":Q,
                          "subgroup_id":iid,"metrics":{"gamma":score}})
            image_info[iid] = (value,dim,residual)
            image_rows.append({"cell":key,"domain":domain,"form":FIXED,"image":iid,"prescribed":entry["semantic"],
                               "C16":value,"budget_over_16":score,"fraction_of_own_total_energy":value/residual_trace})
        registration = core.registration_summary(units,domain,Q)
        assert registration["status"]=="complete_descriptive"
        value, dim, _ = image_info[registration["semantic_image_id"]]
        canonical_value, _, _ = fixed_image(
            xc, core.spectral.group_spec(domain), inputs.semantic_baselines[key], bases, residual_trace, True, x)
        assert close(canonical_value, value)
        cell_rows.append({
            "cell":key,"domain":domain,"model":c["model_id"],"layer":c["layer"],
            "form":FIXED,"C16":value,"budget_over_16":value/Q,
            "fraction_of_own_total_energy":value/residual_trace,
            "total_energy_in_metric":residual_trace,"e_triv":0.0,"b_triv":0.0,
            "rank_low":registration["rank_low"],"rank_high":registration["rank_high"],
            "classification":registration["classification"],"d_top_over_16":registration["d_top"],
            "image_count":registration["spectral_evaluable"],
            "original_whitening_residual":original_whitening_residual,
            "second_moment_difference_from_I":float(np.linalg.norm(sigma-np.eye(Q),"fro")/np.sqrt(Q)),
            "selected_dimension":dim,
            "max_crosscheck_eigenvalue_residual":max(v[2] for v in image_info.values()),
        })
    print("Completed fixed-metric spectral ranks and direct residual-Gram cross-checks.",flush=True)
assert len(cell_rows)==144 and len(image_rows)==8856
by=defaultdict(list)
for r in cell_rows:
    by[r["domain"],r["form"]].append(r)
results={}
for domain in TRIVIAL:
    results[domain]={}
    for form in FORMS:
        rows=by[domain,form]
        assert len(rows)==24
        results[domain][form]={
            "rank_classes":dict(Counter(r["classification"] for r in rows)),
            "rank_intervals":dict(Counter(f'{r["rank_low"]}–{r["rank_high"]}' for r in rows)),
            **{k:distribution([r[k] for r in rows]) for k in
               ("C16","budget_over_16","fraction_of_own_total_energy","total_energy_in_metric","d_top_over_16")},
        }
    for left,right in zip(FORMS,FORMS[1:]):
        l={r["cell"]:r for r in by[domain,left]}
        r={r["cell"]:r for r in by[domain,right]}
        results[domain][left+" -> "+right]={
            "rank_intervals_changed":sum((l[k]["rank_low"],l[k]["rank_high"])!=(r[k]["rank_low"],r[k]["rank_high"]) for k in l),
            "leading_to_not_leading":sum(l[k]["classification"]=="unique_leading" and r[k]["classification"]!="unique_leading" for k in l),
            "not_leading_to_leading":sum(l[k]["classification"]!="unique_leading" and r[k]["classification"]=="unique_leading" for k in l),
        }
csv_write("three_way_cell_results.csv",cell_rows)
csv_write("three_way_image_results.csv",image_rows)
summary={
    "scope":"Remove each test template's state mean after applying its original bank whitening. No rewhitening, PCA refit, re-extraction, or Lstar bounds for the fixed-metric residual.",
    "normalization":"Ranking and d_top use C16/16 for every form. For the fixed-metric residual this is not a fraction of the residual energy. Residual-energy shares are recorded separately.",
    "rank_pairwise_atol":ATOL, "rank_pairwise_rtol":RTOL, "input_archives_sha256":metadata["archive_sha256"],
    "archived_comparison_file_sha256":receipt["archived_files_sha256"],"kernel":kernel,"numpy":np.__version__,
    "fixed_metric_images":2952, "fixed_metric_banks":48,
    "max_crosscheck_eigenvalue_residual":max(r["max_crosscheck_eigenvalue_residual"] for r in cell_rows if r["form"]==FIXED),
    "domains":results,
}
with (OUT/"summary.json").open("x") as stream:
    json.dump(summary,stream,indent=2,allow_nan=False)
    stream.write("\n")
print(json.dumps(summary,indent=2,allow_nan=False))
