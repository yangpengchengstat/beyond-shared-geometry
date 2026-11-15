"""Optional MAIN30-D CPU analysis. Importing this file never reads coordinates.

The default CLI only audits the retained input. Execution requires BOTH
config.full_run=true and --execute-family-sensitivity. No model dependencies.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import zipfile

# The optional full run is sequential; set limits before importing NumPy.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
CORE_PATH = PROJECT / "studies/spectral_finite_bank_v1/gamma_r.py"
EXT_PATH = PROJECT / "studies/spectral_extensions_v1/numerics.py"
_spec = importlib.util.spec_from_file_location("rank_sensitivity_frozen_core", CORE_PATH)
core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(core)
_spec = importlib.util.spec_from_file_location("rank_sensitivity_frozen_extension", EXT_PATH)
extension = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(extension)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def image_identity(subgroup):
    """Identity is the actual sorted permutation image, never its display index."""
    perms = sorted(subgroup["image_permutations"])
    if perms != sorted(subgroup["spec"]["actions"]):
        raise ValueError("Image canonical identity differs from action grid")
    return digest(canonical(perms))


def validate_contexts(contexts, domain, families):
    """Require exactly two complete, canonical state blocks per declared family."""
    p = {"C7": 7, "D4": 4}[domain]
    row_domain = {"C7": "C7", "D4": "D4_corner"}[domain]
    if len(families) != len(set(families)):
        raise ValueError("Duplicate declared family ID")
    pairs, rows = {}, set()
    for i, ctx in enumerate(contexts):
        family, variant = ctx["family_id"], ctx["paraphrase_id"]
        key = (family, variant)
        expected = [f"{row_domain}/{family}/p{variant}/lexical_{row_domain}/s{s}" for s in range(p)]
        if (key in pairs or type(variant) is not int or variant not in (0, 1)
                or ctx["split"] != "test" or ctx["orbit_id"] != "lexical_" + row_domain
                or ctx["state_ids"] != list(range(p)) or ctx["row_ids"] != expected
                or rows.intersection(expected)):
            raise ValueError("Invalid/duplicate family, variant or complete state identity")
        pairs[key] = i
        rows.update(expected)
    wanted = {(f, v) for f in families for v in (0, 1)}
    if set(pairs) != wanted:
        raise ValueError("Missing/unexpected family or paired variant")
    return np.array([[pairs[f, v] for v in (0, 1)] for f in families], dtype=int)


def delete_family(Z, contexts, domain, families, deleted_family):
    """Direct-coordinate oracle used by synthetic tests, retaining every state."""
    indices = validate_contexts(contexts, domain, families)
    if deleted_family not in families:
        raise ValueError("Deletion ID not declared")
    keep = indices[np.array(families) != deleted_family].ravel()
    return np.asarray(Z)[keep], [contexts[i] for i in keep]


def family_moments(Z, contexts, domain, families, rank=16):
    """F[f,s,t,a,b] = mean_variant z[f,v,s,a] z[f,v,t,b]."""
    indices = validate_contexts(contexts, domain, families)
    p = {"C7": 7, "D4": 4}[domain]
    raw = np.asarray(Z)
    if (raw.shape != (2 * len(families), p, rank) or raw.dtype.kind not in "fiu"
            or not np.isfinite(raw).all()):
        raise ValueError("Missing, nonfinite or wrong-shape original PCA coordinates")
    ordered = np.asarray(raw, dtype=np.float64)[indices]
    return np.einsum("fvsa,fvtb->fstab", ordered, ordered, optimize=True) / 2


def deleted_moment(contributions, family_index):
    if len(contributions) < 2 or not 0 <= family_index < len(contributions):
        raise ValueError("Invalid deletion")
    return (contributions.sum(axis=0) - contributions[family_index]) / (len(contributions) - 1)


def whiten_moment(T):
    """Fresh uncentered second moment; exactly the frozen strict rank rule."""
    p, _, r, _ = T.shape
    raw = np.einsum("ssab->ab", T) / p
    sigma = (raw + raw.T) / 2
    summary = {"status": "complete", "checks": {}, "diagnostics": {}}
    checks = core._Checks(summary["checks"])
    try:
        checks.condition("finite_second_moment", np.isfinite(raw).all())
        checks.matrix("second_moment_symmetry", raw, raw.T)
        w, U = np.linalg.eigh(sigma)
        checks.condition("finite_second_moment_eigensystem", np.isfinite(w).all() and np.isfinite(U).all())
        lo, hi = float(w[0]), float(w[-1])
        summary["diagnostics"] = {"sigma_eigenvalues": w.tolist(), "sigma_min": lo,
            "sigma_max": hi, "sigma_min_max_ratio": lo / hi if hi > 0 else None,
            "sigma_condition_number": hi / lo if lo > 0 and np.isfinite(hi / lo) else None}
        passed = hi > 0 and lo > core.RANK_RATIO * hi
        summary["checks"]["second_moment_rank"] = {"passed": bool(passed),
            "required_min_max_ratio_strictly_above": core.RANK_RATIO}
        if not passed:
            raise core.ValidationFailure("not_evaluable_rank", "empirical second moment fails fixed full-rank rule")
        W = (U * w ** -0.5) @ U.T
        checks.matrix("whitening_identity", W.T @ sigma @ W, np.eye(r))
        return summary, np.einsum("ai,stab,bj->stij", W, T, W, optimize=True)
    except core.ValidationFailure as exc:
        summary.update(status=exc.status, reason=exc.reason)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        summary.update(status="not_evaluable_numerics", reason=str(exc))
    return summary, None


def orbit_gram(Twhite, spec):
    """K[g,a,h,b] = mean_state Twhite[g.s,h.s,a,b] / |G|."""
    actions = np.asarray(spec["actions"], dtype=int)
    m, p, r = spec["m"], spec["p"], Twhite.shape[-1]
    K = np.empty((m, r, m, r), dtype=float)
    for g in range(m):
        for h in range(m):
            K[g, :, h, :] = Twhite[actions[g], actions[h]].sum(axis=0) / (m * p)
    return K.reshape(m * r, m * r)


def invariant_structure(spec, rank):
    """Cacheable group-only projectors, validated with frozen tolerances."""
    core.validate_group(spec)
    m, D = spec["m"], spec["m"] * rank
    table, inverse = np.array(spec["multiplication"]), np.array(spec["inverse"])
    indices = np.array([table[np.arange(m), inverse[t]] for t in range(m)])
    lambdas = np.array([np.kron(np.eye(m)[idx], np.eye(rank)) for idx in indices])
    checks = core._Checks({})
    for g in range(m):
        checks.condition(f"block_orthogonal_{g}", np.array_equal(np.sort(indices[g]), np.arange(m)))
        for h in range(m):
            checks.condition(f"block_law_{g}_{h}", np.array_equal(indices[h][indices[g]], indices[table[h, g]]))
    projectors, bases = [], []
    for j, t in enumerate(spec["types"]):
        raw = t["coefficient"] * np.einsum("g,gij->ij", t["character"], lambdas)
        checks.matrix(f"projector_{j}_symmetry", raw, raw.T)
        P = (raw + raw.T) / 2
        checks.matrix(f"projector_{j}_idempotence", P @ P, P)
        for g in range(m):
            checks.matrix(f"projector_{j}_commutation_{g}", P @ lambdas[g], lambdas[g] @ P)
        for k, prev in enumerate(projectors):
            checks.matrix(f"projector_orthogonality_{j}_{k}", P @ prev, np.zeros_like(P))
        values, vectors = np.linalg.eigh(P)
        for k, value in enumerate(values):
            checks.scalar(f"projector_{j}_eigenvalue_{k}", float(value), float(value > .5))
        B = vectors[:, values > .5]
        checks.condition(f"type_{j}_dimension", B.shape[1] > 0 and B.shape[1] % t["d"] == 0)
        projectors.append(P)
        bases.append(B)
    checks.matrix("projector_completeness", sum(projectors), np.eye(D))
    return lambdas, bases


def gamma_from_moment(Twhite, spec, structure=None):
    """Moment equivalent of the frozen core's K/type/budget path (no clipping)."""
    r = Twhite.shape[-1]
    out = {"status": "complete", "checks": {}, "diagnostics": {}}
    checks = core._Checks(out["checks"])
    try:
        lambdas, bases = structure if structure is not None else invariant_structure(spec, r)
        raw = orbit_gram(Twhite, spec)
        checks.condition("finite_K", np.isfinite(raw).all())
        checks.matrix("K_symmetry", raw, raw.T)
        K = (raw + raw.T) / 2
        eig = np.linalg.eigvalsh(K)[::-1]
        checks.condition("finite_K_spectrum", np.isfinite(eig).all())
        checks.interval("K_psd", eig[-1], 0, 1, tolerance=core.EIGEN_TOL)
        checks.interval("K_contraction", eig[0], 0, 1, tolerance=core.EIGEN_TOL)
        checks.scalar("K_trace", np.trace(K), r)
        out["diagnostics"] = {"K_min_eigenvalue": float(eig[-1]), "K_max_eigenvalue": float(eig[0]),
                              "trace_K": float(np.trace(K))}
        for g, lam in enumerate(lambdas):
            checks.matrix(f"K_commutation_{g}", K @ lam, lam @ K)
        types, energies = [], []
        for j, (t, B) in enumerate(zip(spec["types"], bases)):
            restricted = B.T @ K @ B
            checks.matrix(f"type_{j}_K_symmetry", restricted, restricted.T)
            restricted = (restricted + restricted.T) / 2
            values = np.linalg.eigvalsh(restricted)[::-1].reshape(-1, t["d"])
            for k, pair in enumerate(values):
                checks.scalar(f"type_{j}_pairing_{k}", pair.max(), pair.min())
            multiplicities = values.mean(axis=1)
            energy = float(np.trace(restricted) / r)
            checks.interval(f"type_{j}_energy", energy, 0, 1)
            if t.get("absent"):
                checks.scalar(f"type_{j}_analytic_absent_energy", energy, 0)
            types.append((t["d"], multiplicities))
            energies.append(energy)
        checks.scalar("type_energy_sum", sum(energies), 1)
        budget = core.integer_budget(types, r)
        best, counts = budget["value"], budget["selected_copies"]
        used = sum(d * k for (d, _), k in zip(types, counts))
        checks.condition("integer_budget_feasible", used <= r and used == budget["used_dimension"]
                         and all(0 <= k <= len(lam) for (_, lam), k in zip(types, counts)))
        checks.scalar("integer_budget_value", sum(float(d * sum(lam[:k])) for (d, lam), k in zip(types, counts)), best)
        gamma = best / r
        checks.interval("gamma_range", gamma, 0, 1)
        checks.interval("typed_budget_vs_untyped", float(sum(eig[:r])) - best, 0, r)
        checks.interval("analytic_type_correction_bound", float(sum(eig[:r])) - best, 0, 1)
        selected = [float(d * sum(lam[:k]) / r) for (d, lam), k in zip(types, counts)]
        for j, b in enumerate(selected):
            checks.interval(f"type_{j}_selected_energy", b, 0, energies[j])
        checks.scalar("selected_energy_sum", sum(selected), gamma)
        out.update(metrics={"gamma": gamma}, total_type_energies=energies, selected_type_energies=selected)
    except core.ValidationFailure as exc:
        out.update(status=exc.status, reason=exc.reason)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        out.update(status="not_evaluable_numerics", reason=str(exc))
    return out


def margins_and_ranks(summaries):
    """Call the original rank function, not a newly chosen tie convention."""
    out = extension.registration_ranks(summaries)
    out.update(d_top=None, d_median=None, spread=None, uniquely_leading=False)
    if out["status"] == "complete_descriptive":
        alternatives = [s["test_bank_spectral"]["metrics"]["gamma"] for s in summaries if not s["semantic_registration"]]
        values = [s["test_bank_spectral"]["metrics"]["gamma"] for s in summaries]
        out.update(d_top=out["semantic_gamma"] - max(alternatives),
                   d_median=out["semantic_gamma"] - float(np.median(alternatives)),
                   spread=max(values) - min(values), uniquely_leading=out["rank_high"] == 1)
    return out


def execute_cell(Z, contexts, domain, families, subgroups, rank=16):
    """Yield one terminal record per deletion, including every numerical failure."""
    contributions = family_moments(Z, contexts, domain, families, rank)
    total = contributions.sum(axis=0)
    structure = invariant_structure(subgroups[0]["spec"], rank)
    for index, family in enumerate(families):
        whitening, Twhite = whiten_moment((total - contributions[index]) / (len(families) - 1))
        summaries = []
        for subgroup in subgroups:
            result = gamma_from_moment(Twhite, subgroup["spec"], structure) if Twhite is not None else dict(whitening)
            summaries.append({"subgroup_id": subgroup["id"], "image_sha256": image_identity(subgroup),
                              "semantic_registration": subgroup["semantic"], "test_bank_spectral": result})
        yield {"deleted_family": family, "remaining_families": len(families) - 1,
               "remaining_contexts": 2 * (len(families) - 1), "whitening": whitening,
               "comparison": margins_and_ranks(summaries), "images": summaries}


def checked_read(z, name, expected):
    if name.startswith("/") or ".." in Path(name).parts or z.namelist().count(name) != 1:
        raise ValueError("Unsafe, missing or duplicated archive member")
    raw = z.read(name)
    if digest(raw) != expected:
        raise ValueError(f"Hash mismatch: {name}")
    return raw


def inspect_input(archive, config):
    """Read hashes, metadata and NPY headers; never load coordinate arrays."""
    if any(config.get(key) != value for key, value in {
            "workers": 1, "blas_threads": 1, "rank": 16,
            "family_count": 64, "paired_variants": 2,
            "population_inference": False, "cross_domain_pairing_claimed": False}.items()):
        raise ValueError("Configuration changes the prepared fixed protocol")
    for path, sha in config["source_hashes"].items():
        if digest((PROJECT / path).read_bytes()) != sha:
            raise ValueError(f"Frozen numerical source changed: {path}")
    with zipfile.ZipFile(archive) as z:
        snapshots = {name: json.loads(checked_read(z, name, sha)) for name, sha in config["metadata_hashes"].items()}
        inputs = snapshots["alternatives/input_snapshot/inputs.json"]
        manifest = snapshots["alternatives/input_snapshot/prepared_manifest.json"]
        plan = snapshots["alternatives/plan.json"]
        protocol = json.loads((PROJECT / "studies/spectral_finite_bank_v1/protocol.json").read_text())
        families = protocol["test_families"]
        if len(families) != 64 or protocol["rank"] != 16 or len(inputs["cells"]) != 48:
            raise ValueError("Frozen family/rank/cell panel differs")
        expected_cells = {(m["id"], layer, domain) for m in protocol["models"]
                          for layer in m["layers"] for domain in ("C7", "D4")}
        actual_cells = {(c["model_id"], c["layer"], c["domain"]) for c in inputs["cells"]}
        if actual_cells != expected_cells or len({c["cell"] for c in inputs["cells"]}) != 48:
            raise ValueError("Unexpected or duplicate cell identity")
        rows_by_cell, receipts = {}, []
        for domain, expected in (("C7", 120), ("D4", 3)):
            groups = plan["subgroups"][domain]
            base_image = sorted(core.group_spec(domain)["actions"])
            identities = [image_identity(g) for g in groups]
            if len(groups) != expected or len(set(identities)) != expected or sum(g["semantic"] for g in groups) != 1:
                raise ValueError("Full distinct image comparator set is required")
            for group in groups:
                core.validate_group(group["spec"])
                if bool(group["semantic"]) != (sorted(group["image_permutations"]) == base_image):
                    raise ValueError("Semantic image identity differs")
                for key in ("domain", "p", "m", "multiplication", "inverse", "types"):
                    if group["spec"][key] != core.group_spec(domain)[key]:
                        raise ValueError("Comparator changes frozen group structure")
        for cell in inputs["cells"]:
            if cell["status"] != "prepared_inputs_only" or cell["pca_schema"]["basis"]["shape"][1] != 16:
                raise ValueError("Unprepared cell or changed PCA dimension")
            bank_rel = cell["bank_path"]
            rows_rel = str(Path(bank_rel).with_name("rows.json"))
            raw = checked_read(z, "alternatives/input_snapshot/" + rows_rel, manifest["files"][rows_rel])
            rows = json.loads(raw)
            validate_contexts(rows["test_contexts"], cell["domain"], families)
            p = {"C7": 7, "D4": 4}[cell["domain"]]
            inds = rows["test_indices"]
            if np.asarray(inds).shape != (128, p) or len(set(np.asarray(inds).ravel())) != 128 * p:
                raise ValueError("Invalid retained original test row indices")
            bank_raw = checked_read(z, "alternatives/input_snapshot/" + bank_rel, manifest["files"][bank_rel])
            with zipfile.ZipFile(io.BytesIO(bank_raw)) as arrays:
                if set(arrays.namelist()) != {"test.npy", "train.npy"}:
                    raise ValueError("Bank array names differ")
                with arrays.open("test.npy") as stream:
                    version = np.lib.format.read_magic(stream)
                    if version != (1, 0):
                        raise ValueError("Unexpected NPY version")
                    shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
                if shape != (128, p, 16) or dtype != np.dtype("float64"):
                    raise ValueError("Retained test coordinate shape/dtype mismatch")
            rows_by_cell[cell["cell"]] = rows
            receipts.append({"cell": cell["cell"], "domain": cell["domain"], "test_shape": list(shape),
                             "bank_sha256": manifest["files"][bank_rel], "rows_sha256": manifest["files"][rows_rel]})
    report = {"status": "PREPARED_NOT_RUN", "input_status": "AVAILABLE_HEADERS_AND_IDENTITIES_VERIFIED",
              "real_family_deletions_executed": 0, "coordinate_values_loaded": False,
              "archive_sha256_expected": config["archive_sha256_expected"],
              "whole_archive_rehashed_by_this_entrypoint": False,
              "read_member_hashes_verified": True, "metadata_hashes": config["metadata_hashes"],
              "source_hashes": config["source_hashes"], "planned_cells": 48,
              "planned_deletions": 64, "planned_cell_deletions": 3072,
              "planned_image_evaluations": 64 * 24 * (120 + 3), "cells": receipts,
              "family_ids": families, "cross_domain_pairing_claimed": False,
              "rank_rule": "strict min/max eigenvalue ratio > 1e-10",
              "tie_rule": "1e-8 + 1e-7 * max(abs(candidate_gamma), abs(semantic_gamma))"}
    return report, inputs, manifest, plan, rows_by_cell, families


def execution_authorized(config, opt_in):
    return config.get("full_run") is True and opt_in is True


def cpu_thread_limit():
    """Additional loaded-BLAS control; import-time environment limits are primary."""
    try:
        from threadpoolctl import threadpool_limits
        return threadpool_limits(limits=1)
    except ImportError:
        return nullcontext()


def run(archive, config, output, opt_in=False):
    # A fresh destination is required even for metadata-only inspection.
    if output.exists():
        raise ValueError("Choose a new output directory; existing evidence is never overwritten")
    output.mkdir(parents=True)
    try:
        prepared = inspect_input(archive, config)
    except (OSError, KeyError, ValueError, zipfile.BadZipFile) as exc:
        report = {"status": "PREPARED_NOT_RUN", "input_status": "MISSING_INPUT_OR_INVALID_INPUT",
                  "reason": str(exc), "real_family_deletions_executed": 0}
        (output / "availability.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    report, inputs, manifest, plan, rows, families = prepared
    (output / "availability.json").write_text(json.dumps(report, indent=2) + "\n")
    if not execution_authorized(config, opt_in):
        return report
    # Beyond this line requires separately authorized scientific execution.
    # Environment limits above apply before NumPy is imported by the CLI.
    # When installed, threadpoolctl additionally enforces the loaded BLAS pool.
    thread_limit = cpu_thread_limit()
    (output / "execution_config.json").write_bytes(canonical(config))
    summaries = []
    with thread_limit, zipfile.ZipFile(archive) as z, (output / "deletions.jsonl").open("x") as dst:
        for cell in inputs["cells"]:
            name = cell["bank_path"]
            raw = checked_read(z, "alternatives/input_snapshot/" + name, manifest["files"][name])
            with np.load(io.BytesIO(raw), allow_pickle=False) as bank:
                Z = bank["test"]
            results = []
            for result in execute_cell(Z, rows[cell["cell"]]["test_contexts"], cell["domain"], families, plan["subgroups"][cell["domain"]]):
                result.update(cell=cell["cell"], domain=cell["domain"], binding=cell["binding"],
                              pca_schema=cell["pca_schema"], original_bank_sha256=manifest["files"][name])
                dst.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
                dst.flush()
                results.append(result["comparison"])
            valid = [r for r in results if r["status"] == "complete_descriptive"]
            summaries.append({"cell": cell["cell"], "domain": cell["domain"], "planned": 64,
                "valid": len(valid), "failed": 64 - len(valid),
                "unique_leading_deletion_count": sum(r["uniquely_leading"] for r in valid),
                "frequency_denominator": 64, "frequency_is_probability": False,
                **{key: [min(r[key] for r in valid), max(r[key] for r in valid)] if valid else None
                   for key in ("d_top", "d_median", "rank_low", "rank_high")}})
    done = {"status": "DONE", "interpretation": "Descriptive finite-family composition sensitivity only",
            "cells": summaries, "population_inference": False, "independent_confirmation": False,
            "coordinate_execution": True, "full_panel_valid": all(c["failed"] == 0 for c in summaries)}
    (output / "execution_summary.json").write_text(json.dumps(done, indent=2, allow_nan=False) + "\n")
    return done


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-family-sensitivity", action="store_true", default=False)
    args = parser.parse_args(argv)
    result = run(args.archive, json.loads(args.config.read_text()), args.output, args.execute_family_sensitivity)
    print(json.dumps({k: result[k] for k in ("status", "input_status") if k in result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
