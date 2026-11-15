"""Bounded MAIN30 tasks B/C: new CPU derivations from immutable saved results.

No original gamma, eta, type budget, witness, or image experiment is rerun.
Only the additional identity-action reference is checked from saved test banks.
Run from the release root: python analysis/main30_bounded_v1/derive.py --archive ZIP
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics as st
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "studies/spectral_extensions_v1"))
import numerics as frozen

SOURCE = ROOT / "results/tmlr_followup_v2/spectral"
OUTPUT = ROOT / "results/main30_bounded_v1"
SPEC = "spectral-results-436df4be8558423f8e84aea0e4632408"
SNAP = "alternatives/input_snapshot"
ARCHIVE_SHA = "ef87e9aaeb8f093ba0c2438590d6a30b5dffcd256993e2366bd600cca3cb5bbb"
CHECKS = []


def sha(b):
    return hashlib.sha256(b).hexdigest()


def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def file_hash(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def require(condition, name):
    if not condition:
        raise ValueError("NUMERICAL_FAILURE_OR_INPUT_MISMATCH: " + name)


def close(a, b, name):
    residual = abs(a - b)
    tolerance = frozen._tol(a, b)
    require(math.isfinite(residual) and residual <= tolerance, name)
    CHECKS.append(dict(check=name, residual=residual, tolerance=tolerance))


def leq(a, b, name):
    require(math.isfinite(a) and math.isfinite(b) and a <= b + frozen._tol(a, b), name)


def read_csv(p):
    with p.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_json(path, x):
    path.write_text(json.dumps(x, indent=2, sort_keys=True, allow_nan=False) + "\n")


def risk_fields(gamma, eta, e_triv, b_triv):
    a, u = 2 * (1 - gamma), 2 * (1 - gamma * gamma)
    return dict(gamma=gamma, eta=eta, e_triv=e_triv, b_triv=b_triv,
                total_nontrivial_energy=1-e_triv, selected_nontrivial_energy=gamma-b_triv,
                a=a, u=u, identity_risk=2*(1-e_triv),
                premium_lower=max(0.0, a-eta), premium_upper=u-eta)


def registration_fields(rows):
    semantic_rows = [r for r in rows if r["semantic_registration"]]
    require(len(semantic_rows) == 1, "one semantic image")
    sem = semantic_rows[0]
    # Compare canonical identities, never assume an image ordinal is semantic.
    alt = [r for r in rows if r["image_sha256"] != sem["image_sha256"]]
    require(len(alt) == len(rows)-1 and len(alt) > 0, "unique image identities")
    g = sem["gamma"]
    best = max(r["gamma"] for r in alt)
    better = sum(r["gamma"] > g + frozen._tol(r["gamma"], g) for r in alt)
    tied = sum(abs(r["gamma"]-g) <= frozen._tol(r["gamma"], g) for r in alt)
    rank_low, rank_high = 1+better, 1+better+tied
    # Cross-check the exact original rank implementation, including semantic row.
    original = frozen.registration_ranks([
        dict(subgroup_id=r["image_sha256"], semantic_registration=r["semantic_registration"],
             test_bank_spectral=dict(status="complete", metrics=dict(gamma=r["gamma"]))) for r in rows])
    require((rank_low, rank_high) == (original["rank_low"], original["rank_high"]), "frozen rank rule")
    return dict(semantic_image_sha256=sem["image_sha256"], semantic_gamma=g,
                alternatives=len(alt), d_top=g-best,
                d_median=g-st.median(r["gamma"] for r in alt),
                spread=max(r["gamma"] for r in rows)-min(r["gamma"] for r in rows),
                rank_low=rank_low, rank_high=rank_high,
                standing="unique_leading" if rank_high == 1 else "tied_leading" if rank_low == 1 else "behind",
                alternative_ties=tied,
                best_alternative_images=";".join(sorted(r["image_sha256"] for r in alt if r["gamma"] == best)),
                pairwise_tolerance_min=min(frozen._tol(r["gamma"],g) for r in alt),
                pairwise_tolerance_max=max(frozen._tol(r["gamma"],g) for r in alt))


def direct_identity(Z, spec):
    """Only new identity endpoint: no K, budget optimization, or fitting."""
    Z = np.asarray(Z, dtype=np.float64)
    r = Z.shape[-1]
    flat = Z.reshape(-1, r)
    sigma = flat.T @ flat / len(flat)
    vals, vectors = np.linalg.eigh((sigma+sigma.T)/2)
    require(vals[0] > frozen.spectral.RANK_RATIO*vals[-1], "saved-bank full rank")
    X = Z @ ((vectors*(vals**-0.5))@vectors.T)
    residual = frozen.spectral._matrix_residual(X.reshape(-1,r).T@X.reshape(-1,r)/len(flat),np.eye(r))
    require(residual <= frozen.spectral.MATRIX_TOL, "saved-bank whitening")
    total_trivial = float(np.mean(np.sum(np.mean(X,axis=1)**2,axis=1))/r)
    direct = float(np.mean([np.mean(np.sum((X-X[:,action,:])**2,axis=-1))/r for action in spec["actions"]]))
    return dict(identity_risk_direct=direct,e_triv_from_context_mean=total_trivial,
                whitening_residual=residual, sigma_min_max_ratio=float(vals[0]/vals[-1]))


def validate_contexts(old, row, domain, protocol):
    p = 7 if domain == "C7" else 4
    require(row["test_contexts"] == old["contexts"] and row["test_indices"] == old["row_indices"], "same test row bank")
    contexts = row["test_contexts"]
    require(len(contexts)==128 and len({c["family_id"] for c in contexts})==64, "64 paired families")
    require({(c["family_id"],c["paraphrase_id"]) for c in contexts} ==
            {(f,v) for f in protocol["test_families"] for v in [0,1]}, "complete family/paraphrase panel")
    for c in contexts:
        require(c["split"]=="test" and c["state_ids"]==list(range(p)), "complete ordered test states")
        label = old["archive_row_domain"]
        require(c["row_ids"] == [f'{label}/{c["family_id"]}/p{c["paraphrase_id"]}/lexical_{label}/s{s}' for s in range(p)], "canonical row identities")


def tex_number(x, precision=5):
    # Raw CSV floats are untouched; TeX uses ordinary decimal rounding only.
    return f"{x:.{precision}f}"


def tex_scientific(x):
    if x == 0:
        return "$0$"
    mantissa, exponent = f"{x:.2e}".split("e")
    return "$"+mantissa+r"\!\times\!10^{"+str(int(exponent))+"}$"


def render_tables(out, risks, domains, margins, margin_domains, types):
    def table(filename, columns, header, rows):
        text = "% Generated by analysis/main30_bounded_v1/derive.py; never edit numbers.\n"
        text += "\\begin{tabular}{"+columns+"}\n\\toprule\n"+header+" \\\\\n\\midrule\n"
        text += "\n".join(" & ".join(row)+r" \\" for row in rows)
        text += "\n\\bottomrule\n\\end{tabular}\n"
        (out/filename).write_text(text)
    table("risk_reference_table.tex","lrrrr",r"Domain & Mean $\eta$ & Mean $L(I)$ & Mean $p^-$ & Mean $p^+$",
          [[d["domain"]]+[tex_number(d[k]) for k in ["mean_eta","mean_identity_risk","mean_premium_lower","mean_premium_upper"]] for d in domains])
    margin_table=[]
    for d in margin_domains:
        for key,label in [("d_top",r"$d_{\rm top}$"),("d_median",r"$d_{\rm median}$")]:
            margin_table.append([d["domain"] if key=="d_top" else "",label]+[tex_scientific(d[key+"_"+k]) for k in ["median","min","max"]]+[f'{d["unique_leading"]}/{d["tied_leading"]}/{d["behind"]}' if key=="d_top" else ""])
    table("registration_margin_table.tex","llrrrc",r"Domain & Margin & Median & Min. & Max. & U/T/B",margin_table)
    def short(r):
        return r["model_id"].split("/")[-1].replace("Llama-", "L-").replace("Qwen2.5-","Q-")
    for domain in ["C7","D4"]:
        rs=[r for r in risks if r["domain"]==domain]
        table(f"cell_risk_{domain}.tex","lr rrrrrrr",r"Model & Layer & $\eta$ & $e_0$ & $b_0$ & $L(I)$ & $p^-$ & $p^+$ & $\gamma$",
              [[short(r),str(r["layer"])]+[tex_number(r[k]) for k in ["eta","e_triv","b_triv","identity_risk","premium_lower","premium_upper","gamma"]] for r in rs])
        ms=[r for r in margins if r["domain"]==domain]
        table(f"cell_margins_{domain}.tex","lr rrrcc",r"Model & Layer & $d_{\rm top}$ & $d_{\rm med}$ & spread & Rank & Standing",
              [[short(r),str(r["layer"])]+[tex_number(r[k],7) for k in ["d_top","d_median","spread"]]+[f'{r["rank_low"]}--{r["rank_high"]}',dict(unique_leading="U",tied_leading="T",behind="B")[r["standing"]]] for r in ms])
        names=["trivial","plane_1","plane_2","plane_3"] if domain=="C7" else ["chi_pp","chi_pm","chi_mp","chi_mm","E"]
        tm={(r["cell"],r["type"]):r for r in types}
        # Each type cell reports total/selected normalized energy, including tiny raw signed zeros in CSV.
        table(f"cell_type_energy_{domain}.tex","lr"+"r"*len(names),"Model & Layer & "+" & ".join(r"$e/b$: "+n.replace("_",r"\_") for n in names),
              [[short(r),str(r["layer"])]+[f'{tm[r["cell"],n]["e"]:.5f}/{tm[r["cell"],n]["b"]:.5f}' for n in names] for r in rs])
    macros=[]
    for d in domains:
        suffix="Cseven" if d["domain"]=="C7" else "Dfour"
        for key,name in [("mean_identity_risk","Identity"),("mean_premium_lower","PremiumLower"),("mean_premium_upper","PremiumUpper")]:
            macros.append(r"\newcommand{\Bounded"+name+suffix+"}{"+tex_number(d[key])+"}")
        macros.append(r"\newcommand{\BoundedPositive"+suffix+"}{"+str(d["positive_lower_beyond_tolerance"])+"}")
        md=next(m for m in margin_domains if m["domain"]==d["domain"])
        for key,name in [("unique_leading","Unique"),("tied_leading","Tied"),("behind","Behind")]:
            macros.append(r"\newcommand{\Bounded"+name+suffix+"}{"+str(md[key])+"}")
    macros.extend([
        r"\newcommand{\BoundedAllPositive}{"+str(sum(r["positive_lower_beyond_tolerance"] for r in risks))+"}",
        r"\newcommand{\BoundedCellCount}{"+str(len(risks))+"}",
        r"\newcommand{\BoundedCellsPerDomain}{"+str(domains[0]["n_cells"])+"}",
        r"\newcommand{\BoundedIdentityMaxError}{"+tex_scientific(max(abs(r["identity_risk"]-r["identity_risk_direct"]) for r in risks))+"}",
        r"\newcommand{\BoundedScalarAtol}{"+tex_scientific(frozen.spectral.SCALAR_ATOL)+"}",
        r"\newcommand{\BoundedScalarRtol}{"+tex_scientific(frozen.spectral.SCALAR_RTOL)+"}"])
    (out/"bounded_macros.tex").write_text("% Generated from full precision cell rows.\n"+"\n".join(macros)+"\n")


def run(archive, output=OUTPUT):
    CHECKS.clear()
    output.mkdir(parents=True,exist_ok=True)
    metadata = output / "input_metadata"
    metadata.mkdir(exist_ok=True)
    source_paths = list(SOURCE.glob("*.csv")) + [SOURCE/n for n in ["file_hashes.json","provenance.json","metrics.json","original_descriptive_ranks.json"]]
    source_paths += [ROOT/n for n in ["studies/spectral_finite_bank_v1/gamma_r.py","studies/spectral_extensions_v1/numerics.py","studies/spectral_finite_bank_v1/protocol.json","studies/spectral_extensions_v1/protocol.json","studies/spectral_finite_bank_v1/source_manifest.json","studies/spectral_extensions_v1/source_manifest.json","scripts/summarize_spectral_extensions.py"]]
    before={str(p.relative_to(ROOT)):file_hash(p) for p in source_paths}
    write_json(output/"source_manifest_before.json",before)
    saved_hashes=json.loads((SOURCE/"file_hashes.json").read_text())
    for p in source_paths:
        if p.parent==SOURCE and p.name in saved_hashes:
            require(before[str(p.relative_to(ROOT))]==saved_hashes[p.name],"compact input hash "+p.name)
    require(file_hash(archive)==ARCHIVE_SHA,"archive SHA-256")
    require(archive.with_name(archive.name+".sha256").read_text().split()[0]==ARCHIVE_SHA,"archive checksum sidecar")
    provenance=json.loads((SOURCE/"provenance.json").read_text())
    core=read_csv(SOURCE/"spectral_cells.csv")
    originals={r["cell"]:r for r in read_csv(SOURCE/"original_spectral_cell_summary.csv")}
    alternatives=read_csv(SOURCE/"alternatives_units.csv")
    repeated_types=read_csv(SOURCE/"original_spectral_type_spectra.csv")
    original_types={}
    for t in repeated_types:
        key=(t["cell"],t["type"])
        value=(float(t["e"]),float(t["b"]))
        require(key not in original_types or original_types[key]==value,"type-energy repeated rows consistent")
        original_types[key]=value
    require(len(original_types)==216,"all 216 cell/type energies")
    ranks=json.loads((SOURCE/"original_descriptive_ranks.json").read_text())["cells"]
    require(len(core)==len(originals)==48 and Counter(r["domain"] for r in core)=={"C7":24,"D4":24},"all 48 spectral cells")
    require(len(alternatives)==2952,"all 2952 image conditions")
    member_hashes={}
    risks=[]; all_types=[]; image_rows=[]; identities=[]; groups={}
    with zipfile.ZipFile(archive) as z:
        require(len(z.namelist())==len(set(z.namelist())),"no duplicate archive names")
        def read(name,expected=None,retain=False):
            b=z.read(name); digest=sha(b)
            require(expected is None or digest==expected,"archive member hash "+name)
            member_hashes[name]=digest
            if retain:
                (metadata/name.replace("/","__")).write_bytes(b)
            return b
        def js(name,expected=None,retain=False):
            return json.loads(read(name,expected,retain))
        binding=provenance["unit_verification"]["alternatives"]["binding"]
        plan=js("alternatives/plan.json",binding["plan_sha256"],True)
        run_manifest=js("alternatives/run_manifest.json",binding["run_manifest_sha256"],True)
        prepared=js(SNAP+"/prepared_manifest.json",run_manifest["prepared_manifest_sha256"],True)
        def snapshot(name,retain=False):
            return js(SNAP+"/"+name,prepared["files"][name],retain)
        parent=snapshot("parent_input_manifest.json",True)
        inputs=snapshot("inputs.json",True)
        spec_input=js(SPEC+"/input_manifest.json",retain=True)
        require(parent==spec_input,"spectral and alternative parent bank identity")
        protocol=parent["protocol"]
        require(protocol==json.loads((ROOT/"studies/spectral_finite_bank_v1/protocol.json").read_text()),"frozen protocol identity")
        require(protocol["rank"]==16 and all(protocol[k]=="uniform" for k in ["context_weight","state_weight","action_weight"]),"rank and uniform weights")
        oldmap={r["cell"]:r for r in parent["cells"]}
        inputmap={r["cell"]:r for r in inputs["cells"]}
        require(set(oldmap)==set(inputmap)==set(originals),"same 48 input identities")
        for domain, specs in plan["subgroups"].items():
            require(len(specs)=={"C7":120,"D4":3}[domain],"complete saved image set")
            hashes=[]
            for spec in specs:
                frozen.spectral.validate_group(spec["spec"])
                perms=spec["image_permutations"]
                require(perms==sorted(spec["spec"]["actions"]),"canonical saved image permutations")
                h=sha(canonical(perms)); hashes.append(h)
                groups[(domain,spec["id"])]=dict(spec,image_sha256=h)
            require(len(set(hashes))==len(hashes) and sum(s["semantic"] for s in specs)==1,"distinct saved images")
            semantic=next(s for s in specs if s["semantic"])
            require(semantic["image_permutations"]==sorted(parent["groups"][domain]["actions"]),"semantic action image identity")
        for row in sorted(core,key=lambda r:(r["domain"],r["model_id"],int(r["layer"]))):
            cell,domain=row["cell"],row["domain"]
            old,new=oldmap[cell],inputmap[cell]
            require(new["source_members"]==old["members"] and new["pca_schema"]==old["pca_schema"] and new["binding"]==old["binding"],"matching source/PCA/cache identity "+cell)
            summary=js(row["source_summary"],row["source_summary_sha256"])
            require(summary["status"]=="complete" and summary["r"]==16 and summary["n_contexts"]==128,"complete spectral summary")
            for k,v in summary["metrics"].items():
                require(v==float(row[k])==float(originals[cell][k]),"original full precision endpoint "+cell+"/"+k)
            types=summary["types"]
            close(sum(t["e"] for t in types),1.0,cell+"/type_energy_sum")
            close(sum(t["b"] for t in types),float(row["gamma"]),cell+"/selected_energy_sum")
            for t in types:
                require(t["e"]==float(row[t["name"]+"_e"]) and t["b"]==float(row[t["name"]+"_b"]),"type full precision")
                require((t["e"],t["b"])==original_types[(cell,t["name"])],"original type spectra normalization")
                leq(0,t["b"],"selected nonnegative");leq(t["b"],t["e"],"selected <= total")
                all_types.append(dict(cell=cell,domain=domain,type=t["name"],e=t["e"],b=t["b"],real_dimension=t["real_dimension"],selected_copies=t["selected_copies"],selected_dimension=t["selected_dimension"]))
            trivial=next(t for t in types if t["name"] in ["trivial","chi_pp"])
            require(trivial["e"]==float(row["trivial_e"]) and trivial["b"]==float(row["trivial_b"]),"total vs selected trivial identity")
            fields=risk_fields(float(row["gamma"]),float(row["eta_LS"]),trivial["e"],trivial["b"])
            close(fields["eta"],float(row["eta_spec"]),cell+"/eta_LS_spectral")
            close(fields["a"],float(row["L_lower"]),cell+"/lower");close(fields["u"],float(row["L_upper"]),cell+"/upper")
            leq(fields["eta"],fields["identity_risk"],"eta <= identity");leq(fields["a"],fields["identity_risk"],"a <= identity")
            leq(fields["premium_lower"],fields["premium_upper"],"premium bounds")
            rowmeta=snapshot("cells/"+cell+"/rows.json")
            validate_contexts(old,rowmeta,domain,protocol)
            bank_member=SNAP+"/"+new["bank_path"]
            bank_bytes=read(bank_member,prepared["files"][new["bank_path"]])
            with np.load(io.BytesIO(bank_bytes),allow_pickle=False) as banks:
                Z=banks["test"]
            require(Z.shape==(128,7 if domain=="C7" else 4,16),"saved test bank shape")
            direct=direct_identity(Z,parent["groups"][domain])
            close(direct["e_triv_from_context_mean"],fields["e_triv"],cell+"/trivial_context_mean")
            close(direct["identity_risk_direct"],fields["identity_risk"],cell+"/identity_direct")
            identity=dict(cell=cell,domain=domain,model_id=old["model_id"],model_revision=old["revision"],tokenizer_revision=old["tokenizer_revision"],layer=old["layer"],readout=old["readout"],partition="test",rank=16,contexts=128,families=64,
                          source_archive_sha256=parent["archive_sha256"],source_members=old["members"],pca_schema=old["pca_schema"],binding=old["binding"],test_contexts_sha256=sha(canonical(old["contexts"])),bank_member=bank_member,bank_member_sha256=sha(bank_bytes),
                          state_weight="uniform",context_weight="uniform",group_weight="uniform",include_identity=True,include_fixed_points=True,metric="uncentered empirical second moment of this complete original Study 2 test bank",semantic_action_sha256=sha(canonical(parent["groups"][domain]["actions"])))
            join_key=sha(canonical(identity)); identity["join_key"]=join_key;identities.append(identity)
            risks.append(dict(cell=cell,domain=domain,model_id=old["model_id"],layer=old["layer"],join_key=join_key,**fields,**direct,
                              identity_source="derived_from_saved_type_energy_and_direct_saved_bank_check",positive_lower_beyond_tolerance=fields["premium_lower"]>frozen._tol(fields["premium_lower"],0),source_summary=row["source_summary"],source_summary_sha256=row["source_summary_sha256"]))
        planmap={r["unit"]:r for r in plan["units"]}
        require(set(planmap)=={r["unit"] for r in alternatives},"full planned image units")
        joinmap={r["cell"]:r["join_key"] for r in risks}
        for row in alternatives:
            planned=planmap[row["unit"]]
            require(all(row[k]==planned[k] for k in ["cell","domain","subgroup_id"]),"saved unit identity")
            group=groups[row["domain"],row["subgroup_id"]]
            semantic=row["semantic_registration"]=="True"
            require(semantic==planned["semantic_registration"]==group["semantic"],"canonical semantic flag")
            require(row["status"]==row["test_spectral_status"]=="complete","image evaluable")
            image_rows.append(dict(cell=row["cell"],domain=row["domain"],join_key=joinmap[row["cell"]],image_sha256=group["image_sha256"],saved_subgroup_id=row["subgroup_id"],semantic_registration=semantic,gamma=float(row["test_bank_gamma"]),source_summary=row["source_summary"],source_summary_sha256=row["source_summary_sha256"]))
    bycell=defaultdict(list)
    for r in image_rows:bycell[r["cell"]].append(r)
    margins=[]
    for r in risks:
        rows=bycell[r["cell"]]
        require(len(rows)=={"C7":120,"D4":3}[r["domain"]],"images per cell")
        m=registration_fields(rows)
        close(m["semantic_gamma"],r["gamma"],r["cell"]+"/semantic_gamma_same_endpoint")
        require((m["rank_low"],m["rank_high"])==(ranks[r["cell"]]["rank_low"],ranks[r["cell"]]["rank_high"]),"published rank reproduction")
        margins.append({k:r[k] for k in ["cell","domain","model_id","layer","join_key"]}|m)
    domains=[];margin_domains=[]
    for domain in ["C7","D4"]:
        rs=[r for r in risks if r["domain"]==domain];ms=[m for m in margins if m["domain"]==domain]
        d=dict(domain=domain,n_cells=24,**{"mean_"+k:st.mean(r[k] for r in rs) for k in ["gamma","eta","identity_risk","a","u","premium_lower","premium_upper"]},positive_lower_beyond_tolerance=sum(r["positive_lower_beyond_tolerance"] for r in rs),min_premium_lower=min(r["premium_lower"] for r in rs),max_premium_lower=max(r["premium_lower"] for r in rs))
        domains.append(d)
        md=dict(domain=domain,n_cells=24,**{s:sum(m["standing"]==s for m in ms) for s in ["unique_leading","tied_leading","behind"]})
        for k in ["d_top","d_median","spread"]:
            md.update({k+"_"+s:f(m[k] for m in ms) for s,f in [("median",st.median),("min",min),("max",max)]})
        margin_domains.append(md)
        require(f'{d["mean_gamma"]:.5f}'=={"C7":"0.98510","D4":"0.93528"}[domain],"Table3 printed gamma reproduction")
        require(md["unique_leading"]+md["tied_leading"]=={"C7":7,"D4":16}[domain],"published semantic top count reproduction")
    for name,rows in [("cell_risk_reference.csv",risks),("domain_risk_reference.csv",domains),("registration_margins.csv",margins),("registration_margin_summary.csv",margin_domains),("image_endpoints.csv",image_rows),("type_energy.csv",all_types)]:write_csv(output/name,rows)
    write_json(output/"cell_identities.json",identities)
    write_json(output/"canonical_images.json",[{"domain":d,"saved_subgroup_id":i,"image_sha256":g["image_sha256"],"semantic":g["semantic"],"image_permutations":g["image_permutations"]} for (d,i),g in groups.items()])
    render_tables(output,risks,domains,margins,margin_domains,all_types)
    after={str(p.relative_to(ROOT)):file_hash(p) for p in source_paths}
    require(before==after,"immutable inputs unchanged")
    write_json(output/"source_manifest_after.json",after)
    report=dict(status="DONE",new_analysis=True,original_experiments_rerun=False,new_model_execution=False,family_sensitivity_run=False,archive_name=archive.name,archive_sha256=ARCHIVE_SHA,archive_members_read=member_hashes,
                analysis_sources={str(p.relative_to(ROOT)):file_hash(p) for p in sorted(Path(__file__).parent.rglob("*")) if p.is_file() and p.suffix in [".py",".json",".md"]},
                counts=dict(spectral_cells=48,image_conditions=2952,type_rows=len(all_types),identity_direct_checks=48),checks=CHECKS,
                tolerance=dict(absolute=frozen.spectral.SCALAR_ATOL,relative=frozen.spectral.SCALAR_RTOL,formula="atol+rtol*max(abs(a),abs(b))",units="dimensionless normalized energy or risk",source="studies/spectral_extensions_v1/numerics.py:_tol,registration_ranks"),
                gamma_above_one=[dict(cell=r["cell"],gamma=r["gamma"]) for r in risks if r["gamma"]>1],
                raw_float_policy="No clipping or tolerance changes; display uses ordinary rounding, no numerical zeroing.",
                identity_max_abs_residual=max(abs(r["identity_risk"]-r["identity_risk_direct"]) for r in risks),
                source_files_unchanged=before==after,domain_risk_reference=domains,registration_margin_summary=margin_domains,
                scope="New offline algebraic derivation and identity residual check on previously saved original Study 2 test banks; not original experiment recertification or independent confirmation.")
    write_json(output/"validation_report.json",report)
    write_json(output/"output_manifest.json",{str(p.relative_to(output)):file_hash(p) for p in sorted(output.rglob("*")) if p.is_file() and p.name!="output_manifest.json"})
    print(json.dumps({k:report[k] for k in ["status","counts","identity_max_abs_residual","domain_risk_reference","registration_margin_summary"]},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive",type=Path,required=True)
    parser.add_argument("--output",type=Path,default=OUTPUT)
    args=parser.parse_args()
    try:
        run(args.archive,args.output)
    except (ValueError,KeyError,OSError) as error:
        args.output.mkdir(parents=True,exist_ok=True)
        write_json(args.output/"validation_report.json",dict(status="BLOCKED_OR_NUMERICAL_FAILURE",reason=str(error)))
        raise
