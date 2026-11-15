"""Temporary software fixtures only: never observations or manuscript evidence."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from bsg_next.cli import main,DEFAULT_CONFIG
from bsg_next.config import load_config
from bsg_next.io import atomic_json,prepare_run,identity,read_json
from bsg_next.reporting import build_report,export_paper,collect_results
from bsg_next.native import native_decisions
from bsg_next.statistics import family_decisions,crossed_decisions
from bsg_next.pipeline import StudyPipeline,ScientificGateError


def production_fixture(tmp_path,native_enabled=True):
    """Fabricated deterministic software input, confined to pytest's temp tree."""
    cfg=load_config(DEFAULT_CONFIG);cfg["study_id"]="TEST_FIXTURE_NEVER_PUBLISH"
    cfg["technical_test"]=False
    cfg.setdefault("native",{})["technical_run"]=False
    if not native_enabled:
        cfg["native"].update(enabled=False,models=[])
        cfg["peers"].update(reproduction_anchors=False,factorial_anova=False)
        cfg["crossmodel"]["enabled"]=False
    run=tmp_path/"run";marker=prepare_run(run,cfg,"prospective_study")
    atomic_json(run/"study_config.json",cfg)
    for name in ("structural_confirmation","native_confirmation","peer_anchors","peer_factorial"):
        atomic_json(run/"stage_state"/(name+".json"),{"status":"complete","identity":marker["identity"],"result":{"fixture_only":True}})
    atomic_json(run/"confirmation_freeze.json",{"configuration":marker["identity"],"fixture_only":True,"test_families":32})
    variation=np.linspace(-.01,.01,32)
    structural={"fixture_only":True,"panel_complete":True,"formal_inference_enabled":True,
                "decisions":family_decisions({"C7":{"pred":-.12+variation,"ctx":variation,"eligible":True},
                                                "D4_corner":{"pred":.02+variation,"ctx":.001+variation,"eligible":False}}),
                "S3":{"decisions":crossed_decisions({"pred":np.arange(32).reshape(8,4)/1000-.03,
                                                        "ctx":np.arange(32).reshape(8,4)/10000},eligible=False)}}
    native={"fixture_only":True,"technical_run":False,"status":"completed",
            "decisions":native_decisions({f"m{i}.{domain}":-.08+variation for i in range(2) for domain in ("C7","D4_free8")}),
            "cells":{},"private_path":"/workspace/private-user/native_cache/abcdef.npz",
            "error_context":"source /"+"Users/private-user/research/data.json unavailable"}
    atomic_json(run/"outputs/structural_confirmation/summary.json",structural)
    atomic_json(run/"outputs/native_study/native_results.json",native)
    project=tmp_path/"publication-fixture"
    atomic_json(project/"release/artifact_manifest.json",{"schema_version":1,"artifacts":[]})
    atomic_json(project/"release/pdf_qa.json",{"fixture_only":True})
    return cfg,run,project


def test_moving_model_revision_and_configuration_changes_are_rejected(tmp_path,capsys):
    cfg=load_config(DEFAULT_CONFIG);cfg["models"][0]["revision"]="main"
    path=tmp_path/"moving.json";atomic_json(path,cfg)
    assert main(["validate","--config",str(path)])==2
    assert "40-character" in capsys.readouterr().err
    cfg,run,project=production_fixture(tmp_path/"fixture")
    cfg["seed"]+=1;atomic_json(run/"study_config.json",cfg)
    with pytest.raises(ValueError,match="identity mismatch"):export_paper(run,project)
    with pytest.raises(ValueError,match="identity mismatch"):build_report(run,tmp_path/"report")


@pytest.mark.parametrize("command",["native","peers","extensions"])
def test_model_commands_require_explicit_execution_flag(tmp_path,capsys,command):
    output=tmp_path/command
    assert main([command,"--output",str(output)])==2
    assert "--execute-models" in capsys.readouterr().err
    assert not output.exists()


def test_truncated_cli_runs_need_technical_marker(tmp_path,capsys):
    assert main(["synthetic","--mode","calibration","--limit","1","--output",str(tmp_path/"synthetic")])==2
    assert "--technical" in capsys.readouterr().err
    assert not (tmp_path/"synthetic").exists()
    assert main(["run","--phase","pilot","--output",str(tmp_path/"pilot")])==2
    assert not (tmp_path/"pilot").exists()


def test_pipeline_resumes_matching_stages_and_preserves_gate_failure(tmp_path):
    cfg=load_config(DEFAULT_CONFIG);run=tmp_path/"resume";pipeline=StudyPipeline(cfg,run);calls=[]
    def actual_stage(out):calls.append(str(out));return {"fixture_only":True,"value":3}
    assert pipeline.stage("fixture",actual_stage)["value"]==3
    resumed=StudyPipeline(cfg,run)
    assert resumed.stage("fixture",actual_stage)["value"]==3 and len(calls)==1
    def stopped(_out):raise ScientificGateError("fixture pilot power insufficient")
    with pytest.raises(ScientificGateError):resumed.stage("fixture_gate",stopped)
    assert read_json(run/"stage_state/fixture_gate.json")["status"]=="gate_stopped"
    changed=deepcopy(cfg);changed["seed"]+=1
    with pytest.raises(ValueError,match="different configuration"):StudyPipeline(changed,run)


def test_technical_results_cannot_populate_paper(tmp_path):
    cfg,run,project=production_fixture(tmp_path)
    cfg["technical_test"]=True;atomic_json(run/"study_config.json",cfg)
    with pytest.raises(ValueError,match="technical/smoke"):export_paper(run,project,allow_partial=True)
    assert not (project/"paper/generated/prospective_results.tex").exists()


def test_real_schema_export_has_three_structural_four_native_and_retains_negative(tmp_path):
    cfg,run,project=production_fixture(tmp_path)
    result=export_paper(run,project)
    tex=(project/result["tex"]).read_text()
    assert "-0.1200" in tex and tex.count("-0.0800")==4
    assert "$C_7$" in tex and "$D_4$ corners" in tex and "$S_3$ joint new" in tex
    assert "Llama" in tex and "Qwen" in tex and "Adjusted CI" in tex
    assert "footnotesize" in tex and "tabcolsep" in tex
    evidence=project/result["evidence"]
    report=read_json(evidence/"study_report.json")
    assert not report["technical_test"] and report["run_identity_verified"]
    assert not report["all_stages_complete"]
    native=report["native"]["decisions"]["contrasts"]
    assert len(native)==4 and all(v["mean"]<0 for v in native.values())
    assert all(v["status"]=="practical_target_excluded" for v in native.values())
    assert report["structural"]["decisions"]["contrasts"]["D4_corner.ctx"]["status"]=="uninformative"
    assert "/workspace/private-user" not in (evidence/"study_report.json").read_text()
    assert ("/"+"Users/private-user") not in (evidence/"study_report.json").read_text()
    assert not (project/"release/pdf_qa.json").exists()
    paths={a["path"] for a in read_json(project/"release/artifact_manifest.json")["artifacts"]}
    assert "paper/generated/prospective_status.tex" in paths and "paper/generated/prospective_results.tex" in paths
    rows=list(csv.DictReader((evidence/"paired_effects.csv").open()))
    assert len(rows)==10 and sum(row["study"]=="native" for row in rows)==4
    assert all(float(row["mean"])<0 for row in rows if row["study"]=="native")


def test_partial_export_masks_unfinished_or_gate_stopped_primary_results(tmp_path):
    cfg,run,project=production_fixture(tmp_path)
    state=read_json(run/"stage_state/native_confirmation.json");state["status"]="gate_stopped"
    atomic_json(run/"stage_state/native_confirmation.json",state)
    with pytest.raises(ValueError,match="complete primary stages"):export_paper(run,project)
    result=export_paper(run,project,allow_partial=True)
    tex=(project/result["tex"]).read_text()
    assert "-0.0800" not in tex and "Llama $C_7$ & Pending & Pending" in tex
    status=(project/result["status_tex"]).read_text()
    assert "Gate stopped" in status and "does not imply a positive" in status
    provenance=read_json(project/result["evidence"]/"provenance.json")
    assert provenance["partial"] and "native_confirmation" in provenance["missing_primary_stages"]


def test_empty_results_are_pending_not_zero_and_wrong_stage_identity_rejected(tmp_path):
    cfg,run,project=production_fixture(tmp_path)
    (run/"outputs/native_study/native_results.json").unlink()
    with pytest.raises(ValueError,match="complete primary results"):export_paper(run,project)
    out=export_paper(run,project,allow_partial=True)
    assert "Llama $C_7$ & Pending & Pending" in (project/out["tex"]).read_text()
    state=read_json(run/"stage_state/native_confirmation.json");state["identity"]="invalid"
    atomic_json(run/"stage_state/native_confirmation.json",state)
    with pytest.raises(ValueError,match="stage identity"):export_paper(run,project,allow_partial=True)


def test_disabled_native_peers_and_crossmodel_are_not_required_or_pending(tmp_path):
    cfg,run,project=production_fixture(tmp_path,native_enabled=False)
    for name in ("native_confirmation","peer_anchors","peer_factorial"):(run/"stage_state"/(name+".json")).unlink()
    result=export_paper(run,project)
    report=read_json(project/result["evidence"]/"study_report.json")
    expected=report["expected_core_stages"]
    assert "pilot_context_eligibility" in expected
    assert not any(name.startswith(("native_","peer_","crossmodel_")) for name in expected)
    assert "N/A (disabled)" in (project/result["status_tex"]).read_text()
    assert "Llama $C_7$ & N/A & N/A" in (project/result["tex"]).read_text()


def test_private_collection_excludes_checkpoints_credentials_hidden_and_default_caches(tmp_path):
    cfg,run,project=production_fixture(tmp_path)
    for relative in ("weights/model.safetensors","weights/pytorch_model.bin","access_token.txt","credentials.json",
                     "nested/.ssh/id_rsa",".env","native_cache/data.npy"):
        path=run/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text("DO_NOT_DISTRIBUTE")
    archive=tmp_path/"collected.zip";collect_results(run,archive)
    with zipfile.ZipFile(archive) as bundle:
        names=bundle.namelist()
        assert "run/study_config.json" in names and "COLLECTION_MANIFEST.json" in names
        assert not any(any(term in n for term in ("safetensors","pytorch_model","access_token","credentials","id_rsa",".env","data.npy")) for n in names)
    assert archive.with_suffix(".zip.sha256").is_file()
    archive2=tmp_path/"with-cache.zip";collect_results(run,archive2,include_caches=True)
    with zipfile.ZipFile(archive2) as bundle:assert "run/native_cache/data.npy" in bundle.namelist()
    with pytest.raises(ValueError,match="already exists"):collect_results(run,archive)


def test_public_report_redacts_accidental_secrets_and_collector_refuses_embedded_token(tmp_path):
    cfg,run,project=production_fixture(tmp_path)
    token="hf_"+"A"*32
    path=run/"outputs/native_study/native_results.json";value=read_json(path)
    value["HF_TOKEN"]=token;value["message"]="accidental credential "+token;atomic_json(path,value)
    build_report(run,tmp_path/"report")
    rendered=(tmp_path/"report/study_report.json").read_text()
    assert token not in rendered and "[redacted]" in rendered
    with pytest.raises(ValueError,match="credential content"):collect_results(run,tmp_path/"unsafe.zip")
