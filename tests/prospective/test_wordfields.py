"""Actual cache-to-comparison integration with no remote weights or GPU use."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from bsg_next.wordfields import (CONTEXTS, build_word_records, validate_word_rows,
    collect_word_fields, analyze_word_fields, analyze_structural_transport, _legal_automorphisms)
from bsg_next.data import build_structural_dataset, select_phase_records
from bsg_next.groups import get_group
from bsg_next.io import atomic_json, file_sha256, read_json
from bsg_next.extraction import extract_features


class WordRuntime:
    def __init__(self, model, runtime_config):
        self.model = None  # Deliberately lazy: reading model.config is a bug.
        self.model_id, self.n_layers = model["id"], 12
        self.telemetry = []
        self.matrix = np.random.default_rng(int(hashlib.sha256(self.model_id.encode()).hexdigest()[:8],16)).normal(size=(6, 7))

    def identity(self):
        return {"model_id":self.model_id,"revision":"mock-v1","dtype":"float32"}

    def features(self, records, layers, readout):
        assert readout == "span_mean"
        for begin in range(0,len(records),17):
            batch = records[begin:begin+17]
            values = []
            for row in batch:
                i = row["word_id"]
                c = list(CONTEXTS).index(row["context"])
                latent = np.array([np.sin(i*.7),np.cos(i*.4),i/20,(i%3)/3,(i%5)/5,1.])
                values.append(((1+c*.1)*latent + c*.03)@self.matrix)
            x = np.asarray(values,np.float32)
            yield list(range(begin,begin+len(batch))),{layer:x*(1+layer*.05) for layer in layers}

    def close(self):
        pass


def field_config(tmp_path):
    word_file = tmp_path/"words.json"
    word_file.write_text(json.dumps({"words":[f"concept{i}" for i in range(20)]}))
    return {"seed":101,"models":[{"id":"mock-source","layers":[3,6,9]},{"id":"mock-target","layers":[3,6,9]}],
        "crossmodel":{"word_file":str(word_file),"source_models":["mock-source"],"target_models":["mock-target"],
            "depths":[.25,.5,.75],"train_fraction":.8,"validation_fraction":.1,"test_fraction":.1,
            "rank":4,"nulls":0}}


def test_word_units_contexts_and_explicit_spans_are_sealed_together():
    rows = build_word_records([f"concept{i}" for i in range(20)])
    units = validate_word_rows(rows)
    assert len(rows)==120
    assert [sum(v["split"]==split for v in units.values()) for split in ("train","validation","test")] == [16,2,2]
    changed = [dict(row) for row in rows]
    changed[0]["split"] = "test" if changed[1]["split"]!="test" else "train"
    with pytest.raises(ValueError,match="partitions"):
        validate_word_rows(changed)
    badspan = [dict(row) for row in rows]
    badspan[0]["source_span"] = [0,4]
    with pytest.raises(ValueError,match="source span"):
        validate_word_rows(badspan)


def test_lazy_collection_reaches_real_equation12_and_never_fits_target_test(tmp_path,monkeypatch):
    import bsg_next.crossmodel as cross
    cfg = field_config(tmp_path)
    collection = collect_word_fields(cfg,tmp_path/"fields",runtime_factory=WordRuntime)
    assert collection["rows_per_model"]==120
    index = read_json(tmp_path/"fields"/"field_caches.json")
    assert all(not Path(path).is_absolute() for path in index["cache_dirs"])
    calls=[]
    real=cross.hu_transport
    def spy(source,target_train,train,test,**kwargs):
        prediction=real(source,target_train,train,test,**kwargs)
        calls.append((source.copy(),target_train.copy(),np.asarray(train).copy(),np.asarray(test).copy(),prediction.copy()))
        return prediction
    monkeypatch.setattr(cross,"hu_transport",spy)
    first=analyze_word_fields(cfg,tmp_path/"analysis1",tmp_path/"fields")
    assert len(first["pairs"])==15 and len(calls)==15
    assert all("hu_equation12" in result for result in first["pairs"].values())
    assert all(len(call[2])==16 and len(call[3])==2 for call in calls)
    original=[tuple(value.copy() for value in call) for call in calls]
    # Controlled fixture perturbation: only target test fields change; refresh
    # fixture checksums rather than weakening strict cache validation.
    leaf=tmp_path/"fields"/"mock-target"
    rows=[json.loads(line) for line in (leaf/"rows.jsonl").read_text().splitlines()]
    mask=np.asarray([row["split"]=="test" for row in rows])
    manifest=read_json(leaf/"manifest.json")
    for entry in manifest["arrays"].values():
        array=np.load(leaf/entry["path"],mmap_mode="r+")
        array[mask]+=np.arange(mask.sum())[:,None]*10
        array.flush()
        entry["sha256"]=file_sha256(leaf/entry["path"])
    atomic_json(leaf/"manifest.json",manifest)
    calls.clear()
    analyze_word_fields(cfg,tmp_path/"analysis2",tmp_path/"fields")
    for before,after in zip(original,calls):
        for a,b in zip(before,after):
            np.testing.assert_allclose(a,b,atol=0,rtol=0)
    provenance=read_json(tmp_path/"analysis2"/"field_provenance.json")
    assert provenance["target_test_used_for_fit"] is False
    assert "centered projected field" in provenance["mean_semantics"]


class FloorRuntime:
    def __init__(self, model, config):
        self.model_id=model["id"];self.telemetry=[]
    def identity(self):
        return {"model_id":self.model_id,"dtype":"float32","revision":"mock-floor"}
    def features(self,*args,**kwargs):
        raise AssertionError("Floor must call embedding_only, never transformer features")
    def embedding_only(self,records,readout):
        for start in range(0,len(records),256):
            batch=records[start:start+256]
            values=np.stack([np.eye(8)[0 if readout=="final_token" else row["state_id"]] for row in batch]).astype("float32")
            yield list(range(start,start+len(batch))),{0:values}
    def close(self):
        pass


def test_embedding_runner_six_models_category_ranks_and_zero_rank_s3(tmp_path):
    from bsg_next.embedding import run_embedding_floor
    from bsg_next.hf import MODEL_SPECS
    rows={"C7":build_structural_dataset("C7"),"D4":build_structural_dataset("D4")}
    # Complete tiny S3 orbits, preserving real metadata and train/val boundaries.
    all_s3=build_structural_dataset("S3")
    rows["S3"]=[r for r in all_s3 if (r["family_id"],r["orbit_id"]) in {
        ("structural_train_000","main_triple_train_000"),("structural_validation_000","main_triple_validation_000")}]
    cfg={"models":[{"id":name} for name in MODEL_SPECS]}
    result=run_embedding_floor(cfg,tmp_path/"floor",runtime_factory=FloorRuntime,records_by_domain=rows)
    assert result["models"]==6 and result["transformer_forwards"]==0
    assert not result["primary_panel_eligible"]
    assert len(result["cells"])==18
    for cell in result["cells"]:
        assert not cell["primary_panel_eligible"]
        if cell["group"]=="S3_role":
            assert cell["status"]=="zero_rank" and cell["q"] is None
        else:
            assert cell["q"]==(6 if cell["group"]=="C7" else 3)
            assert cell["scores"]["validation"]["mean_squared_target_error"]<1e-20


@pytest.mark.parametrize("domain,count",[("C7",6),("D4",8),("S3",6)])
def test_structural_registration_nulls_are_actual_automorphisms(domain,count):
    assert len(_legal_automorphisms(get_group(domain)))==count


class StructuralRuntime(WordRuntime):
    def features(self, records, layers, readout):
        for start in range(0,len(records),64):
            batch=records[start:start+64]
            latent=np.stack([np.r_[np.eye(4)[row["state_id"]],len(row["text"])/100,row["paraphrase_id"]] for row in batch])
            values=(latent@self.matrix).astype("float32")
            yield list(range(start,start+len(batch))),{layer:values for layer in layers}


def test_structural_alignment_joins_rows_and_holds_out_constraints(tmp_path):
    rows=build_structural_dataset("D4")
    caches=[]
    models=[{"id":"struct-source","layers":[1]},{"id":"struct-target","layers":[1]}]
    for model in models:
        ordered=rows if model["id"]=="struct-source" else rows[::-1]
        leaf=tmp_path/"cached"/model["id"]
        extract_features(StructuralRuntime(model,{}),ordered,[1],"span_mean",leaf)
        caches.append(str(leaf))
    atomic_json(tmp_path/"cached"/"caches.json",{"cache_dirs":caches})
    cfg={"models":models,"structural":{"q_primary":3},"crossmodel":{"source_models":["struct-source"],
        "target_models":["struct-target"],"structural_rank":3,"action_lambda_grid":[0,.01],"nulls":0}}
    result=analyze_structural_transport(cfg,tmp_path/"transport",tmp_path/"cached")
    assert len(result["pairs"])==1
    pair=next(iter(result["pairs"].values()))
    assert len(pair["intertwiner"]["heldout"])==5
    assert len(pair["intertwiner_legal_refitted_nulls"])==8
    assert all(item["refit"] for item in pair["intertwiner_legal_refitted_nulls"])
    provenance=read_json(tmp_path/"transport"/"structural_transport_provenance.json")
    assert provenance["new_model_forwards"]==0
    assert all("all R_g" in p["heldout_scope"] for p in provenance["pairs"].values())
