"""Grammar nuisance sensitivity preserves held-out units and S3 crossing."""
from collections import defaultdict

import numpy as np
import pytest

from bsg_next.data import build_structural_dataset
from bsg_next.extraction import extract_features
from bsg_next.grammar import run_grammar_sensitivity
from bsg_next.io import atomic_json


class MetadataRuntime:
    def __init__(self, name):
        self.name=name
    def identity(self):
        return {"model_id":self.name,"revision":"controlled-cache-fixture"}
    def features(self, rows, layers, readout):
        x=np.asarray([[row["state_id"],len(row["text"])] for row in rows],dtype="float32")
        yield list(range(len(rows))),{layer:x+layer for layer in layers}


def fixture(tmp_path, *, wrong_root=False, leak=False):
    c7=build_structural_dataset("C7",seed=101)
    test_families={r["family_id"]:r["grammar_root_id"] for r in c7 if r["split"]=="test"}
    roots=sorted(set(test_families.values()))[:3]
    selected=[]
    for root in roots:
        selected.extend(sorted(f for f,value in test_families.items() if value==root)[:2])
    selected=sorted(selected)
    nuisance=roots[0]
    by_group={}
    for domain in ("C7","S3"):
        allrows=build_structural_dataset(domain,seed=101)
        rows=[r for r in allrows if (r["split"]=="train" and r["family_id"]=="structural_train_000"
              and (domain=="C7" or r["orbit_id"]=="main_triple_train_000")) or
              (r["family_id"] in selected and r["split"] in ("test","test_joint")
               and (domain=="C7" or r["orbit_id"] in ("main_triple_test_000","main_triple_test_001")))]
        by_group[rows[0]["domain"]]=rows
    models=[{"id":"mock-a","layers":[1]},{"id":"mock-b","layers":[1]}]
    cfg={"seed":101,"technical_test":True,"models":models,"structural":{"q_primary":2,"groups":["C7","S3"]}}
    cells, aggregate= [],defaultdict(lambda:{"pred":defaultdict(list),"ctx":defaultdict(list)})
    for model_index,model in enumerate(models):
        for group,rows in by_group.items():
            cached=[dict(row) for row in rows]
            if wrong_root and model_index==0 and group=="C7":
                cached[0]["grammar_root_id"]="grammar_99"
            manifest=extract_features(MetadataRuntime(model["id"]),cached,[1],"span_mean",tmp_path/"caches"/model["id"]/group)
            from bsg_next.io import file_sha256
            digest=file_sha256(tmp_path/"caches"/model["id"]/group/"manifest.json")
            pairs={(r["family_id"],r["orbit_id"]):r for r in rows if r["split"] in ("test","test_joint")}
            fields={"pred":{},"ctx":{}}
            for (family,orbit),row in sorted(pairs.items()):
                base=3. if row["grammar_root_id"]==nuisance else 0.
                entity=(.2 if orbit.endswith("001") else -.2) if group=="S3_role" else 0.
                for endpoint,multiplier in (("pred",1.),("ctx",2.)):
                    value=base*multiplier+entity+(model_index-.5)*.04
                    fields[endpoint][family+"|"+orbit]=value
                    aggregate[group][endpoint][family,orbit].append(value)
            if leak and model_index==0 and group=="C7":
                key=next(iter(fields["pred"]))
                orbit=key.split("|")[1]
                fields["pred"]["structural_train_000|"+orbit]=fields["pred"].pop(key)
            cells.append({"group":group,"test_split":"test_joint" if group=="S3_role" else "test",
                "q":2,"is_primary_dimension":True,"model":manifest["model"],"layer":"1",
                "cache_identity":digest,"eligible":True,"family_orbit_prediction":fields["pred"],
                "family_orbit_context":fields["ctx"]})
    summary={"mode":"all","panel_complete":True,"family_contrasts":{},"S3":{}}
    for group,entry in aggregate.items():
        averaged={ep:{key:float(np.mean(values)) for key,values in pairs.items()} for ep,pairs in entry.items()}
        families=sorted({f for f,o in averaged["pred"]})
        orbits=sorted({o for f,o in averaged["pred"]})
        if group=="S3_role":
            summary["S3"]={"families":families,"orbits":orbits,
                "decisions":{"contrasts":{ep:{"mean":float(np.mean(list(values.values())))} for ep,values in averaged.items()}}}
        else:
            summary["family_contrasts"][group]={"families":families,**{
                ep:[float(np.mean([value for (f,o),value in values.items() if f==family])) for family in families]
                for ep,values in averaged.items()}}
    atomic_json(tmp_path/"primary"/"summary.json",summary)
    for i,cell in enumerate(cells):
        atomic_json(tmp_path/"primary"/"cells"/f"{i:02}.json",[cell])
    return cfg,nuisance,len(selected)


def test_grammar_nuisance_leave_root_out_preserves_crossed_entities_without_inference(tmp_path,monkeypatch):
    cfg,nuisance,nfamilies=fixture(tmp_path)
    # This analysis must not load even cached activation arrays, let alone a model.
    monkeypatch.setattr(np,"load",lambda *a,**k:(_ for _ in ()).throw(AssertionError("Activation payload read")))
    result=run_grammar_sensitivity(cfg,tmp_path/"analysis",tmp_path/"caches",tmp_path/"primary")
    assert result["new_model_forwards"]==0 and not result["activation_payloads_read"]
    assert not result["operators_refitted"] and not result["primary_decisions_changed"]
    assert result["independently_authored_grammar_population_inference"] is False
    for group,report in result["groups"].items():
        assert report["family_count"]==nfamilies and report["fixed_model_depth_conditions"]==2
        assert report["effects"]["pred"]>0
        assert report["leave_one_root_out"][nuisance]["effects"]["pred"]==pytest.approx(0,abs=1e-15)
        assert report["per_root"][nuisance]["effects"]["pred"]==pytest.approx(3.)
        assert report["leave_one_root_out"][nuisance]["effects"]["ctx"]==pytest.approx(0,abs=1e-15)
        assert all(report["leave_one_root_out"][other]["effects"]["pred"]>0 for other in report["per_root"] if other!=nuisance)
        if group=="S3_role":
            assert report["crossing_preserved"] and report["entity_orbit_count"]==2
            entity=report["leave_one_root_out"][nuisance]["entity_effects"]
            assert entity["main_triple_test_000"]["pred"]==pytest.approx(-.2)
            assert entity["main_triple_test_001"]["pred"]==pytest.approx(.2)
            assert all(part["entity_orbit_count"]==2 for part in report["leave_one_root_out"].values())


@pytest.mark.parametrize("kind",["wrong_root","leak"])
def test_grammar_rejects_wrong_root_or_non_test_contrast_units(tmp_path,kind):
    cfg,_,_=fixture(tmp_path,**{kind:True})
    with pytest.raises(ValueError,match="grammar root|non-test"):
        run_grammar_sensitivity(cfg,tmp_path/"analysis",tmp_path/"caches",tmp_path/"primary")
    assert not (tmp_path/"analysis"/"summary.json").exists()
