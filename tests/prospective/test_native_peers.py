"""CPU-only scientific-contract tests; no checkpoint downloads or real findings."""
import json
import re

import numpy as np
import pytest

from bsg_next.groups import get_group,DAYS
from bsg_next.data import build_native_dataset
from bsg_next.native import (run_native,native_decisions,native_power,LocalDisplacement,
                             _native_power_pass,_domain,_build,normalized_config,run_rebinding_stage)
from bsg_next.peers import (functional_anova,complete_factorial_array,anova_diagnostics,
                            fit_calendar_plane,fit_manifold,build_native_factorial,run_factorial_stage,
                            run_reproduction_anchor)
from bsg_next.crossmodel import (hu_transport,compare_pair,fit_intertwiner,intertwiner_scores,
                                 congruity,validate_action_permutation)
from bsg_next.io import identity,atomic_json


class SemanticRuntime:
    """An explicit synthetic state carrier, not evidence about language models.

    Exact simulator parsing and one-hot carriers make all group operations
    known. Closing invalidates this object, exposing resource lifecycle bugs.
    """
    def __init__(self,model,runtime):
        self.model_id=model.get("model_id",model.get("id","mock"));self.revision="synthetic-v1"
        self.n_layers=4;self.closed=False;self.registry={};self.calls=0

    def _record(self,text):
        if text in self.registry:return self.registry[text]
        domain="C7" if "initial weekday is " in text else "D4_free8"
        group=get_group(domain)
        initial=next((i for i,s in enumerate(group.state_names) if ("initial weekday is "+s+".") in text or ("is "+s+".") in text),None)
        if initial is None:raise ValueError("Mock only accepts registered native prompts")
        def word(part):
            items=part.split(", then ")
            result=[]
            for item in items:
                if item=="do nothing":continue
                if group.name=="C7":result.append(1 if item=="advance one weekday" else 6)
                else:result.append((1,0) if item=="rotate clockwise by one quarter-turn" else (0,1))
            return result
        prefix=word(text.split("Prefix program: ")[1].split(". Continue")[0])
        remaining=word(text.split("Remaining program: ")[1].split(".")[0])
        intermediate=group.apply_word(initial,prefix)
        row={"domain":domain,"initial_state":initial,"intermediate_state":intermediate,
             "prefix_elements":prefix,"continuation_elements":remaining,
             "answer":group.state_names[group.apply_word(intermediate,remaining)],"text":text}
        self.registry[text]=row;return row

    def vector(self,row,readout):
        group=get_group(_domain(row["domain"]))
        if row.get("task")=="rebinding":state=int(row["state_id"])
        else:state=group.apply_word(row["intermediate_state"],row["continuation_elements"]) if readout=="final_token" else int(row["intermediate_state"])
        vector=np.zeros(16);vector[state]=1.
        # A small nuisance component makes residual-preservation testable.
        code=sum(map(ord,row["text"]))%23
        vector[10:12]=[code/1000.,(code%3)/1000.]
        return vector

    def features(self,records,layers,readout="carrier_span",**kwargs):
        if self.closed:raise RuntimeError("Closed runtime reused")
        for row in records:self.registry[row["text"]]=row
        array=np.stack([self.vector(r,readout) for r in records]).astype(np.float32)
        yield list(range(len(records))),{layer:array.copy() for layer in layers}

    def score(self,prompt,candidates,patch=None,generation=False,max_new_tokens=48,**kwargs):
        if self.closed:raise RuntimeError("Closed runtime reused")
        self.calls+=1;row=self._record(prompt);group=get_group(_domain(row["domain"]))
        final=patch is not None and patch.span is None
        readout="final_token" if final else "carrier_span"
        h=self.vector(row,readout)
        if patch is not None:
            if patch.delta is not None:h=h+patch.dose*np.asarray(patch.delta)
            elif patch.replacement_mean is not None:h=np.asarray(patch.replacement_mean)
        state=int(np.argmax(h[:group.state_count]))
        if row.get("task")=="rebinding":
            assignment=group.elements[state];answer=row["entities"][assignment[row.get("query_role",0)]]
        else:
            target=state if final else group.apply_word(state,row["continuation_elements"])
            answer=group.state_names[target]
        if "Repeat only the initial" in prompt:answer=group.state_names[row["initial_state"]]
        if "report the first item in this list" in prompt:answer="copper"
        if "Report only the case reference code" in prompt:answer=re.search(r"R\d{8}",prompt).group()
        index=candidates.index(answer) if answer in candidates else 0
        scores=np.full(len(candidates),-8.);scores[index]=-.05
        probability=np.full(len(candidates),.01);probability[index]=.90
        return {"prediction":candidates[index],"generated_text":answer if generation else None,
                "candidate_logprobs":scores.tolist(),"candidate_first_token_logits":scores.tolist(),
                "candidate_first_token_probabilities":probability.tolist(),"candidate_first_token_ids":list(range(len(candidates))),
                "other_first_token_probability":float(1-probability.sum()),
                "forward_counts":{"candidate_forwards":len(candidates),"diagnostic_prefill_forwards":1,"generation_forwards":int(generation)}}

    def score_components(self,prompt,candidates,components,donor_prompt=None,generation=False,max_new_tokens=48):
        if self.closed:raise RuntimeError("Closed runtime reused")
        row=self._record(prompt)
        if components[0].get("mode")=="zero":
            return {"prediction":candidates[0],"generated_text":candidates[0],"candidate_logprobs":[-1.]*len(candidates)}
        donor=self._record(donor_prompt);source=self.vector(row,"carrier_span");target=self.vector(donor,"carrier_span")
        from bsg_next.hf import PatchSpec
        marker=prompt.index("Continue")
        return self.score(prompt,candidates,PatchSpec(layer=components[0]["layer"],span=[marker,marker+8],delta=target-source),generation=generation)

    def close(self):self.closed=True


def mock_config(**native):
    return {"technical_test":True,"models":[{"id":"mock-a","revision":"mock-v1","num_hidden_layers":4},
                                                {"id":"mock-b","revision":"mock-v1","num_hidden_layers":4}],
            "native":{"models":["mock-a","mock-b"],"domains":["C7","D4_free"],
                      "families":{"train":4,"validation":2,"test":2,"pilot":2},"rank":8,
                      "sites":{"early":[.5],"late":[1.]},"dose_sensitivity":False,
                      "generation":True,"include_peers":False,
                      "power":{"replicates":12,"test_families":[32],"correlations":[0.],"stress":["normal"]},**native}}


def test_native_prepare_then_confirm_never_reuses_closed_runtime(tmp_path):
    instances=[]
    def factory(m,c):
        rt=SemanticRuntime(m,c);instances.append(rt);return rt
    config=mock_config(include_operations=True,include_pathway=True,include_shams=True)
    prepared=run_native(config,tmp_path/"native",mode="prepare",runtime_factory=factory)
    assert prepared["status"]=="prepared_no_test_outcomes"
    assert all(rt.closed for rt in instances)
    assert not list((tmp_path/"native").rglob("test_evaluations.jsonl"))
    frozen=json.loads((tmp_path/"native"/"selection_frozen.json").read_text())
    assert frozen["technical_run"] and frozen["frozen_before_test_activations"]
    assert all("operation_alignment" in cell and "pathway_selection" in cell for cell in frozen["cells"].values())
    result=run_native(config,tmp_path/"native",mode="confirm",runtime_factory=factory)
    assert result["technical_run"] and result["status"]=="completed" and len(result["cells"])==4
    assert all(rt.closed for rt in instances)
    for cell in result["cells"].values():
        assert cell["all_test_items_scored"] and cell["rows"]==48
        assert cell["operation_joint"]["evaluations"]==96
        assert cell["pathways"]["evaluations"]==48*6
        assert cell["shams"]["evaluations"]==48*3*5


def test_native_failed_run_closes_every_runtime(tmp_path):
    instances=[]
    def broken(m,c):
        rt=SemanticRuntime(m,c);instances.append(rt)
        def fail(*a,**k):raise RuntimeError("intentional feature failure")
        rt.features=fail;return rt
    with pytest.raises(RuntimeError,match="intentional"):
        run_native(mock_config(),tmp_path/"failure",runtime_factory=broken)
    assert instances and all(rt.closed for rt in instances)


def test_secondary_rank_panel_frozen_at_primary_sites_and_all_long_cases_scored(tmp_path):
    config=mock_config(rank=16,rank_sensitivity=[2,4,8],include_peers=True)
    directory=tmp_path/"rank-panel";made=[]
    def factory(model,runtime):
        value=SemanticRuntime(model,runtime);made.append(value);return value
    run_native(config,directory,mode="prepare",runtime_factory=factory)
    frozen=json.loads((directory/"selection_frozen.json").read_text())
    for cell in frozen["cells"].values():
        assert cell["selected"]["intermediate"]["fit"]["pca"]["requested_rank"]==16
        panel=cell["rank_sensitivity"]
        assert panel["frozen_before_test"] and panel["primary_rank"]==16
        assert set(panel["ranks"])=={"2","4","8"}
        for q,sites in panel["ranks"].items():
            for site,entry in sites.items():
                primary=cell["selected"][site]
                assert (entry["layer"],entry["readout"])==(primary["layer"],primary["readout"])
                assert entry["requested_rank"]==int(q)==entry["actual_rank"]
                assert not entry["validation_selection_uses_transfer"]
    assert not list(directory.rglob("test_intermediate.jsonl"))
    result=run_native(config,directory,mode="confirm",runtime_factory=factory)
    assert len(result["decisions"]["contrasts"])==4
    for name,cell in result["cells"].items():
        assert cell["rows"]==48 # primary panel was not replaced by the rank study.
        assert cell["rank_sensitivity"]["recipients_per_site_rank"]==32
        assert not cell["rank_sensitivity"]["primary_power_or_decision_changed"]
        for q,sites in cell["rank_sensitivity"]["ranks"].items():
            for site,summary in sites.items():
                assert summary["recipients"]==32 and summary["actual_rank"]==int(q)
                assert summary["inferential_status"]=="secondary_descriptive_no_confirmatory_claim"
                path=directory/name/"rank_sensitivity"/f"q{q}"/f"test_{site}.jsonl"
                rows=[json.loads(line) for line in path.read_text().splitlines()]
                assert len(rows)==summary["evaluations"]
                assert {row["word_split"] for row in rows}=={"transfer"}
                assert all(row["secondary"] and row["requested_rank"]==int(q) for row in rows)
                assert len({row["record_id"] for row in rows})==32
                if name.endswith("C7") and q=="2":assert "calendar_plane_matched" in summary["methods"]
    assert all(runtime.closed for runtime in made)
    # A deliberately corrupted test fixture recomputes only its outer hash;
    # independent reconstruction must still catch changed fitted metadata.
    frozen["cells"]["m0.C7"]["rank_sensitivity"]["ranks"]["2"]["intermediate"]["fit"]["pca"]["rank"]=1
    frozen["selection_identity"]=identity({k:v for k,v in frozen.items() if k!="selection_identity"})
    atomic_json(directory/"selection_frozen.json",frozen)
    with pytest.raises(ValueError,match="rank sensitivity construction fit changed"):
        run_native(config,directory,mode="confirm",runtime_factory=factory)
    assert all(runtime.closed for runtime in made)


def test_native_selected_48_family_pool_is_not_truncated_to_configured32():
    cfg=normalized_config(mock_config(families={"train":2,"validation":2,"test":32,"pilot":2}))
    rows=_build(cfg,"C7","test",test_families=48)
    assert len({r["family_id"] for r in rows})==48


def test_native_family_t_intervals_and_entire_power_gate():
    x=np.linspace(.06,.14,32)
    decisions=native_decisions({str(i):x for i in range(4)})
    assert decisions["all_four_pass"]
    assert all(r["interval_level"]==.9875 for r in decisions["contrasts"].values())
    below=np.tile(np.linspace(.03,.049,32)[None,:,None],(3,1,4))
    assert not _native_power_pass(below).any()
    constant=native_decisions({str(i):np.full(32,.1) for i in range(4)})
    assert not constant["all_four_pass"]


def test_kernel_infinite_bandwidth_is_state_conditional_mean_delta():
    residuals={0:np.array([[0.,1.],[2.,3.]])}
    delta={1:{0:np.array([[1.,2.],[3.,4.]])}}
    local=LocalDisplacement(residuals,delta,np.zeros((1,2)),{1:np.inf},1.,{})
    assert np.allclose(local.predict(np.array([9.,8.]),0,1),[11.,11.])


def test_full_anova_reconstructs_and_rejects_incomplete_or_duplicate_cells():
    rng=np.random.default_rng(7);x=rng.normal(size=(3,2,4,6));terms=functional_anova(x)
    assert np.allclose(sum(terms.values()),x)
    assert sum(anova_diagnostics(x)["energy_fractions"].values())==pytest.approx(1.)
    rows=[{"A":a,"B":b,"C":c} for a in range(3) for b in range(2) for c in range(4)]
    with pytest.raises(ValueError,match="Incomplete"):complete_factorial_array(rows[:-1],x.reshape(-1,6)[:-1])
    with pytest.raises(ValueError,match="Duplicate"):complete_factorial_array(rows+[rows[0]],np.r_[x.reshape(-1,6),x.reshape(-1,6)[:1]])


def test_calendar_plane_preserves_complement_but_faithful_ablation_does_not():
    angle=np.tile(2*np.pi*np.arange(7)/7,3);states=np.tile(np.arange(7),3)
    x=np.c_[np.cos(angle),np.sin(angle),np.repeat([-2.,0.,2.],7)]
    model=fit_calendar_plane(x,states)
    sample=np.array([[1.,0.,5.]])
    assert model.predict(sample,[3],preserve=True)[0,2]==pytest.approx(5.)
    assert model.predict(sample,[3],preserve=False)[0,2]==pytest.approx(0.)
    spline=fit_manifold(x*np.array([1.,1.,.001]),states)
    assert np.allclose(spline.path(0,3,50)[[0,-1]],spline.centroids[[0,3]])
    with pytest.raises(ValueError,match="cyclic"):fit_manifold(np.ones((8,4)),np.arange(8),state_count=8)


def test_native_factorial_is_real_complete_three_factor_and_runs(tmp_path):
    cfg=mock_config();cfg["native"]["models"]=["mock-a"]
    for domain,n in (("C7",7),("D4_free",8)):
        rows=build_native_factorial(normalized_config(cfg),domain,"test")
        assert len(rows)==2*2*n*2*2
        first=[r for r in rows if r["family_id"]==rows[0]["family_id"] and r["paraphrase_id"]==0]
        assert len({(r["A"],r["B"],r["C"]) for r in first})==n*4
        assert all(len(r["continuation_elements"])==2 for r in rows)
    result=run_factorial_stage(cfg,tmp_path/"factorial",runtime_factory=SemanticRuntime)
    assert result["technical_run"] and len(result["cells"])==2
    for cell in result["cells"].values():
        assert cell["splits"]["test"]["evaluations"]==8*cell["splits"]["test"]["recipients"]


def test_rebinding_prepares_before_test_then_scores_all_crossed_arms(tmp_path):
    cfg=normalized_config(mock_config(rebinding={"max_orbits_per_entity_split":1}))
    runtime=SemanticRuntime(cfg["models"][0],{})
    prepared=run_rebinding_stage(runtime,cfg,cfg["models"][0],tmp_path/"rebinding",mode="prepare")
    assert prepared["status"]=="prepared_no_test_outcomes"
    assert not (tmp_path/"rebinding"/"test_evaluations.jsonl").exists()
    result=run_rebinding_stage(runtime,cfg,cfg["models"][0],tmp_path/"rebinding")
    assert set(result["arms"])=={"test_context","test_entity","test_joint"}
    assert result["evaluations"]==5*result["recipients"]


class CalendarRuntime(SemanticRuntime):
    def features(self,records,layers,readout="span_mean",**kwargs):
        for row in records:self.registry[row["text"]]=row
        values=[]
        for row in records:
            state=row["state_id"] if readout=="span_mean" else row["answer_state"]
            t=2*np.pi*state/7;values.append([np.cos(t),np.sin(t),0.,0.])
        yield list(range(len(records))),{layer:np.array(values,np.float32) for layer in layers}

    def score(self,prompt,candidates,patch=None,generation=False,**kwargs):
        if self.closed:raise RuntimeError("Closed calendar runtime reused")
        row=self.registry[prompt];source=row["answer_state"] if patch.span is None else row["state_id"]
        t=2*np.pi*source/7;h=np.array([np.cos(t),np.sin(t),0.,0.])+patch.delta
        angles=2*np.pi*np.arange(7)/7
        logits=3*np.stack((np.cos(angles),np.sin(angles)),1)@h[:2]
        if patch.span is not None:logits=np.roll(logits,row["offset"])
        full=np.r_[logits,0.];prob=np.exp(full-full.max());prob/=prob.sum()
        answer=candidates[int(np.argmax(logits))]
        return {"prediction":answer,"generated_text":answer if generation else None,"candidate_logprobs":logits.tolist(),
                "candidate_first_token_logits":logits.tolist(),"candidate_first_token_probabilities":prob[:-1].tolist(),
                "other_first_token_probability":float(prob[-1]),"candidate_first_token_ids":list(range(7))}


def test_wurgaft_anchor_executes_paths_and_closes_runtime(tmp_path):
    made=[]
    def factory(m,c):
        runtime=CalendarRuntime(m,c);made.append(runtime);return runtime
    cfg={"models":[{"id":"synthetic-calendar","revision":"mock"}],"peers":{"anchor":"wurgaft","technical_run":True,
          "layer":2,"max_pairs":1,"path_base_prompts":1,"path_targets_per_prompt":1,"path_points":5}}
    result=run_reproduction_anchor(cfg,tmp_path/"anchor",runtime_factory=factory)
    assert result["technical_run"] and result["path_evaluations"]==15 and len(result["path_summaries"])==3
    assert all(runtime.closed for runtime in made)


def test_hu_never_needs_destination_test_and_nulls_refit():
    rng=np.random.default_rng(5);source=rng.normal(size=(24,3));target=source@rng.normal(size=(3,4))
    train=np.arange(12);test=np.arange(16,24)
    expected=hu_transport(source,target[train],train,test)
    changed=target.copy();changed[test]=1000.
    assert np.array_equal(expected,hu_transport(source,changed[train],train,test))
    arrays={"source":source,"target":target,"source_displacements":source,"target_displacements":target,
            "train_mask":np.arange(24)<12,"validation_mask":(np.arange(24)>=12)&(np.arange(24)<16),"test_mask":np.arange(24)>=16}
    result=compare_pair(arrays,{"null_replicates":3})
    assert len(result["hu_refitted_weight_nulls"])==3 and "hu_gaussian_training_RMS_matched" in result
    assert result["methods"]["ridge_affine"]["mean_squared_error"]<1e-6


def test_heldout_intertwiner_and_congruity_do_not_fake_implementation_sets():
    theta=2*np.pi*np.arange(7)/7
    source=np.array([[[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]] for t in theta])
    a,meta=fit_intertwiner(source,source,[1,2])
    assert max(intertwiner_scores(a,source,source,[3,4,5,6]).values())<1e-8
    assert meta["rank_warning"]
    with pytest.raises(ValueError,match="actual implementations"):
        congruity(np.zeros((1,8,2)),np.zeros((1,8,2)),np.arange(8)<4,np.arange(8)>=4,preservation_a=[True],preservation_b=[True])
    multiplication=(np.arange(7)[:,None]+np.arange(7)[None,:])%7
    with pytest.raises(ValueError,match="preserve multiplication"):validate_action_permutation([1,0,2,3,4,5,6],multiplication)
