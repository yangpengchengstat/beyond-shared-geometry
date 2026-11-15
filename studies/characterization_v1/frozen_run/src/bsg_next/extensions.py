"""Conditional constructive experiments, separate from primary identification.

These routines measure an explicitly trained interface or an additional finite
action. They never describe interface training as a discovery in pretraining.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
import json
import os
import time
import numpy as np
from .io import prepare_run,atomic_json,read_json,identity,file_sha256
from .estimators import numeric_code_identity,select_action_ridge,LAMBDA_GRID
from .structural import analyze_cell,fit_train_pca,pair_indices,_pairs_for,_state
from .groups import get_group


@dataclass(frozen=True)
class ProductGroup:
    name: str="C4xC4"
    elements: tuple=tuple(product(range(4),repeat=2))
    identity: tuple=(0,0)
    generators: tuple=((1,0),(0,1))
    state_count: int=16
    state_names: tuple=tuple(f"({x},{y})" for x,y in product(range(4),repeat=2))
    @property
    def order(self):return 16
    def normalize(self,g):
        if isinstance(g,str):
            aliases={"e":(0,0),"r":(1,0),"s":(0,1)}
            if g not in aliases:raise ValueError("Unknown product generator")
            g=aliases[g]
        return (int(g[0])%4,int(g[1])%4)
    def compose(self,g,h):
        a,b=self.normalize(g);c,d=self.normalize(h);return ((a+c)%4,(b+d)%4)
    def inverse(self,g):a,b=self.normalize(g);return ((-a)%4,(-b)%4)
    def apply(self,g,state_id):
        if not 0<=int(state_id)<16:raise ValueError("Toroid state outside0..15")
        x,y=divmod(int(state_id),4);a,b=self.normalize(g);return ((x+a)%4)*4+(y+b)%4
    def evaluate_word(self,word):
        result=self.identity
        for g in word:result=self.compose(g,result)
        return result
    def action_table(self):return [[self.apply(g,x) for x in range(16)] for g in self.elements]
    def as_dict(self):return {"name":self.name,"elements":self.elements,"identity":self.identity,"generators":self.generators,"action_table":self.action_table()}


def build_product_dataset(seed=2027101700,splits=None):
    """Complete toroidal states; independently sampled scenario IDs are families."""
    splits=splits or {"train":48,"validation":16,"test":32}
    rng=np.random.default_rng(seed);records=[]
    for split,count in splits.items():
        for f in range(int(count)):
            scenario=int(rng.integers(0,2**53));family=f"toroid_{split}_{scenario:014x}"
            for p in range(2):
                for x,y in product(range(4),repeat=2):
                    target=f"({x},{y})"
                    prefix=(f"In toroidal board record {scenario}, the grid has four columns and four rows. " if p==0 else
                            f"A marker is logged in board scenario {scenario}. Both coordinates wrap modulo four. ")
                    text=prefix+f"The marker coordinate is {target}."
                    state=4*x+y
                    records.append({"row_id":f"{family}_{p}_{state}","record_id":f"{family}_{p}_{state}",
                        "family_id":family,"paraphrase_id":p,"template_id":f"{family}_{p}","state_id":state,
                        "state":target,"split":split,"context_split":split,"orbit_id":"toroid",
                        "domain":"C4xC4","text":text,"target":target,"readout":"final_token",
                        "grammar_root_id":f"toroid_grammar_{p}","sampling_scope":"IID scenario identifiers within fixed toroid grammars"})
    return records


def run_product(config,output,runtime_factory=None):
    from .extraction import extract_features,read_cache
    from .hf import runtime_from_config
    cfg=config.get("product",config.get("extensions",{}).get("product",{}))
    if not config.get("execution",{}).get("allowed_model_execution",False) and runtime_factory is None:
        raise PermissionError("Product extraction requires explicit model-execution authorization")
    prepare_run(output,config,"product_action",provenance={"numeric_code_identity":numeric_code_identity(),"extension_sha256":file_sha256(__file__)})
    records=build_product_dataset(config.get("seed",2027101700),cfg.get("splits"))
    atomic_json(Path(output)/"dataset_manifest.json",{"records":records,"group":ProductGroup().as_dict(),"conditional_secondary":True})
    results=[]
    for spec in config.get("models",[]):
        if not spec.get("structural",True):continue
        runtime=(runtime_factory(spec) if runtime_factory else runtime_from_config(spec,config.get("runtime",{})))
        modelkey=identity(spec)[:16];cache=Path(output)/"cache"/modelkey
        extract_features(runtime,records,spec["layers"],"final_token",cache)
        rows,arrays,manifest=read_cache(cache)
        for layer,h in arrays.items():
            target=Path(output)/"results"/f"{modelkey}_{layer}.json"
            if target.exists():results.append(read_json(target));continue
            train=np.asarray([r["split"]=="train" for r in rows]);q=int(cfg.get("q",16))
            center,basis,pca=fit_train_pca(h,train,q);z=(np.asarray(h,float)-center)@basis
            controls={**config.get("structural",{}),"legal_action_reference":False,"equal_support":False}
            # Generic product defining relations are handled by the joint fitter.
            result=analyze_cell(z,rows,ProductGroup(),controls)
            result.update({"model":spec["id"],"layer":layer,"pca":pca,"conditional_secondary":True,
                           "not_primary_C7_D4_S3_family":True})
            atomic_json(target,result);results.append(result)
        if hasattr(runtime,"close"):runtime.close()
    summary={"group":"C4xC4","rows":len(records),"cells":len(results),"results":results,
             "scope":"conditional additional-domain experiment; no primary-family substitution"}
    atomic_json(Path(output)/"summary.json",summary);return summary


def _train_interface(h,rows,group,q,arm,weight,seed,steps=500,batch_size=256,device="cpu"):
    import torch
    torch.manual_seed(int(seed));rng=np.random.default_rng(seed)
    train=np.asarray([r["split"]=="train" for r in rows]);val=np.asarray([r["split"]=="validation" for r in rows])
    center,basis,pca=fit_train_pca(h,train,q)
    x=np.asarray(h,float)-center;scale=max(float(np.mean(np.sum(x[train]**2,1))),1e-12)
    states=np.asarray([_state(r) for r in rows]);d=x.shape[1]
    enc=torch.nn.Linear(d,q,bias=False,dtype=torch.float64,device=device)
    dec=torch.nn.Linear(q,d,bias=False,dtype=torch.float64,device=device)
    head=torch.nn.Linear(q,group.state_count,dtype=torch.float64,device=device)
    with torch.no_grad():enc.weight.copy_(torch.tensor(basis.T,device=device));dec.weight.copy_(torch.tensor(basis,device=device))
    operators=torch.nn.Parameter(torch.eye(q,dtype=torch.float64,device=device).repeat(len(group.elements),1,1))
    parameters=list(enc.parameters())+list(dec.parameters())+list(head.parameters())+[operators]
    optimizer=torch.optim.AdamW(parameters,lr=.002,weight_decay=1e-5)
    actions={g:tuple(group.apply(g,s) for s in range(group.state_count)) for g in group.elements}
    pairs=pair_indices(rows,actions);target_indices=np.stack([pairs[g][1] for g in group.elements])
    product_table=np.asarray([[group.elements.index(group.compose(b,a)) for b in group.elements] for a in group.elements])
    ids=np.flatnonzero(train);eye=torch.eye(q,dtype=torch.float64,device=device);start=time.perf_counter()
    for step in range(int(steps)):
        sampled=rng.choice(ids,size=min(int(batch_size),len(ids)),replace=True)
        action=rng.integers(len(group.elements),size=len(sampled));dest=target_indices[action,sampled]
        if arm=="complete_orbit_augmentation":sampled=dest  # complete-orbit duplication, not novel observations
        xt=torch.tensor(x[sampled],device=device);labels=torch.tensor(states[sampled],device=device)
        hidden=enc(xt);reconstruction=((dec(hidden)-xt)**2).sum(1).mean()/scale
        classification=torch.nn.functional.cross_entropy(head(hidden),labels)
        loss=reconstruction+classification
        if arm in {"equivariance_consistency","group_regularized"}:
            yt=torch.tensor(x[dest],device=device);gi=torch.tensor(action,device=device)
            transported=torch.bmm(hidden[:,None,:],operators[gi]).squeeze(1)
            actual_target=((dec(transported)-yt)**2).sum(1).mean()/scale
            loss=loss+float(weight)*actual_target
            if arm=="group_regularized":
                hom=torch.stack([((operators[i]@operators[j]-operators[int(product_table[i,j])])**2).mean()
                                  for i in range(len(group.elements)) for j in range(len(group.elements))]).mean()
                ident=((operators[group.elements.index(group.identity)]-eye)**2).mean()
                loss=loss+float(weight)*(hom+ident)
        optimizer.zero_grad();loss.backward();optimizer.step()
    with torch.no_grad():
        latent=enc(torch.tensor(x,device=device)).cpu().numpy()
        head_weight=head.weight.detach().cpu().numpy();head_bias=head.bias.detach().cpu().numpy()
        decoder=dec.weight.detach().cpu().numpy().T
        learned=operators.detach().cpu().numpy()
    if arm not in {"equivariance_consistency","group_regularized"}:
        fitted,_=select_action_ridge(_pairs_for(latent,rows,pairs,"train"),_pairs_for(latent,rows,pairs,"validation"),LAMBDA_GRID)
        learned=np.stack([fitted[g].coef for g in group.elements])
    def evaluate(split):
        losses=[];correct=[]
        for gi,g in enumerate(group.elements):
            a,b=pairs[g];mask=np.asarray([rows[i]["split"]==split for i in a])&(states[a]!=states[b]);a,b=a[mask],b[mask]
            if not len(a):continue
            moved=latent[a]@learned[gi];pred=moved@decoder+center
            losses.extend((np.sum((np.asarray(h)[b]-pred)**2,1)/scale).tolist())
            correct.extend((np.argmax(moved@head_weight.T+head_bias,1)==states[b]).tolist())
        return {"target_prediction_loss":float(np.mean(losses)),"classification_accuracy":float(np.mean(correct)),"target_rows":len(losses)}
    validation=evaluate("validation")
    test={split:evaluate(split) for split in sorted({r["split"] for r in rows}) if split.startswith("test")}
    if not test:raise ValueError("Interface evaluation needs held-out target rows")
    return {"arm":arm,"weight":float(weight),"seed":int(seed),"steps":int(steps),"validation":validation,"test":test,
            "validation_selection_score":validation["target_prediction_loss"]+1-validation["classification_accuracy"],
            "actual_q":q,"pca":pca,"training_rows":int(train.sum()),"wall_seconds":time.perf_counter()-start,
            "augmentation_disclosure":"complete-orbit observations duplicated; no new information" if arm=="complete_orbit_augmentation" else None,
            "parameter_count":sum(p.numel() for p in parameters),"claim":"constructive supervised interface; not native pretraining mechanism",
            "_parameters":{"encoder":enc.weight.detach().cpu().numpy().T,"decoder":decoder,"head_weight":head_weight,
                           "head_bias":head_bias,"operators":learned,"center":center}}


def run_interface_training(config,output,cache_root=None):
    from .extraction import read_cache
    from .structural import _collect_paths
    cfg=config.get("interface",config.get("extensions",{}).get("interface",{}));paths=_collect_paths(config,cache_root)
    prepare_run(output,config,"constructive_interface",provenance={"caches":{str(p):file_sha256(p/"manifest.json") for p in paths},
                "extension_sha256":file_sha256(__file__),"numeric_code_identity":numeric_code_identity()})
    selected=[]
    for path in paths:
        rows,arrays,manifest=read_cache(path);group=get_group(rows[0]["domain"])
        for layer,h in arrays.items():
            for seed in cfg.get("seeds",[101,303,909]):
                for arm in ("lowrank_reconstruction","complete_orbit_augmentation","equivariance_consistency","group_regularized"):
                    candidates=[]
                    weights=cfg.get("weights",[.01,.1,1.]) if arm in {"equivariance_consistency","group_regularized"} else [0.]
                    for weight in weights:
                        descriptor={"cache":manifest.get("cache_key",file_sha256(path/"manifest.json")),"layer":layer,"seed":seed,"arm":arm,"weight":weight}
                        target=Path(output)/"fits"/(identity(descriptor)[:20]+".json")
                        if target.exists():
                            value=read_json(target)
                            checkpoint=target.parent/value["checkpoint"]["path"]
                            if not checkpoint.exists() or file_sha256(checkpoint)!=value["checkpoint"]["sha256"]:
                                raise ValueError("Constructive interface checkpoint identity mismatch")
                        else:
                            value=_train_interface(h,rows,group,int(cfg.get("q",16)),arm,weight,seed,
                                    steps=cfg.get("steps",500),batch_size=cfg.get("batch_size",256),device=cfg.get("device","cpu"))
                            parameters=value.pop("_parameters");checkpoint=target.with_suffix(".npz")
                            temporary=checkpoint.with_suffix(".partial.npz")
                            np.savez_compressed(temporary,**parameters);os.replace(temporary,checkpoint)
                            value["checkpoint"]={"path":checkpoint.name,"sha256":file_sha256(checkpoint)}
                            value.update(descriptor);atomic_json(target,value)
                        candidates.append(value)
                    chosen=min(candidates,key=lambda v:(v["validation_selection_score"],v["weight"]))
                    selected.append(chosen)
    result={"selected":selected,"selection_scope":"validation only","equal_training_steps":True,
            "test_results_not_used_for_selection":True,"pretraining_mechanism_claim":False}
    atomic_json(Path(output)/"summary.json",result);return result


def run_extensions(config,output,cache_root=None):
    cfg=config.get("extensions",{});prepare_run(output,config,"conditional_extensions",provenance={"extension_sha256":file_sha256(__file__)})
    result={}
    if cfg.get("product_actions",False):result["product"]=run_product(config,Path(output)/"product")
    if cfg.get("interface_training",False):result["interface"]=run_interface_training(config,Path(output)/"interface",cache_root)
    atomic_json(Path(output)/"summary.json",result);return result
