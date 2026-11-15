"""Cached-feature prospective structural evaluation, with train-only coordinates."""
from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
import json
import numpy as np
from .estimators import (LAMBDA_GRID, RETENTION_GRID, fit_ridge, select_action_ridge,
                         fit_procrustes, state_centroids, decode_states,
                         fit_joint_secondary, fit_potential_field, support_diagnostics,
                         consistent_action_images, inclusive_reference_rank, mse,
                         fit_local_displacement)
from .statistics import family_decisions, crossed_decisions, bootstrap_sensitivity
from .io import prepare_run, atomic_json, read_json, identity, file_sha256
from .groups import get_group


def fit_train_pca(h, train_mask, q=16):
    train = np.asarray(h[np.asarray(train_mask, bool)], dtype=np.float64)
    if not len(train): raise ValueError("No training features")
    center = train.mean(0); x = train-center
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    rank = int(np.sum(s > (s[0]*1e-12 if len(s) else 0)))
    if q > rank:
        raise ValueError(f"Primary PCA rank failure: requested {q}, observed {rank}")
    return center, vt[:int(q)].T, {"q": int(q), "observed_rank": rank,
            "singular_values": s.tolist(), "explained_variance": float(np.sum(s[:q]**2)/np.sum(s**2)),
            "fit_rows": int(len(train)), "fit_scope": "training_rows_only"}


def _state(row): return int(row.get("state_id", row.get("state", 0)))
def _orbit(row): return str(row.get("orbit_id", "single"))
def _family(row): return str(row["family_id"])
def _context(row): return (_family(row), str(row.get("paraphrase_id", row.get("template_id",0))), _orbit(row))


def pair_indices(rows, action_table):
    lookup = {}
    for i, row in enumerate(rows):
        key = (_context(row), _state(row))
        if key in lookup: raise ValueError(f"Duplicate context/state {key}")
        lookup[key] = i
    result = {}
    for g, action in action_table.items():
        src, dst = [], []
        for i, row in enumerate(rows):
            key = (_context(row), int(action[_state(row)]))
            if key not in lookup: raise ValueError(f"Incomplete orbit: {key}")
            j = lookup[key]
            if rows[j].get("split") != row.get("split"): raise ValueError("An orbit crosses partitions")
            src.append(i); dst.append(j)
        result[g] = (np.asarray(src, int), np.asarray(dst, int))
    return result


def difference_features(z, rows):
    contexts = defaultdict(dict)
    for i, row in enumerate(rows):
        contexts[(_family(row), _orbit(row), _state(row))][str(row.get("paraphrase_id",0))] = i
    out, outrows = [], []
    for key, members in sorted(contexts.items()):
        if len(members) != 2: raise ValueError(f"Expected exactly two paraphrases per family/orbit/state: {key}")
        a,b = [members[k] for k in sorted(members)]
        if rows[a]["split"] != rows[b]["split"]: raise ValueError("Paraphrase pair crosses split")
        out.append(z[a]-z[b]); row = dict(rows[a]); row["paraphrase_id"]="difference"; outrows.append(row)
    return np.asarray(out), outrows


def _pairs_for(z, rows, indices, split, changed=False):
    result={}
    for g,(a,b) in indices.items():
        mask=np.asarray([rows[i].get("split")==split for i in a])
        if changed: mask &= np.asarray([_state(rows[i])!=_state(rows[j]) for i,j in zip(a,b)])
        if np.any(mask): result[g]=(z[a[mask]],z[b[mask]])
    return result


def _fit_actions(z, rows, indices, grid=LAMBDA_GRID):
    train=_pairs_for(z,rows,indices,"train"); val=_pairs_for(z,rows,indices,"validation",True)
    selecttrain={g:train[g] for g in val}
    selected,meta=select_action_ridge(selecttrain,val,grid=grid)
    return {g:fit_ridge(*xy,meta["normalized_lambda"]) for g,xy in train.items()},meta


def _family_average(values):
    return {"|".join(k):float(np.mean(v)) for k,v in sorted(values.items())}


def _generator_words(group):
    words={group.identity:()}; queue=deque([group.identity])
    while queue:
        element=queue.popleft()
        for gen in group.generators:
            nxt=group.compose(gen,element)
            if nxt not in words: words[nxt]=words[element]+(gen,);queue.append(nxt)
    return words


def analyze_cell(z, rows, group, config=None, potential_features=None, test_split="test",
                 return_models=False):
    """Evaluate one model/depth; input z already has train-fitted coordinates.

    Rows must be complete orbits; test_split may be pilot/test_joint/etc.
    Primary O is independent linear ridge. Q is independently tuned difference
    ridge. Secondary estimators can never replace either primary.
    """
    cfg=config or {}; z=np.asarray(z,dtype=np.float64); group=get_group(group) if isinstance(group,str) else group
    states=np.asarray([_state(r) for r in rows]); train=np.asarray([r.get("split")=="train" for r in rows])
    val=np.asarray([r.get("split")=="validation" for r in rows])
    if not np.any(train) or not np.any(val): raise ValueError("Both train and validation rows are mandatory")
    actions={g:tuple(group.apply(g,x) for x in range(group.state_count)) for g in group.elements}
    indices=pair_indices(rows,actions); grid=cfg.get("lambda_grid",LAMBDA_GRID)
    models,selection=_fit_actions(z,rows,indices,grid)
    mu=state_centroids(z,states,train); murow=np.stack([mu[x] for x in states])
    scale=float(np.mean(np.sum((z[train]-z[train].mean(0))**2,1)))
    if scale<=1e-12: raise ValueError("Zero training activation energy")
    # Validation C*: oracle-source centroids with declared metadata asymmetry.
    baseline_scores={"centroid":[],"displacement":[]}
    for a in RETENTION_GRID: baseline_scores[f"retention_{a:g}"]=[]
    potential=None; potential_meta=None; fvalues=None
    if group.name=="S3_role":
        if potential_features is None: raise ValueError("S3 primary requires source input-embedding potential features")
        # The same metadata embedding repeats across contexts. Compress the
        # unique orbit/state vectors with train repetition weights, never make
        # a float64 copy of the multi-gigabyte repeated feature memmap.
        metadata_keys=[(_orbit(r),_state(r)) for r in rows]
        unique_keys=sorted(set(metadata_keys));first={}
        for i,key in enumerate(metadata_keys):first.setdefault(key,i)
        unique=np.asarray(potential_features[[first[k] for k in unique_keys]],float)
        positions={k:i for i,k in enumerate(unique_keys)};inverse=np.asarray([positions[k] for k in metadata_keys])
        weights=np.bincount(inverse[train],minlength=len(unique)).astype(float)
        pc=np.average(unique,axis=0,weights=weights)
        _,sing,vt=np.linalg.svd((unique-pc)*np.sqrt(weights[:,None]),full_matrices=False)
        rank=int(np.sum(sing>sing[0]*1e-12))
        q=min(int(cfg.get("potential_q",16)),rank)
        if q<1:raise ValueError("No train input-embedding feature variation")
        phi=((unique-pc)@vt[:q].T)[inverse]
        pinfo={"q":q,"observed_rank":rank,"fit_scope":"training_only_repetition_weighted_unique_metadata",
               "unique_vectors":len(unique),"fit_rows":int(train.sum())}
        potential,potential_meta=fit_potential_field(phi[train],z[train],phi[val],z[val],
                    widths=cfg.get("potential_widths",[32,128]),steps=cfg.get("potential_steps",500),seed=cfg.get("seed",42))
        potential_meta["feature_pca"]=pinfo;fvalues=potential.predict(phi);baseline_scores["potential_field"]=[]
    for g,(a,b) in indices.items():
        mask=val[a]&(states[a]!=states[b]);a,b=a[mask],b[mask]
        if not len(a):continue
        baseline_scores["centroid"].append(mse(z[b],murow[b]))
        baseline_scores["displacement"].append(mse(z[b],z[a]+murow[b]-murow[a]))
        for alpha in RETENTION_GRID:
            baseline_scores[f"retention_{alpha:g}"].append(mse(z[b],murow[b]+alpha*(z[a]-murow[a])))
        if fvalues is not None:baseline_scores["potential_field"].append(mse(z[b],z[a]+fvalues[b]-fvalues[a]))
    scores={k:float(np.mean(v)) for k,v in baseline_scores.items()}
    best=min(scores.values());chosen=next(k for k in scores if scores[k]<=best+1e-8)
    dz,drows=difference_features(z,rows);dindices=pair_indices(drows,actions)
    qmodels,qselection=_fit_actions(dz,drows,dindices,grid)
    dtrain=np.asarray([r["split"]=="train" for r in drows]);dval=np.asarray([r["split"]=="validation" for r in drows])
    energy=float(np.mean(np.sum(dz[dtrain]**2,1)));eligible=bool(energy/scale>=float(cfg.get("context_energy_floor",.05)))
    dalpha_scores=[]
    for alpha in RETENTION_GRID:
        errors=[]
        for g,(a,b) in dindices.items():
            mask=dval[a]&(np.asarray([_state(drows[i]) for i in a])!=np.asarray([_state(drows[i]) for i in b]))
            if np.any(mask):errors.append(mse(dz[b[mask]],alpha*dz[a[mask]]))
        dalpha_scores.append(float(np.mean(errors)))
    di=next(i for i,v in enumerate(dalpha_scores) if v<=min(dalpha_scores)+1e-8);dalpha=RETENTION_GRID[di]
    secondary={};secondary_meta={}
    requested=cfg.get("secondary",["affine","procrustes","residual","generator"])
    trainpairs=_pairs_for(z,rows,indices,"train");vpairs=_pairs_for(z,rows,indices,"validation",True)
    local=None
    if "local" in requested:
        local_validation={}
        for g,(a,b) in indices.items():
            mask=val[a]&(states[a]!=states[b]);a,b=a[mask],b[mask]
            if len(a):local_validation[g]=(z[a],z[b],states[a])
        local=fit_local_displacement(z[train],states[train],{g:xy[1] for g,xy in trainpairs.items()},
                                     local_validation,chunk_size=cfg.get("local_chunk_size",128),
                                     max_scale_pairs=cfg.get("local_scale_pairs",4096))
        secondary_meta["local_displacement"]=local.metadata
    if "affine" in requested:
        fitted,meta=select_action_ridge({g:trainpairs[g] for g in vpairs},vpairs,grid,True)
        secondary["affine"]={g:fit_ridge(*trainpairs[g],meta["normalized_lambda"],affine=True) for g in trainpairs};secondary_meta["affine"]=meta
    if "procrustes" in requested:secondary["procrustes"]={g:fit_procrustes(*v) for g,v in trainpairs.items()}
    if "residual" in requested:
        rm,rmeta=_fit_actions(z-murow,rows,indices,grid);secondary["residual"]=rm;secondary_meta["residual"]=rmeta
    if "joint" in requested:
        keys=list(trainpairs);product=np.asarray([[keys.index(group.compose(h,g)) for h in keys] for g in keys])
        jm,jmeta=fit_joint_secondary(trainpairs,models,product,keys.index(group.identity),steps=cfg.get("joint_steps",500),
                                    seed=cfg.get("seed",42),device=cfg.get("joint_device","cpu"))
        secondary["joint"]=jm;secondary_meta["joint"]=jmeta
    generator_ops={};words=_generator_words(group)
    if "generator" in requested:
        gens=set(group.generators)
        gm,gmeta=select_action_ridge({g:trainpairs[g] for g in gens},{g:vpairs[g] for g in gens},grid)
        for g,word in words.items():
            matrix=np.eye(z.shape[1])
            for token in word:matrix=matrix@gm[token].coef
            generator_ops[g]=matrix
        secondary_meta["generator"]={**gmeta,"words":{str(g):list(w) for g,w in words.items()},"validation_scope":"generators_only"}
    predictions=defaultdict(list);family_pred=defaultdict(list);family_ctx=defaultdict(list);family_local=defaultdict(list)
    decoded=decode_states(z,mu)
    support_mu={};support_counts={}
    if cfg.get("equal_support",False) and group.name=="S3_role":
        if not any(r.get("split")=="support" for r in rows):
            raise ValueError("Requested equal-support S3 arm is missing its separately charged support cache")
        for orbit in sorted({_orbit(r) for r in rows if r.get("split")=="support"}):
            for state in range(group.state_count):
                ids=[i for i,r in enumerate(rows) if r.get("split")=="support" and _orbit(r)==orbit and _state(r)==state]
                if len(ids)!=4:raise ValueError("Equal-support arm requires exactly four support contexts/state")
                support_mu[orbit,state]=z[ids].mean(0);support_counts[f"{orbit}|{state}"]=len(ids)
    for g,(a,b) in indices.items():
        mask=np.asarray([rows[i].get("split")==test_split for i in a])&(states[a]!=states[b]);a,b=a[mask],b[mask]
        if not len(a):continue
        predicted={"primary_operator":models[g].predict(z[a]),"identity":z[a],"centroid":murow[b],
                   "displacement":z[a]+murow[b]-murow[a]}
        for alpha in RETENTION_GRID:predicted[f"retention_{alpha:g}"]=murow[b]+alpha*(z[a]-murow[a])
        if fvalues is not None:predicted["potential_field"]=z[a]+fvalues[b]-fvalues[a]
        predicted["decoded_displacement"]=np.stack([z[i]+mu[group.apply(g,int(decoded[i]))]-mu[int(decoded[i])] for i in a])
        if local is not None:predicted["local_displacement"]=local.predict(z[a],states[a],g)
        for name,maps in secondary.items():
            predicted[name]=murow[b]+maps[g].predict(z[a]-murow[a]) if name=="residual" else maps[g].predict(z[a])
        if g in generator_ops and g not in group.generators and g!=group.identity:
            predicted["generator_unseen_actual_target"]=z[a]@generator_ops[g]
        if support_mu and all((_orbit(rows[i]),_state(rows[i])) in support_mu for i in a):
            sx=np.stack([support_mu[_orbit(rows[i]),_state(rows[i])] for i in a]);sy=np.stack([support_mu[_orbit(rows[j]),_state(rows[j])] for j in b])
            predicted["support_centroid"]=sy;predicted["support_displacement"]=z[a]+sy-sx
        losses={k:np.sum((z[b]-v)**2,1)/scale for k,v in predicted.items()}
        for k,v in losses.items():predictions[k].extend(v.tolist())
        for k,i in enumerate(a):family_pred[(_family(rows[i]),_orbit(rows[i]))].append(losses[chosen][k]-losses["primary_operator"][k])
        if local is not None:
            for k,i in enumerate(a):family_local[(_family(rows[i]),_orbit(rows[i]))].append(losses["local_displacement"][k]-losses["primary_operator"][k])
    for g,(a,b) in dindices.items():
        mask=np.asarray([drows[i].get("split")==test_split for i in a])&np.asarray([_state(drows[i])!=_state(drows[j]) for i,j in zip(a,b)])
        a,b=a[mask],b[mask]
        if not len(a):continue
        delta=(np.sum((dz[b]-dalpha*dz[a])**2,1)-np.sum((dz[b]-qmodels[g].predict(dz[a]))**2,1))/max(energy,1e-12)
        for value,i in zip(delta,a):family_ctx[(_family(drows[i]),_orbit(drows[i]))].append(float(value))
    output={"group":group.name,"test_split":test_split,"primary":"independent_linear_ridge",
            "context_primary":"independent_difference_ridge","primary_baseline":chosen,
            "metadata_asymmetry":"oracle-source category/potential controls versus z,g-only operator",
            "coordinate_dimension":z.shape[1],"training_scale":scale,"context_energy":energy,
            "context_energy_ratio":energy/scale,"eligible":eligible,"validation_baselines":scores,
            "selection":selection,"context_selection":qselection,"context_alpha":float(dalpha),
            "secondary_selection":secondary_meta,"potential":potential_meta,
            "mean_losses":{k:float(np.mean(v)) for k,v in predictions.items()},
            "family_orbit_prediction":_family_average(family_pred),"family_orbit_context":_family_average(family_ctx),
            "secondary_family_orbit_prediction":{"local_displacement_minus_primary":_family_average(family_local)} if local is not None else {},
            "secondary_inference_scope":"descriptive contrasts only; primary eight hypotheses unchanged; no nonlinear-comparison calibration claim",
            "support_examples":support_counts,"operator_diagnostics":{str(g):m.diagnostics for g,m in models.items()},
            "effective_dof":{str(g):m.effective_dof for g,m in models.items()},
            "support":support_diagnostics(z[train])}
    if cfg.get("legal_action_reference",False) and group.name!="S3_role":
        output["legal_action_reference"]=action_reference(z,rows,group,cfg,test_split)
    if return_models:output["_fitted"]={"operators":models,"difference_operators":qmodels,"centroids":mu,"local_displacement":local,
                                                       "context_alpha":dalpha,"baseline":chosen}
    return output


def action_reference(z,rows,group,config=None,test_split="test"):
    cfg=config or {};images=consistent_action_images(group.name)
    def score(actions):
        indices=pair_indices(rows,dict(enumerate(actions)));models,_=_fit_actions(z,rows,indices,cfg.get("lambda_grid",LAMBDA_GRID))
        values=[]
        for g,(a,b) in indices.items():
            mask=np.asarray([rows[i].get("split")==test_split for i in a])
            if np.any(mask):values.append(mse(z[b[mask]],models[g].predict(z[a[mask]])))
        return float(np.mean(values))
    canonical=tuple(sorted(tuple(group.apply(g,x) for x in range(group.state_count)) for g in group.elements))
    values=[score(actions) for actions in images];observed=values[images.index(canonical)]
    return {"image_count":len(images),"scores":values,"canonical_score":observed,
            "inclusive_tail_rank":inclusive_reference_rank(observed,values),
            "semantic_p_value":False,"reference_law":"uniform image registration; not assigned to natural semantics",
            "automorphism_tie_multiplicity":6 if group.name=="C7" else 4}


def _collect_paths(config,cache_root):
    cfg=config.get("structural",{});paths=cfg.get("cache_manifests",cfg.get("caches",[]))
    if cache_root:
        root=Path(cache_root)
        paths=[root] if (root/"manifest.json").exists() else sorted({p.parent for p in root.rglob("manifest.json")})
    if not paths:raise ValueError("No activation cache manifests supplied")
    return [Path(p).parent if Path(p).is_file() else Path(p) for p in paths]


def run_structural(config,output,cache_root=None,mode="all",**kwargs):
    from .extraction import read_cache
    cfg=dict(config.get("structural",{}));paths=_collect_paths(config,cache_root)
    from .estimators import numeric_code_identity
    provenance={str(p):file_sha256(p/"manifest.json") for p in paths}
    provenance["numeric_code_identity"]=numeric_code_identity()
    prepare_run(output,config,f"structural_{mode}",provenance=provenance)
    if mode=="actions":cfg["legal_action_reference"]=True
    if mode=="primary":cfg["secondary"]=[]
    if mode=="pilot":cfg["equal_support"]=False
    results=[]
    for cache in paths:
        rows,arrays,manifest=read_cache(cache)
        domain=manifest.get("domain",manifest.get("task",rows[0].get("domain")))
        group=get_group(domain);train=np.asarray([r.get("split")=="train" for r in rows])
        potential=None
        if group.name=="S3_role":
            feature=manifest.get("potential_features")
            if not feature:raise ValueError("S3 cache missing registered input-embedding potential_features")
            fp=cache/feature["path"]
            if feature.get("sha256") and file_sha256(fp)!=feature["sha256"]:raise ValueError("Potential feature hash mismatch")
            potential=np.load(fp,mmap_mode="r")
        splits=["pilot"] if mode=="pilot" else (["test_joint","test_context","test_entity"] if group.name=="S3_role" else ["test"])
        for layer,h in arrays.items():
            qs=[int(cfg.get("q_primary",16))]+[int(q) for q in cfg.get("q_secondary",[]) if int(q)!=int(cfg.get("q_primary",16))]
            if mode=="pilot":qs=qs[:1]
            for q in qs:
                key=identity({"cache":provenance[str(cache)],"layer":str(layer),"mode":mode,"q":q})[:20]
                target=Path(output)/"cells"/(key+".json")
                if target.exists():results.extend(read_json(target));continue
                try:
                    center,basis,pca=fit_train_pca(h,train,q)
                except ValueError as exc:
                    if q==int(cfg.get("q_primary",16)):raise
                    outputs=[{"status":"secondary_rank_failure","q":q,"layer":str(layer),"is_primary_dimension":False,"error":str(exc)}]
                    atomic_json(target,outputs);results.extend(outputs);continue
                z=(np.asarray(h,dtype=np.float64)-center)@basis
                outputs=[]
                for split in splits:
                    if not any(r.get("split")==split for r in rows):continue
                    value=analyze_cell(z,rows,group,cfg,potential,split)
                    value.update({"cache_identity":provenance[str(cache)],"model":manifest.get("model"),"layer":str(layer),"pca":pca,"q":q,"is_primary_dimension":q==int(cfg.get("q_primary",16))})
                    outputs.append(value)
                atomic_json(target,outputs);results.extend(outputs)
    # Average fixed model/depth conditions inside family/orbit before inference.
    grouped=defaultdict(lambda: {"pred":defaultdict(list),"ctx":defaultdict(list),"eligible":True})
    for r in results:
        if not r.get("is_primary_dimension",True) or r.get("status")=="secondary_rank_failure":continue
        if r["test_split"] not in {"test","test_joint","pilot"}:continue
        entry=grouped[r["group"]];entry["eligible"] &= r["eligible"]
        for endpoint,field in (("pred","family_orbit_prediction"),("ctx","family_orbit_context")):
            for k,v in r[field].items():entry[endpoint][k].append(v)
    expected_models=[m for m in config.get("models",[]) if m.get("structural",True)]
    wanted_groups={get_group(g).name for g in cfg.get("groups",["C7","D4","S3"])}
    panel_complete=bool(expected_models)
    for wanted in wanted_groups:
        observed=set()
        for record in results:
            if record.get("group")!=wanted or not record.get("is_primary_dimension",True):continue
            model=record.get("model",{});model_id=model.get("id",model.get("model_id")) if isinstance(model,dict) else model
            observed.add((model_id,str(record.get("layer"))))
        for model in expected_models:
            for layer in model.get("layers",[]):
                panel_complete &= (model["id"],str(layer)) in observed
    main={};s3={};pilot_columns=[]
    for group,entry in grouped.items():
        expected_count=sum(len(m.get("layers",[])) for m in expected_models)
        if panel_complete and any(len(v)!=expected_count for ep in ("pred","ctx") for v in entry[ep].values()):
            raise ValueError("Unequal/duplicate model-depth coverage within a family/orbit; provide one merged cache per model/task")
        keys=sorted(entry["pred"])
        averaged={ep:{k:float(np.mean(v)) for k,v in entry[ep].items()} for ep in ("pred","ctx")}
        if group=="S3_role":
            families=sorted({k.split("|")[0] for k in keys});orbits=sorted({k.split("|")[1] for k in keys})
            table={ep:np.asarray([[averaged[ep][f+"|"+o] for o in orbits] for f in families]) for ep in averaged}
            st=config.get("statistics",{})
            s3={"decisions":crossed_decisions(table,alpha=st.get("s3_alpha",.01),margin=st.get("margin",.01),equivalence=st.get("equivalence",.005),eligible=entry["eligible"]) if panel_complete and mode!="pilot" else None,"families":families,"orbits":orbits,"eligible":entry["eligible"]}
            if mode=="pilot":
                np.savez(Path(output)/"pilot_crossed_contrasts.npz",pred=table["pred"],ctx=table["ctx"],eligible=entry["eligible"],
                         families=np.asarray(families,dtype=str),orbits=np.asarray(orbits,dtype=str))
                s3["pilot_crossed_contrasts_path"]="pilot_crossed_contrasts.npz"
            if mode!="pilot" and cfg.get("bootstrap_sensitivity",False):s3["bootstrap"]={ep:bootstrap_sensitivity(v,replicates=cfg.get("bootstrap_replicates",10000),crossed=True) for ep,v in table.items()}
        else:
            byfamily={ep:defaultdict(list) for ep in averaged}
            for ep in averaged:
                for k,v in averaged[ep].items():byfamily[ep][k.split("|")[0]].append(v)
            families=sorted(byfamily["pred"])
            main[group]={ep:[float(np.mean(byfamily[ep][f])) for f in families] for ep in byfamily}
            main[group].update({"eligible":entry["eligible"],"families":families})
    summary={"cells":len(results),"mode":mode,"family_contrasts":main,"S3":s3,
             "ineligible_cells":[{k:r.get(k) for k in ("group","model","layer","q","context_energy_ratio")} for r in results if r.get("is_primary_dimension",True) and r.get("eligible") is False],
             "panel_complete":panel_complete,"formal_inference_enabled":panel_complete and mode!="pilot",
             "decisions":family_decisions(main,alpha=config.get("statistics",{}).get("main_alpha",.04),margin=config.get("statistics",{}).get("margin",.01),equivalence=config.get("statistics",{}).get("equivalence",.005)) if main and panel_complete and mode!="pilot" else None}
    local_grouped=defaultdict(lambda:defaultdict(list))
    for record in results:
        if not record.get("is_primary_dimension",True) or record.get("test_split") not in {"test","test_joint","pilot"}:continue
        for key,value in record.get("secondary_family_orbit_prediction",{}).get("local_displacement_minus_primary",{}).items():
            local_grouped[record["group"]][key].append(value)
    summary["secondary_local_displacement"]={group:{
        "family_orbit_local_minus_primary":{key:float(np.mean(values)) for key,values in sorted(entries.items())},
        "fixed_model_depth_conditions_averaged_within_family_orbit":True,"formal_inference":False,
        "scope":"secondary oracle-source local field; not part of the calibrated eight primary hypotheses"}
        for group,entries in local_grouped.items()}
    if mode=="pilot" and set(main)=={"C7","D4_corner"}:
        a,b=main["C7"],main["D4_corner"]
        if a["families"]==b["families"]:
            pilot=np.column_stack([a["pred"],a["ctx"],b["pred"],b["ctx"]])
            np.save(Path(output)/"pilot_contrasts.npy",pilot);summary["pilot_contrasts_path"]="pilot_contrasts.npy"
    if mode!="pilot" and cfg.get("bootstrap_sensitivity",False):
        summary["bootstrap"]={g:{ep:bootstrap_sensitivity(e[ep],replicates=cfg.get("bootstrap_replicates",10000)) for ep in ("pred","ctx")} for g,e in main.items()}
    atomic_json(Path(output)/"summary.json",summary);return summary
