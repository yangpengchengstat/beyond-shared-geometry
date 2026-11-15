#!/usr/bin/env python3
"""Render reviewed replay evidence. No model calls, fitting, calibration or inference."""
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import math
import statistics

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_review(root):
    review = json.loads((root/'release/reproduction_review.json').read_text())
    for key in ('automatic_promotion','original_flags_changed','independent_confirmation','new_statistical_significance_decisions'):
        if review[key] is not False:
            raise ValueError('Descriptive boundary violated: '+key)
    for name, entry in review['inputs'].items():
        p = root/name
        if Path(name).is_absolute() or '..' in Path(name).parts or p.is_symlink() or not p.resolve().is_relative_to(root.resolve()):
            raise ValueError('Unsafe reviewed input: '+name)
        if sha(p) != entry['sha256']:
            raise ValueError('Reviewed input changed: '+name)
    return review

def aggregate(rows, keys, metric, expected):
    bins=defaultdict(list)
    for r in rows:
        if not isinstance(r[metric],(int,float)) or not math.isfinite(r[metric]):
            raise ValueError('Non-finite observed metric')
        if r.get('invalid',0):
            raise ValueError('Invalid predictions must not be silently dropped')
        bins[tuple(r[k] for k in keys)].append(r)
    out=[]
    for key, group in sorted(bins.items()):
        ids=[(r['cell'],) if 'cell' in r else (r['cache'],r['layer'],r['seed']) for r in group]
        if len(ids)!=len(set(ids)) or len(ids)!=expected:
            raise ValueError('Missing/duplicate panel members: '+str(key))
        row={**dict(zip(keys,key)),'cells':len(group),'mean':statistics.fmean(r[metric] for r in group)}
        if 'parameter_bytes' in group[0]:
            row['mean_parameter_bytes']=statistics.fmean(r['parameter_bytes'] for r in group)
        out.append(row)
    return out

def select(rows, **where):
    v=[r for r in rows if all(r[k]==value for k,value in where.items())]
    if len(v)!=1: raise ValueError('Not exactly one aggregate: '+str(where))
    return v[0]['mean']

def table(path, header, rows, caption, label, layout):
    lines=[r'\begin{table}[htbp]\centering\small',r'\begin{tabular}{'+layout+r'}\toprule',header+r' \\',r'\midrule']
    lines += [' & '.join(row)+r' \\' for row in rows]
    lines += [r'\bottomrule\end{tabular}',r'\caption{'+caption+'}',r'\label{'+label+r'}\end{table}']
    path.write_text('\n'.join(lines)+'\n')

def build(root=ROOT):
    root=Path(root).resolve(); review=load_review(root); d=root/'results/reproduction_v1_review';g=root/'paper/generated'
    read=lambda n:json.loads((d/n).read_text())
    compact=aggregate(read('compact_rows.json'),['domain','registration','track','method','split'],'nmse',24)
    context=aggregate(read('context_rows.json'),['domain','registration','method','split'],'nmse',24)
    interface=aggregate(read('interface_rows.json'),['domain','arm','split'],'target_prediction_loss',72)
    classification=aggregate(read('interface_rows.json'),['domain','arm','split'],'classification_accuracy',72)
    summary={'aggregation':'Equal fixed model/readout means; interface also averages three technical seeds. No new p-values or confidence intervals.',
             'compact':compact,'context':context,'interface_prediction':interface,'interface_classification':classification}
    generated=[]
    def emit(name,header,rows,caption,label,layout):
        p=g/name;table(p,header,rows,caption,label,layout);generated.append(p)
    methods=[('ridge','Ridge'),('decoded_centroid','Decoded centroid'),('limited_support_budget_1','Limited support, $1\\times$'),('rff_budget_1','RFF, $1\\times$'),('nystrom_budget_1',r'Nystr\"om, $1\times$')]
    rows=[[label]+[f'{select(compact,domain=domain,registration="semantic",track="all_element",method=m,split="test"):.5f}' for domain in ['C7','D4_corner']] for m,label in methods]
    emit('reproduction_compact.tex',r'Method & $C_7$ & $D_4$ (corners)',rows,
         r'All-element changed-state prediction in the reviewed replay: equal action then equal cell means over 24 fixed model/readout cells per group; lower is better. $1\times$ denotes nominal incremental storage relative to ridge, not equal capacity or total RAM. Centroids additionally use construction labels and an action table. All budget and registration results are retained in the release.',
         'tab:reproduction-compact','lrr')
    rows=[]
    for m,label in [('identity_context','Identity'),('coordinate_fitted_operator_context','Coordinate-fitted operator'),('linear_transport_context','Context-fitted linear'),('oracle_category_retention_context','Oracle category retention')]:
        rows.append([label]+[f'{select(context,domain=domain,registration="semantic",method=m,split=split):.6f}' for domain,split in [('C7','test'),('D4_corner','test'),('S3_role','test_joint')]])
    rows.append(['Context-fitted, coherent action']+[f'{select(context,domain=domain,registration="consistent_swap01",method="linear_transport_context",split=split):.6f}' for domain,split in [('C7','test'),('D4_corner','test'),('S3_role','test_joint')]])
    emit('reproduction_context.tex',r'Method & $C_7$ & $D_4$ (corners) & $S_3$ (joint)',rows,
         'Context-difference transport on 24 fixed cells per stratum. Coherent swap01 defines a consistently relabeled alternative action; it is not an exchangeable semantic null. The context loss uses its own training-energy normalization, not the coordinate-prediction denominator.',
         'tab:reproduction-context','lrrr')
    rows=[]
    arms=[('lowrank_reconstruction','Reconstruction'),('complete_orbit_augmentation','Orbit augmentation'),('equivariance_consistency','Equivariance'),('group_regularized','Group regularization')]
    for arm,label in arms:
        rows.append([label]+[f'{select(interface,domain=domain,arm=arm,split=split):.6f}' for domain,split in [('C7','test'),('D4','test'),('S3','test_joint')]]+
                    [f'{select(classification,domain="S3",arm=arm,split="test_joint"):.4f}'])
    emit('reproduction_interface.tex',r'Objective & $C_7$ loss & $D_4$ loss & $S_3$ joint loss & $S_3$ joint acc.',rows,
         'Validation-selected supervised interfaces: 72 model/readout/seed cells per objective and domain, 500 steps per candidate. The loss is normalized full-hidden target-prediction error; it is not the compact-coordinate endpoint. State classification is an interface-head score, not language-model answer accuracy. Seed replicates are not independent datasets.',
         'tab:reproduction-interface','lrrrr')
    cross=read('crossmodel_summary.json');prod=read('product_summary.json')['descriptive_mean_losses'];facts=read('factorial_summary.json')
    rows=[]
    for r in cross:
        for method,values in r['methods'].items():
            short={'procrustes':'Procrustes','ridge_affine':'Ridge affine','hu_equation12_source_test_displacement_observed':'Hu Eq.~12 (source displacement observed)','intertwiner':'Constrained intertwiner'}[method]
            rows.append(['Word' if r['task']=='word_analyze' else 'Structural',short,f'{values["row_normalized_squared_error"]:.5f}',str(r['pairs'])])
    for method in ['primary_operator','joint','local_displacement']:
        rows.append([r'$C_4\times C_4$',method.replace('_',' '),f'{prod[method]:.5f}','24'])
    emit('reproduction_secondary.tex',r'Task & Method & Error & Conditions',rows,
         'Secondary fixed-panel summaries. Word and structural transport use row-normalized errors; the product endpoint has its own target-energy normalization. Values across tasks are not directly comparable. The Hu adaptation receives source-model test displacements. These are implemented adaptations, not certified peer-author reproductions.',
         'tab:reproduction-secondary','llrr')
    rows=[]
    for domain in ['C7','D4_corner']:
        for method,short in [('limited_support','Limited support'),('rff','RFF'),('nystrom',r'Nystr\"om')]:
            rows.append([r'$C_7$' if domain=='C7' else r'$D_4$',short]+[f'{select(compact,domain=domain,registration="semantic",track="all_element",method=method+"_budget_"+b,split="test"):.5f}' for b in ['0p5','1','2','4','8']])
    emit('reproduction_budgets.tex',r'Group & Estimator & $0.5\times$ & $1\times$ & $2\times$ & $4\times$ & $8\times$',rows,
         'All five declared nominal budgets for the all-element semantic track, equal action then equal cell means over 24 cells. This table does not choose a budget on the inspected test set. Generator-only and coherent-action results remain separate in the machine-readable release.',
         'tab:reproduction-budgets','llrrrrr')
    unseen=json.loads((root/'results/reproduction_action_review/generator_unseen_summary.json').read_text())
    rows=[]
    for method,label in methods:
        values=[]
        for domain in ['C7','D4_corner']:
            chosen=[x for x in unseen if (x['domain'],x['registration'],x['method'])==(domain,'semantic',method)]
            if len(chosen)!=1 or chosen[0]['cells']!=24 or chosen[0]['unseen_actions_per_cell']!=5:
                raise ValueError('Invalid unseen-only panel')
            values.append(f"{chosen[0]['unseen_equal_action_mean_nmse']:.6f}")
        rows.append([label]+values)
    emit('reproduction_unseen.tex',r'Method & $C_7$ & $D_4$ (corners)',rows,
         r'Generator-only fitting evaluated solely on the five excluded nonidentity elements in each group. Equal action means precede equal means over 24 cells. The unit storage budget now refers to generator matrices. These numbers are distinct from the all-nonidentity evaluation, which includes fitted generators. No new model or predictor was fitted for this aggregation.',
         'tab:reproduction-unseen','lrr')
    # Paired fixed-cell differences, preserving heterogeneity rather than treating rows as independent samples.
    paired=[]
    all_rows=read('compact_rows.json')
    for domain in ['C7','D4_corner']:
        for reg in ['semantic','consistent_swap01']:
            for track in ['all_element','generator_only']:
                pool=[r for r in all_rows if (r['domain'],r['registration'],r['track'],r['split'])==(domain,reg,track,'test')]
                if not pool: continue
                ridge={r['cell']:r['nmse'] for r in pool if r['method']=='ridge'}
                for method in sorted({r['method'] for r in pool}):
                    chosen={r['cell']:r['nmse'] for r in pool if r['method']==method}
                    if chosen.keys()!=ridge.keys():raise ValueError('Unpaired cells')
                    delta=[chosen[k]-ridge[k] for k in sorted(ridge)]
                    paired.append({'domain':domain,'registration':reg,'track':track,'method':method,'cells':len(delta),'mean_method_minus_ridge':statistics.fmean(delta),'cells_lower_than_ridge':sum(x<0 for x in delta),'scope':'descriptive; correlated panel cells, no new significance test'})
    summary['paired_compact']=paired
    out=d/'derived_summary.json';out.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    owned=[*generated,out,*[root/n for n in review['inputs']]]
    p=root/'release/artifact_manifest.json';manifest=json.loads(p.read_text());names={str(x.relative_to(root)) for x in owned}
    manifest['artifacts']=[x for x in manifest['artifacts'] if x['path'] not in names]
    for x in owned:
        name=str(x.relative_to(root));meta=review['inputs'].get(name,{})
        manifest['artifacts'].append({'path':name,'sha256':sha(x),'source_run':review['source_run'],
             'source_path':'; '.join(meta.get('archive_source_paths',['parent-from-zero-v1/compact_context/checkpoints/*.json; from-zero-v1-rank-recovery/prospective/'])),
             'source_archive_sha256':review['source_archive_sha256'],'role':'reviewed-descriptive-replay','renderer_sha256':sha(Path(__file__)),
             'transformation':'scripts/build_reproduction_assets.py; release/reproduction_review.json; equal fixed-panel means, no inferential promotion'})
    manifest['artifacts'].sort(key=lambda x:x['path']);p.write_text(json.dumps(manifest,indent=2)+'\n')
    return {'generated_tables':len(generated),'input_files':len(review['inputs']),'independent_confirmation':False,'model_execution':False}

if __name__=='__main__': print(json.dumps(build(),indent=2))
