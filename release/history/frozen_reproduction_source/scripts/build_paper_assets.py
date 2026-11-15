#!/usr/bin/env python3
"""Build manuscript tables and publication figures from bundled frozen evidence.

This script fits no operators, runs no models and recomputes no confidence intervals.
It reports frozen estimates and descriptive means of the declared evidence subsets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'results'
GENERATED = ROOT / 'paper/generated'
FIGURES = ROOT / 'paper/figures'
COLORS = {'joint': '#147D92', 'ridge': '#9C4A17', 'identity': '#6F7378', 'random': '#86529B'}
RUNS = {'full': 'main_full_003', 'afs_abc': 'afs_full_abc_001', 'afs_d': 'afs_d_rank_aware_001', 'additive_field': 'additive_field_falsification_001'}

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(kind, family, name):
    return pd.read_csv(R / kind / family / name)

def row(frame, **where):
    selected = frame
    for key, value in where.items():
        selected = selected[selected[key] == value]
    if len(selected) != 1:
        raise ValueError(f'Expected one frozen row for {where}; observed {len(selected)}')
    return selected.iloc[0]

def interval(r, digits=3, scale=1):
    return rf'${scale*r["mean"]:.{digits}f}\;[{scale*r["ci_low"]:.{digits}f},\,{scale*r["ci_high"]:.{digits}f}]$'

def small_error(value):
    if 0 < abs(value) < 1e-4:
        exponent=int(np.floor(np.log10(abs(value))))
        return rf'${value/(10.**exponent):.2f}\!\times\!10^{{{exponent}}}$'
    return f'{value:.6f}'

def table(name, label, columns, rows, caption, wide=False):
    env = 'table*' if wide else 'table'
    text = '\\begin{' + env + '}[t]\n\\centering\n\\small\n'
    text += '\\begin{tabular}{' + columns + '}\n\\toprule\n'
    text += '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n'
    text += '\\caption{' + caption + '}\n\\label{' + label + '}\n\\end{' + env + '}\n'
    (GENERATED / (name + '.tex')).write_text(text)

def write_source(name, frame):
    frame.to_csv(FIGURES / (name + '.source.csv'), index=False, float_format='%.17g')

def savefig(fig, name, preview):
    fig.savefig(FIGURES / (name + '.pdf'), bbox_inches='tight', metadata={'Creator': 'Beyond Shared Geometry evidence build', 'CreationDate': None, 'ModDate': None})
    if preview:
        preview.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview / (name + '.png'), dpi=150, bbox_inches='tight')

def generate_tables(s, g, c, d, gc, ci, dc, b, sc):
    means = s.groupby(['group', 'method'])[['E_eq', 'E_rel_data', 'E_hom_op']].mean()
    labels = [('group_consistent_joint', 'Joint'), ('independent_ridge', 'Ridge'), ('identity', 'Identity'), ('random_orthogonal', 'Random orthogonal')]
    rows = [r'Group & Estimator & $E_{\rm eq}$ & $E_{\rm rel,data}$ \\', r'\midrule']
    for group in ('C7', 'D4'):
        for method, label in labels:
            r = means.loc[(group, method)]
            rel = '--' if pd.isna(r.E_rel_data) else f'{r.E_rel_data:.4f}'
            rows.append(f'{group} & {label} & {r.E_eq:.4f} & {rel} ' + r'\\')
        if group == 'C7': rows.append(r'\midrule')
    table('structural_main', 'tab:structural-main', 'llrr', rows,
          'Held-out structural errors at $d=16$: descriptive means over six fixed models, four normalized depths, five splits, and six test templates per split. Lower is better. Random values average five controls; these controls are not independent model replications. Identity relation defects are intentionally not interpreted.')

    gm = g.groupby(['group', 'method']).E_gen_eq.mean()
    rows = [r'Group & Composed & Direct & Composed minus direct [95\% CI] \\', r'\midrule']
    for group in ('C7', 'D4'):
        r = row(gc, group=group, contrast='generator_vs_direct', model='ALL_MODELS')
        rows.append(f'{group} & {gm.loc[(group,"generator_composed")]:.4f} & {gm.loc[(group,"direct_ridge")]:.4f} & {interval(r,4)} ' + r'\\')
    table('generator_transfer', 'tab:generator-transfer', 'lrrl', rows,
          'Generator-only predictions of five never-independently-fitted elements on held-out templates ($d=16$). Direct Ridge fits each element separately and has lower error. The final column preserves the registered paired bootstrap contrast; all models form a fixed robustness panel.', wide=True)

    rows = [r'Group & Metric & Learned minus identity [95\% CI] & Learned minus additive [95\% CI] \\', r'\midrule']
    for group in ('C7', 'D4'):
        for metric, label, scale in [('target_logit_change', 'Target logit', 1), ('top1_success', 'Top-1 (pp)', 100)]:
            identity = row(ci, model='ALL', group=group, method='transformation', metric=metric, analysis='causal_learned_minus_identity')
            additive = row(ci, model='ALL', group=group, method='transformation', metric=metric, analysis='causal_learned_minus_source_conditioned_additive')
            rows.append(f'{group} & {label} & {interval(identity,3,scale)} & {interval(additive,3,scale)} ' + r'\\')
    order = row(ci, model='ALL', group='D4', method='learned_operator', metric='relation_final_logit_distance', analysis='d4_true_minus_wrong_order')
    rows += [r'\midrule', r'\multicolumn{4}{l}{D4 correct-minus-wrong final-logit distance: ' + interval(order,4) + r'.} \\']
    table('causal_main', 'tab:causal-main', 'llll', rows,
          'Frozen behavioral contrasts from externally composed activation edits. Additive denotes the training-only source- and transformation-conditioned comparator, distinct from the global additive field. Positive logit/accuracy differences favor the learned edit; the D4 distance contrast is lower for the correct word. Top-1 measures the scored continuation token; all archived continuation counts are one. Registered intervals describe the fixed panel and preserve the original nested bootstrap.', wide=True)

    rows = [r'Group & Model & Location & $E_{\rm eq}$ & $E_{\rm gen,eq}$ & $E_{\rm rel,data}$ \\', r'\midrule']
    dm = d.groupby(['group', 'model_key', 'representation_location', 'metric']).value.mean()
    for group in ('C7', 'D4'):
        for model, label in [('llama_1b','Llama 1B'), ('qwen_1_5b','Qwen 1.5B')]:
            for location, loclabel in [('embedding','Embedding'), ('internal','Internal mean')]:
                eq = dm.loc[(group,model,location,'E_eq')]
                gen = dm.loc[(group,model,location,'E_gen_eq')]
                rel = dm.loc[(group,model,location,'E_rel_data')]
                rows.append(f'{group} & {label} & {loclabel} & {small_error(eq)} & {small_error(gen)} & {small_error(rel)} ' + r'\\')
    table('lexical_control', 'tab:lexical-control', 'lllrrr', rows,
          r'Rank-matched lexical control using independent Ridge, with generator-only composed predictions for $E_{\rm gen,eq}$. Both locations use C7 $d=6$ or D4 $d=3$; these are not the FULL $d=16$ estimates. Internal means average four fixed depths. Embedding errors are smaller in every tested model/group; all 150 archived internal-minus-embedding intervals are positive. Repeated templates at the embedding location reuse the same category vectors.', wide=True)

    rows = [r'Control & Group/metric & Frozen summary \\', r'\midrule']
    for group in ('C7','D4'):
        for metric,label in [('E_hom_data','data homomorphism'),('E_hom_op','operator homomorphism')]:
            cell=b[(b.group==group)&(b.data_split=='test')&(b.metric==metric)]
            n=int((cell.true_relation_defect < cell.null_q025).sum())
            rows.append(f'AFS-B & {group}, {label} & {n}/{len(cell)} below null 2.5th percentile '+r'\\')
    for group in ('C7','D4'):
        for variant,label in [('train_only_state_relabel','state relabel'),('template_independent_target_scramble','target scramble')]:
            r=row(sc,model='ALL_MODELS',group=group,variant=variant,metric='E_eq')
            rows.append(f'AFS-C & {group}, {label} & {interval(r,3)} '+r'\\')
    rows += [r'\midrule', r'Global additive & Balanced complete orbit & $\widehat v_g=0$ by permutation cancellation \\']
    table('diagnostic_controls','tab:diagnostic-controls','lll',rows,
          'Registration/correspondence controls and the scope of global additive falsification. AFS-C reports real-minus-scrambled held-out equivariance error. AFS-B is conditional on fixed operators and excludes automorphisms; it is not a test against every stable category code. The global displacement estimator averages complete state orbits and therefore reduces to identity independently of the activations.',wide=True)

def generate_figures(s,g,c,d,gc,ci,dc,b,sc,preview):
    # Import after evidence verification; plotting uses no model or fitting libraries.
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'bsg-paper-matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'axes.labelsize':8,'axes.titlesize':9,'legend.fontsize':7,'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42,'ps.fonttype':42,'savefig.transparent':False})
    methods=[('group_consistent_joint','Joint',COLORS['joint']),('independent_ridge','Ridge',COLORS['ridge']),('identity','Identity',COLORS['identity']),('random_orthogonal','Random',COLORS['random'])]
    ss=s.groupby(['group','depth_slot','method'])[['E_eq','E_rel_data']].mean().reset_index()
    write_source('structural_errors',ss)
    fig,axes=plt.subplots(2,2,figsize=(6.7,3.8),sharex=True,constrained_layout=True)
    for col,group in enumerate(('C7','D4')):
        for rr,(metric,label) in enumerate([('E_eq',r'$E_{eq}$'),('E_rel_data',r'$E_{rel,data}$')]):
            ax=axes[rr,col]
            for method,name,color in methods:
                z=ss[(ss.group==group)&(ss.method==method)].dropna(subset=[metric])
                if len(z): ax.plot(z.depth_slot,z[metric],'-o',ms=3,lw=1.3,color=color,label=name)
            ax.set(yscale='log',ylabel=label,title=f'{group}, d = 16' if rr==0 else None,xticks=[.2,.4,.6,.8]);ax.grid(axis='y',alpha=.18)
            if rr==1:ax.set_xlabel('Normalized block depth')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,1.08),ncol=4,frameon=False)
    savefig(fig,'structural_errors',preview);plt.close(fig)

    gm=g.groupby(['group','method']).E_gen_eq.mean().reset_index();write_source('generator_transfer',gm)
    order=[('direct_ridge','Direct\nridge'),('generator_composed','Generator\ncomposed'),('identity','Identity'),('matched_random_generator','Matched\nrandom'),('matched_wrong_word','Wrong\nword')]
    fig,axes=plt.subplots(1,2,figsize=(6.7,2.4),sharey=True,constrained_layout=True)
    for ax,group in zip(axes,('C7','D4')):
        sel=[(m,l) for m,l in order if m in set(gm[gm.group==group].method)]
        vals=[float(row(gm,group=group,method=m).E_gen_eq) for m,_ in sel]
        ax.scatter(range(len(vals)),vals,color=[COLORS['ridge'],COLORS['joint'],COLORS['identity'],COLORS['random'],'#A48922'][:len(vals)],s=30,zorder=3)
        ax.set(yscale='log',ylim=(.025,3.),xlim=(-.45,len(vals)-.55),xticks=range(len(vals)),xticklabels=[l for _,l in sel],title=f'{group}, d = 16',ylabel=r'Unseen-element $E_{gen,eq}$');ax.grid(axis='y',alpha=.15)
    savefig(fig,'generator_transfer',preview);plt.close(fig)

    dm=d[d.metric.isin(['E_gen_eq','E_rel_data'])].groupby(['group','model_key','representation_location','depth_slot','metric']).value.mean().reset_index()
    write_source('lexical_internal',dm)
    fig,axes=plt.subplots(2,2,figsize=(6.7,3.7),sharex=True,constrained_layout=True)
    for col,group in enumerate(('C7','D4')):
        for rr,(metric,label) in enumerate([('E_gen_eq',r'$E_{gen,eq}$'),('E_rel_data',r'$E_{rel,data}$')]):
            ax=axes[rr,col]
            for model,mlabel,color in [('llama_1b','Llama 1B',COLORS['joint']),('qwen_1_5b','Qwen 1.5B',COLORS['ridge'])]:
                z=dm[(dm.group==group)&(dm.model_key==model)&(dm.metric==metric)].sort_values('depth_slot')
                ax.plot(z.depth_slot,z.value,'-o',ms=3,lw=1.2,color=color,label=mlabel)
                emb=z[z.representation_location=='embedding'];ax.scatter(emb.depth_slot,emb.value,marker='D',s=23,color=color,zorder=3)
            ax.set(yscale='log',xticks=[0,.2,.4,.6,.8],xticklabels=['Emb.','.2','.4','.6','.8'],ylabel=label,title=f'{group}, matched d = {6 if group=="C7" else 3}' if rr==0 else None);ax.grid(axis='y',alpha=.15)
            if rr==1:ax.set_xlabel('Embedding / normalized block depth')
    axes[0,0].legend(loc='center right',frameon=False)
    savefig(fig,'lexical_internal',preview);plt.close(fig)

    selected=ci[(ci.model=='ALL')&(ci.method=='transformation')&(ci.metric=='target_logit_change')&ci.analysis.isin(['causal_learned_minus_identity','causal_learned_minus_random_orthogonal','causal_learned_minus_source_conditioned_additive'])].copy()
    write_source('causal_additive',selected[['group','analysis','mean','ci_low','ci_high']])
    fig,axes=plt.subplots(1,2,figsize=(6.7,2.0),constrained_layout=True)
    for ax,group in zip(axes,('C7','D4')):
        for y,(analysis,label) in enumerate([('causal_learned_minus_identity','Identity'),('causal_learned_minus_random_orthogonal','Random'),('causal_learned_minus_source_conditioned_additive','Source-cond. additive')]):
            z=row(selected,group=group,analysis=analysis)
            ax.errorbar(z['mean'],y,xerr=[[z['mean']-z.ci_low],[z.ci_high-z['mean']]],fmt='o',ms=4,color=COLORS['joint'],capsize=3,lw=1.3)
        ax.axvline(0,color='#666666',lw=.8,ls='--');ax.set(yticks=[0,1,2],yticklabels=['Identity','Random','Source-cond. additive'],title=group,xlabel='Learned minus baseline target logit');ax.invert_yaxis();ax.grid(axis='x',alpha=.15)
    savefig(fig,'causal_additive',preview);plt.close(fig)

def update_manifest():
    path=ROOT/'release/artifact_manifest.json';path.parent.mkdir(parents=True,exist_ok=True)
    existing=json.loads(path.read_text()).get('artifacts',[]) if path.exists() else []
    registry=json.loads((R/'registry.json').read_text())
    by_path={e['path']:e for e in registry['entries']}
    historical_tables={'structural_main','generator_transfer','causal_main','lexical_control','diagnostic_controls'}
    historical_figures={'structural_errors','generator_transfer','lexical_internal','causal_additive'}
    paths=[ROOT/name for name in by_path]
    paths += [R/'registry.json', R/'README.md']
    paths += [GENERATED/(name+'.tex') for name in historical_tables]
    paths += [p for p in FIGURES.glob('*') if p.is_file() and p.stem.split('.')[0] in historical_figures]
    paths=sorted(set(p for p in paths if p.is_file()))
    owned={str(p.relative_to(ROOT)) for p in paths}
    artifacts=[a for a in existing if a['path'] not in owned]
    for file in paths:
        relative=str(file.relative_to(ROOT));e=by_path.get(relative)
        if e:
            artifacts.append({'path':relative,'sha256':digest(file),'source_run':RUNS[e['family']],'source_path':e['source_relative_path'],'role':'frozen-original' if e['byte_original'] else 'public-derivative','original_sha256':e['source_sha256'],'transformation':e['transformation']})
        else:
            generated=relative.startswith('paper/')
            artifacts.append({'path':relative,'sha256':digest(file),'source_run':'curated-frozen-evidence','source_path':'results/registry.json','role':'generated-paper-asset' if generated else ('evidence-registry' if relative=='results/registry.json' else 'public-evidence-documentation')})
    path.write_text(json.dumps({'schema_version':1,'artifacts':sorted(artifacts,key=lambda a:a['path'])},indent=2)+'\n')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview-dir',type=Path,help='Optional directory for PNG review images.')
    args=parser.parse_args()
    # Verify included bytes before deriving publication assets.
    registry=json.loads((R/'registry.json').read_text())
    for entry in registry['entries']:
        file=ROOT/entry['path']
        if not file.exists() or digest(file)!=entry['sha256']:
            raise ValueError(f'Frozen evidence hash mismatch: {entry["path"]}')
    GENERATED.mkdir(parents=True,exist_ok=True);FIGURES.mkdir(parents=True,exist_ok=True)
    frames=(read('derived','full','primary_structural_template_metrics.csv'),read('derived','afs_abc','primary_generator_composition_metrics.csv'),read('derived','full','causal_transformation_summary.csv'),read('frozen','afs_d','location_metrics.csv'),read('frozen','afs_abc','bootstrap_generator_contrasts.csv'),read('frozen','full','confidence_intervals.csv'),read('frozen','afs_d','bootstrap_depth_contrasts.csv'),read('frozen','afs_abc','permutation_summary.csv'),read('frozen','afs_abc','bootstrap_scramble_contrasts.csv'))
    generate_tables(*frames);generate_figures(*frames,args.preview_dir);update_manifest()
    print('Built 5 TeX tables, 4 PDF figures and their source CSVs from verified bundled frozen evidence.')

if __name__=='__main__':main()
