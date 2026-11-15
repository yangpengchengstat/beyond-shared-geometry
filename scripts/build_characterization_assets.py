#!/usr/bin/env python3
"""Rebuild manually reviewed descriptive aggregates; never read private imports or run models."""
from pathlib import Path
import csv,hashlib,json,os
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def _curated_path(root,name):
    rel=Path(name);p=root/rel
    if rel.is_absolute() or '..' in rel.parts or '\\' in name or not p.resolve().is_relative_to(root.resolve()):
        raise ValueError('Unsafe curated input path')
    if any((root/Path(*rel.parts[:i])).is_symlink() for i in range(1,len(rel.parts)+1)):
        raise ValueError('Unsafe curated input symlink')
    if not p.is_file():raise ValueError('Missing curated input: '+name)
    return p

def load_reviewed(root):
    root=Path(root).resolve()
    review=json.loads((root/'release/characterization_review.json').read_text())
    if review['automatic_promotion'] or review['formal_inference'] or review['original_flags_changed']:
        raise ValueError('Descriptive renderer cannot upgrade original eligibility/inference.')
    expected_names={'results/characterization_v1/'+name for name in
                    ['README.md','prediction_summary.csv','storage_accounting.csv','baseline_by_model.csv','sensitivity_contrasts.csv','reviewed_summary.json']}
    if set(review['reviewed_aggregate_files'])!=expected_names:
        raise ValueError('Curated aggregate manifest does not match the declared input set')
    for name,expected in review['reviewed_aggregate_files'].items():
        if sha(_curated_path(root,name))!=expected:raise ValueError('Reviewed aggregate identity changed: '+name)
    if review['curated_summary']!='results/characterization_v1/reviewed_summary.json':raise ValueError('Unexpected curated summary path')
    summary_path=_curated_path(root,review['curated_summary'])
    if sha(summary_path)!=review['curated_summary_sha256']:raise ValueError('Summary identity mismatch')
    summary=json.loads(summary_path.read_text())
    if summary['original_flags']['paper_eligible'] or summary['integration']['formal_confirmation']:raise ValueError('Unexpected promotion')
    if summary['source_archive_sha256']!=review['source_archive_sha256']:raise ValueError('Archive identity mismatch')
    return review,summary

def build(root=ROOT):
    root=Path(root).resolve();review,summary=load_reviewed(root);d=root/'results/characterization_v1';g=root/'paper/generated';figdir=root/'paper/figures'
    g.mkdir(parents=True,exist_ok=True);figdir.mkdir(parents=True,exist_ok=True)
    with (d/'prediction_summary.csv').open() as f:rows=list(csv.DictReader(f))
    lines=[r'\begin{table}[htbp]\centering\small',r'\begin{tabular}{lrrr}\toprule',r'Group & Ridge & Full field & Near-budget field \\',r'\midrule']
    for row in rows:
        label=r'$C_7$' if row['domain']=='C7' else r'$D_4$'
        lines.append(label+' & '+ ' & '.join(f'{float(row[k]):.5f}' for k in ['generator_ridge_unseen','full_generator_kernel_unseen','near_budget_generator_kernel_unseen'])+r' \\')
    lines += [r'\bottomrule\end{tabular}',r'\caption{New-context unseen-composite loss in the separate characterization (fixed 24 conditions/group, $q=16$; lower is better). Primitive-only fitting is used throughout. Near-budget fields store 256 versus 256 coordinates for $C_7$, and 480 versus 512 for $D_4$. Full fields use 84 and 36 times the generator-matrix storage; common PCA/runtime overhead is separate. These normalized losses are not historical $E_{\rm eq}$ or answer accuracy.}',r'\label{tab:characterization-main}\end{table}']
    (g/'characterization_main.tex').write_text('\n'.join(lines)+'\n')
    lines=[r'\begin{table}[htbp]\centering\small',r'\begin{tabular}{llrrr}\toprule',r'Domain / split & Endpoint & Baseline minus linear & Lower & Upper \\',r'\midrule']
    for key,values in summary['legacy_contrasts'].items():
        domain,split=key.split('|');name={'C7':r'$C_7$','D4_corner':r'$D_4$','S3_role':r'$S_3$'}[domain];split=split.replace('test_','').replace('test','new context')
        for endpoint,v in values.items():lines.append(f'{name}, {split} & {endpoint} & {v["mean"]:.6f} & {v["percentile_ci95"][0]:.6f} & {v["percentile_ci95"][1]:.6f} '+r'\\')
    lines += [r'\bottomrule\end{tabular}',r'\caption{All ten original characterization contrasts with frozen 95\% percentile sensitivity intervals (10,000 draws, seed 314159). Positive favors linear. Prediction uses oracle-source category/retention controls, plus the $S_3$ input-metadata potential, selected on validation; context uses scalar retention. Prediction and context have different training-energy denominators. Fixed-panel sensitivities do not certify significance, equivalence, coverage or an old power gate.}',r'\label{tab:characterization-intervals}\end{table}']
    (g/'characterization_intervals.tex').write_text('\n'.join(lines)+'\n')
    os.environ.setdefault('MPLCONFIGDIR',str(root/'build/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.size':8})
    with (d/'baseline_by_model.csv').open() as f:mr=list(csv.DictReader(f))
    order=['meta-llama/Llama-3.2-1B','meta-llama/Llama-3.2-3B','meta-llama/Llama-3.1-8B','Qwen/Qwen2.5-1.5B','Qwen/Qwen2.5-3B','Qwen/Qwen2.5-7B']
    labels=['Llama\n1B','Llama\n3B','Llama\n8B','Qwen\n1.5B','Qwen\n3B','Qwen\n7B']
    fig,axes=plt.subplots(1,2,figsize=(9,3.3),constrained_layout=True)
    for ax,domain in zip(axes,['C7','D4']):
        cell={x['model']:x for x in mr if x['domain']==domain};x=list(range(6))
        ax.bar([i-.18 for i in x],[float(cell[k]['mean_direct_ridge_error']) for k in order],width=.34,color='#365d8d',label='Direct ridge')
        ax.bar([i+.18 for i in x],[float(cell[k]['mean_decoded_centroid_error']) for k in order],width=.34,color='#cf7a32',label='Decoded centroids')
        ax.set_xticks(x,labels);ax.set_title(domain);ax.set_ylabel('Normalized prediction loss');ax.spines[['top','right']].set_visible(False)
    axes[0].legend(frameon=False,fontsize=7)
    fig.savefig(figdir/'characterization_baselines.pdf',metadata={'Creator':'Reviewed descriptive aggregate renderer','CreationDate':None,'ModDate':None});plt.close(fig)
    owned=[g/'characterization_main.tex',g/'characterization_intervals.tex',figdir/'characterization_baselines.pdf',*sorted(d.iterdir())]
    p=root/'release/artifact_manifest.json';m=json.loads(p.read_text());keys={str(x.relative_to(root)) for x in owned}
    m['artifacts']=[a for a in m['artifacts'] if a['path'] not in keys]
    for x in owned:
        m['artifacts'].append({'path':str(x.relative_to(root)),'sha256':sha(x),'source_run':summary['study_id'],'source_path':'summary.json; analysis/*/legacy.json; analysis/*/fields.json; analysis/descriptive_uncertainty.json','source_archive_sha256':summary['source_archive_sha256'],'role':'reviewed-descriptive-aggregate' if x.is_relative_to(d) else 'reviewed-descriptive-paper-asset','transformation':'Manually curated reviewed aggregates; rendering only. See release/characterization_review.json. Original eligibility/inference unchanged.'})
    m['artifacts'].sort(key=lambda x:x['path']);p.write_text(json.dumps(m,indent=2)+'\n')
    return {'reviewed_aggregate_assets':len(owned),'formal_confirmation':False,'model_execution':False}
if __name__=='__main__':print(json.dumps(build(),indent=2))
