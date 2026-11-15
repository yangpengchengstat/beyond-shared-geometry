#!/usr/bin/env python3
"""Render audited recorded results only. No model, fit, bootstrap, or spectral solve."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'results/tmlr_followup_v2'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(root=ROOT):
    data = root / 'results/tmlr_followup_v2'
    manifest = json.loads((data / 'evidence_manifest.json').read_text())
    for name, value in manifest['files'].items():
        if digest(data / name) != value:
            raise ValueError('Reviewed evidence changed: ' + name)
    context = json.loads((data / 'context_metrics.json').read_text())
    spec = json.loads((data / 'spectral/metrics.json').read_text())
    gen = root / 'paper/generated'
    gen.mkdir(exist_ok=True)
    def f(value):
        return f'{value:.5f}'
    def mean(domain, metric):
        return spec['spectral'][domain][metric]['mean']
    rows = []
    for g in ('C7', 'D4'):
        s, a = spec['spectral'][g], spec['alternatives'][g]
        rows.append(f"${g[0]}_{g[1]}$ & {f(mean(g,'gamma'))} & [{f(mean(g,'L_lower'))}, {f(mean(g,'L_upper'))}] & {f(mean(g,'eta_LS'))} & {f(mean(g,'trivial_b'))} & {a['semantic_unique_top']}/24 \\")
    tex = r'''\begin{table}[t]
\centering\small
\begin{tabular}{@{}lrrrrr@{}}
\toprule
Domain & Mean $\gamma_r$ & Mean bound endpoints & Mean $\eta$ & Mean $b_{\rm triv}$ & Image rank 1 \\
\midrule
''' + '\n'.join(row + '\\' for row in rows) + r'''
\bottomrule
\end{tabular}
\caption{Exact-action compatibility and registration on the retained test bank, at $r=16$. Each domain has 24 model/readout cells. The bounds concern cell-specific $L^\star$ in each test bank's own whitened metric; displayed endpoints average those cell bounds, not confidence limits. $b_{\rm triv}$ is selected trivial-type energy divided by $r$. The last column counts unique highest-$\gamma_r$ ranks of the specified image among 120 C7 or three D4 images; it is not a rank of exact $L^\star$ or a significance test.}
\label{tab:followup-spectral}
\end{table}
'''
    (gen / 'followup_spectral.tex').write_text(tex)
    rows = []
    for g in ('C7', 'D4'):
        w = spec['witness'][g]
        vals = [w[k]['mean'] for k in ('train_witness_risk', 'test_witness_risk', 'test_ols_risk', 'test_ridge_risk')]
        rows.append(f"${g[0]}_{g[1]}$ & " + ' & '.join(f(v) for v in vals) + r' & 24/24 \\')
    (gen / 'followup_witness.tex').write_text(r'''\begin{table}[t]
\centering\small
\begin{tabular}{@{}lrrrrr@{}}
\toprule
& Construction bank & \multicolumn{3}{c}{Old test bank, frozen training metric} & Witness worse \\
Domain & Witness & Witness & OLS & Ridge & than OLS \\
\midrule
''' + '\n'.join(rows) + r'''
\bottomrule
\end{tabular}
\caption{Attained risk of the constructed exact action and independent predictive controls. These means use training whitening for both banks and include identity and fixed points. They must not be compared with Table~\ref{tab:followup-spectral}'s test-whitened bounds. OLS and ridge need not obey group laws. The witness is feasible, not claimed optimal.}
\label{tab:followup-witness}
\end{table}
''')
    rows = []
    for key, label in [('C7', '$C_7$'), ('D4', '$D_4$'), ('primary_equal_domain_panel', 'Equal-domain mean')]:
        a = context['aggregate'][key]
        c = a['recorded_contrast']
        lo, hi = c['family_bootstrap_percentiles_2_5_97_5']
        rows.append(label + ' & ' + ' & '.join(f(a['means'][k]) for k in ('identity','coordinate','difference')) + f" & {c['mean_difference_minus_coordinate']:+.5f} & [{lo:+.5f}, {hi:+.5f}]" + r' \\')
    (gen / 'followup_context.tex').write_text(r'''\begin{table}[t]
\centering\small\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lrrrrr@{}}
\toprule
& \multicolumn{3}{c}{Mean context-difference loss} & Difference minus & Family sensitivity \\
Panel & Identity & Coordinate fit & Difference fit & coordinate & range \\
\midrule
''' + '\n'.join(rows) + r'''
\bottomrule
\end{tabular}
\caption{New authored contexts under matched validation selection. Losses use old-training difference energy. Negative contrasts favor difference fitting. The last column copies the recorded 2.5th/97.5th percentiles from 2,000 paired-family bootstrap draws; it describes conditional sensitivity over 64 authored families, not calibrated population coverage. Model/readout cells and both domains remain paired in each family.}
\label{tab:followup-context}
\end{table}
''')
    rows=[]
    for g in ('C7','D4'):
        for mech in ('M1','M2','M3'):
            for noise in ('0','0.05'):
                a=spec['simulations'][f'{g}_{mech}_noise{noise}']
                rows.append(f"${g[0]}_{g[1]}$ & {mech} & {noise} & {a['test_bank_gamma']['mean']:.5f} & {a['test_witness_risk']['mean']:.5g} & {a['test_ols_risk']['mean']:.5g}" + r' \\')
    (gen/'followup_simulations.tex').write_text(r'''\begin{table}[ht]
\centering\small
\begin{tabular}{@{}lllrrr@{}}
\toprule
Domain & Mechanism & Noise & Test-bank $\gamma_r$ & Test witness & Test OLS \\
\midrule
'''+'\n'.join(rows)+r'''
\bottomrule
\end{tabular}
\caption{All 36 fixed mechanism settings, averaged over three declared seeds per row. M1 is a shared action, M2 uses contextual orientations, and M3 is a stable state-code control admitting a shared action. Gamma uses test whitening; prediction risks use training whitening. Machine-scale values are shown without modifying archived numbers. These are mechanism illustrations, not a calibration or power experiment.}
\label{tab:followup-simulations}
\end{table}
''')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.7, 2.7))
    models=context['descriptive_model_contrasts']
    labels=[m['model_id'].split('/')[-1].replace('Llama-', 'Llama ').replace('Qwen2.5-', 'Qwen ') for m in models]
    for offset, domain, color, marker in [(-.12,'C7','#2166ac','o'),(.12,'D4','#b35806','s')]:
        ax.scatter([m[domain] for m in models],[i+offset for i in range(len(models))],label=domain, color=color, marker=marker,s=34,zorder=3)
    ax.axvline(0,color='#555555',linewidth=.8)
    ax.set_yticks(range(len(labels)),labels)
    ax.invert_yaxis()
    ax.set_xlabel('Difference-fit loss minus coordinate-fit loss')
    ax.grid(axis='x',alpha=.2)
    ax.spines[['top','right']].set_visible(False)
    ax.legend(frameon=False,loc='lower right')
    fig.tight_layout()
    figure=root/'paper/figures/context_transfer_followup.pdf'
    fig.savefig(figure,metadata={'CreationDate':None,'ModDate':None,'Author':'Anonymous'})
    plt.close(fig)
    outputs=[gen/('followup_'+x+'.tex') for x in ('spectral','witness','context','simulations')]+[figure]
    record={'operation':'Format audited aggregates and recorded sensitivity percentiles; no refit, bootstrap, model call or spectral recomputation',
            'script':'scripts/build_tmlr_results.py','script_sha256':digest(Path(__file__)),
            'evidence_manifest_sha256':digest(data/'evidence_manifest.json'),
            'outputs':{str(p.relative_to(root)):digest(p) for p in outputs},
            'source_archive_sha256':context['source_archive_sha256']}
    (gen/'followup_results.source.json').write_text(json.dumps(record,indent=2)+'\n')
    print('RENDERED_REVIEWED_FOLLOWUP_RESULTS',len(outputs))

if __name__=='__main__':
    build()
