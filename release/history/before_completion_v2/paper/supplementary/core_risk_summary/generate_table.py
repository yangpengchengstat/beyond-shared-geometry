#!/usr/bin/env python3
"""Format existing reviewed domain summaries. No matrix or statistical analysis."""
import csv
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'results/main30_bounded_v1/domain_risk_reference.csv'
OUTPUT=ROOT/'paper/generated/core_risk_summary.tex'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=SOURCE)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args()
    source,output=args.source,args.output
    with source.open(newline='') as f:
        rows=list(csv.DictReader(f))
    assert [r['domain'] for r in rows]==['C7','D4']
    assert all(r['n_cells']=='24' for r in rows)
    out=[r'% Presentation only: saved full-precision domain aggregates; no new averaging.',
         r'\begin{table*}[t]',r'\centering\small',r'\setlength{\tabcolsep}{5pt}',
         r'\begin{tabular}{@{}lrrrr@{}}',r'\toprule',
         r'Domain & Mean linear risk & Mean exact-action bounds & Mean additional-risk bounds & Mean identity risk \\',
         r'\midrule']
    for row in rows:
        def fmt(k):return f'{float(row[k]):.5f}'
        domain='$C_7$' if row['domain']=='C7' else '$D_4$'
        out.append(f"{domain} & {fmt('mean_eta')} & [{fmt('mean_a')}, {fmt('mean_u')}] & [{fmt('mean_premium_lower')}, {fmt('mean_premium_upper')}] & {fmt('mean_identity_risk')} \\\\")
    out += [r'\bottomrule',r'\end{tabular}',
            r"\caption{\textbf{The scale of exact orthogonal-action cost.} All entries use the original Study~2 test bank's own second-moment-whitened metric, at dimension $r=16$, with equal averages over 24 fixed model/readout conditions per domain. Linear risk is the independent least-squares optimum $\eta$ on that bank. The next two columns average cell-wise bounds on the best exact orthogonal-action risk $L^\star$ and its additional cost $L^\star-\eta$. These deterministic intervals are neither confidence intervals nor measured optima; the constraints combine exact group laws and orthogonality. Identity risk uses the same bank and metric. The C7 median lower endpoint is $0.000389$, and two conditions contribute $90.7\%$ of its sum; \cref{tab:bounded-risk} and the accompanying text retain the full heterogeneity. Constructed-action transfer uses a different, frozen training metric (\cref{tab:followup-witness}).}",
            r'\label{tab:core-risk-summary}',r'\end{table*}']
    output.write_text('\n'.join(out)+'\n')
    def local_name(path):
        try:return str(path.resolve().relative_to(ROOT.resolve()))
        except ValueError:return path.name
    record={'kind':'format_saved_aggregates_only','source':local_name(source),
            'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'source_columns':['mean_eta','mean_a','mean_u','mean_premium_lower','mean_premium_upper','mean_identity_risk'],
            'output':local_name(output),'output_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
            'rounding':'five decimal places, matching existing tables',
            'no_new_matrix_analysis':True,'no_new_statistics':True,
            'heterogeneity_source':'paper/sections/planned_studies.tex; retained reviewed paragraph'}
    (output.with_suffix('.source.json')).write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))

if __name__=='__main__':main()
