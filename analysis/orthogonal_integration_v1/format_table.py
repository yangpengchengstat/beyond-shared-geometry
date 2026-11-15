#!/usr/bin/env python3
"""Format completed risk summaries only; no fitting or numeric analysis."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with (args.inputs / 'domain_risk_reference.csv').open(newline='') as f:
        old = list(csv.DictReader(f))
    coverage = json.loads((args.inputs / 'coverage.json').read_text())
    assert coverage['status'] == 'complete_descriptive'
    assert coverage['status_counts'] == {'complete': 96}
    assert coverage['inputs_unchanged_after_analysis'] is True
    assert [row['domain'] for row in old] == ['C7', 'D4']
    assert all(row['n_cells'] == '24' for row in old)
    by_key = {(row['dimension'], row['domain']): row
              for row in coverage['domain_summaries']}
    assert set(by_key) == {(r, d) for r in (16, 8) for d in ('C7', 'D4')}
    out = [r'% Presentation only: select and round previously saved means.',
           r'\begin{table*}[t]', r'\centering\small',
           r'\setlength{\tabcolsep}{4pt}',
           r'\textbf{(a) Original joint-constraint comparison, $r=16$}\par\smallskip',
           r'\begin{tabular}{@{}lrrrr@{}}', r'\toprule',
           r'Domain & Mean $\eta$ & Mean $L^\star$ bounds & Mean $L^\star-\eta$ bounds & Mean $L(I)$ \\',
           r'\midrule']
    for row in old:
        f = lambda k: f'{float(row[k]):.5f}'
        domain = '$C_7$' if row['domain'] == 'C7' else '$D_4$'
        out.append(f"{domain} & {f('mean_eta')} & [{f('mean_a')}, {f('mean_u')}] & "
                   f"[{f('mean_premium_lower')}, {f('mean_premium_upper')}] & {f('mean_identity_risk')}" + r' \\')
    out += [r'\bottomrule', r'\end{tabular}', r'\par\medskip',
            r'\textbf{(b) Independent orthogonality and further group-law cost}\par\smallskip',
            r'\begin{tabular}{@{}rlrrrr@{}}', r'\toprule',
            r'$r$ & Domain & Mean $L_{\rm orth}$ & Mean $\beta$ & Mean $L^\star-L_{\rm orth}$ bounds & $\ell^->\tau$ \\',
            r'\midrule']
    for dim in (16, 8):
        for domain in ('C7', 'D4'):
            row = by_key[dim, domain]
            assert row['evaluable'] == row['planned'] == 24
            assert row['status'] == 'complete_descriptive'
            m = row['means']
            name = '$C_7$' if domain == 'C7' else '$D_4$'
            out.append(f"{dim} & {name} & {m['L_orth']:.6f} & {m['beta']:.6f} & "
                       f"[{m['extra_law_lower']:.6f}, {m['extra_law_upper']:.6f}] & "
                       f"{row['positive_extra_law_lower_count']}/24" + r' \\')
    out += [r'\bottomrule', r'\end{tabular}',
            r"\caption{\textbf{Same-bank risks and constraint costs.} Each row averages 24 fixed model/readout conditions on the original Study~2 test banks. Panel (a) retains the primary joint-cost comparison. In (b), $L_{\rm orth}$ is the optimum over independently chosen $O(r)$ maps, $\beta=L_{\rm orth}-\eta$, and $\ell^-=\max\{0,2(1-\gamma_r)-L_{\rm orth}\}$ is a lower bound on further common group-law cost. Bounds are computed per cell before averaging; the last column uses the fixed numerical tolerance $\tau=10^{-8}+10^{-7}|\ell^-|$. The intervals are bounds on average class optima, not confidence intervals or measured optima. Each dimension has its own second-moment-whitened metric. For $r=16$ C7, one condition contributes $99.44\%$ of the sum of $\ell^-$, whose median is $5.35\times10^{-6}$; this is concentration of lower bounds, not of true costs (\cref{par:orthogonal-results,app:orthogonal-results}). Frozen-predictor transfer uses a different training metric (\cref{tab:followup-witness}).}",
            r'\label{tab:core-risk-summary}', r'\end{table*}']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('\n'.join(out) + '\n')
    record = {'operation': 'format_existing_saved_summaries_only',
              'inputs': {name: hashlib.sha256((args.inputs / name).read_bytes()).hexdigest()
                         for name in ('domain_risk_reference.csv', 'coverage.json')},
              'mean_rule': 'use saved per-cell-clipped endpoint means, without recomputing',
              'output_name': args.output.name,
              'output_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
              'numerical_analysis': False, 'tests_rerun': False}
    args.output.with_suffix('.source.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
