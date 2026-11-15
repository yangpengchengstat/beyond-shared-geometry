"""Render a LaTeX table from completed saved summaries; no matrix analysis."""
import argparse
import json
from pathlib import Path

def number(value, places=6):
    return f"{value:.{places}f}"

def scientific(value):
    if value < 0.001:
        mantissa, exponent = f"{value:.2e}".split('e')
        return rf"{mantissa}\!\times\!10^{{{int(exponent)}}}"
    return f"{value:.5f}"

def render(coverage):
    assert coverage['status'] == 'completed'
    assert coverage['anchor_validation_passed']
    assert coverage['unit_status_counts'] == {'complete': 6000}
    values = {(x['domain'],x['dimension']):x for x in coverage['domain_summaries']}
    lines = [r'\begin{table}[htbp]',r'\centering\small',r'\setlength{\tabcolsep}{3pt}',
             r'\begin{tabular}{@{}lrrrrrr@{}}',r'\toprule',
             r'Domain & $r$ & Mean $\gamma_r$ & Mean $p^-$ & Median $p^-$ & Positive & Unique top \\',r'\midrule']
    for domain in ['C7','D4']:
        for q in [16,8]:
            row=values[domain,q]
            assert row['aggregate_status']=='complete_descriptive' and row['fully_evaluable_cells']==24
            group=domain[0]+'_'+domain[1]
            lines.append(f"${group}$ & {q} & {number(row['means']['gamma'])} & {number(row['means']['p_minus'])} & {number(row['medians']['p_minus'])}"
                         +f" & {row['positive_premium_count']}/24 & {row['classification_counts']['unique_leading']}/24 " +r'\\')
        if domain=='C7':lines.append(r'\addlinespace')
    lines.extend([r'\bottomrule',r'\end{tabular}',
      r'\caption{Spectral sensitivity on the same Study~2 banks, with 24 equally weighted conditions per domain and $q=r$. The 16-dimensional rows reproduce the primary results under their original tolerances; the eight-dimensional rows use the leading eight saved training-PCA coordinates. Here $p^-=\max\{0,2(1-\gamma_r)-\eta\}$ bounds the joint cost of exact laws and orthogonality. Positive counts $p^->\tau(p^-,0)$; unique top counts the semantic image with uniquely largest budget among all 120 C7 or three D4 images. Each dimension has its own whitening metric; lower values across dimensions do not establish a superior dimension.}',
      r'\label{tab:spectral-dimension-sensitivity}',r'\end{table}'])
    return '\n'.join(lines)+'\n'

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coverage',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with args.output.open('x') as stream:
        stream.write(render(json.loads(args.coverage.read_text())))
