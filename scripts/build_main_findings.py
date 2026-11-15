#!/usr/bin/env python3
"""Present three core findings using existing recorded values; no new statistics."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
SOURCE='paper/generated/identification_summary.source.json'
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def build(root=ROOT):
    source=root/SOURCE
    data=json.loads(source.read_text())
    manifest=json.loads((root/'release/artifact_manifest.json').read_text())
    matches=[r for r in manifest['artifacts'] if r['path']==SOURCE]
    if len(matches)!=1 or matches[0]['sha256']!=digest(source): raise ValueError('Original evidence table provenance mismatch')
    used={}
    def v(group,key,precision=6):
        name=f'{group}.{key}'; item=data['values'][name]; used[name]=item
        return f"{item['value']:.{precision}f}" if precision else str(int(item['value']))
    def pair(key,precision=6): return v('C7',key,precision)+' / '+v('D4',key,precision)
    tex=r'''% Existing reviewed values only; see main_findings.source.json.
\begin{table*}[t]
\centering\small
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.10}
\begin{tabular}{@{}p{.23\textwidth}p{.39\textwidth}p{.34\textwidth}@{}}
\toprule
Finding & Existing evidence ($C_7$ / $D_4$) & Interpretation \\
\midrule
Recovery admits competitive alternatives & Centroid: CENTROID.\newline Ridge: RIDGE.\newline Centroid lower in WINS cells. & Category information improves the $C_7$ mean, not most cells. The control receives labels and the action table; a coherent registration is also competitive. \\
\addlinespace[3pt]
Rankings depend on storage & Nystr\"om first improves both group means at $2\times$; RFF at $4\times$ nominal storage (\cref{fig:budget-frontier}). & All budgets are retained. Parameter storage does not match statistical capacity or runtime memory. \\
\addlinespace[3pt]
An original-bank transport advantage & Coordinate fit: COORD.\newline Difference fit: DIRECT.\newline Coherent fit: COHERENT. & The new authored bank reverses the advantage in D4 (Table~\ref{tab:followup-context}). Close cross-registration means do not establish semantic preference. \\
\bottomrule
\end{tabular}
\caption{Three original-bank findings motivating the completed follow-up. The new-context result limits the third finding to its original construction. Smaller losses are better. Category and budget results use the replay coordinate-energy denominator; context differences use a separate difference-energy denominator. Each group contains 24 fixed model/readout cells. Targets and changed-state masks are registration-specific, so cross-registration means compare distinct tasks. Full recovery, generator and restricted-registration contrasts remain in \cref{tab:identification-summary}; \cref{sec:setup} defines the descriptive statistical scope.}
\label{tab:main-findings}
\end{table*}
'''
    tex=tex.replace('CENTROID',pair('compact.decoded_centroid')).replace('RIDGE',pair('compact.ridge'))
    tex=tex.replace('WINS',v('C7','centroid_wins',0)+'/24 / '+v('D4','centroid_wins',0)+'/24')
    tex=tex.replace('COORD',pair('context.coordinate')).replace('DIRECT',pair('context.direct_context')).replace('COHERENT',pair('context.coherent'))
    # Budget crossover claims are already recorded, not recalculated here.
    for group in ('C7','D4'):
        for method,expected in (('nystrom',2),('rff',4)):
            name=f'{group}.compact.{method}.first_tested_crossover'; item=data['values'][name]
            if float(item['value'])!=expected: raise ValueError('Recorded crossover changed')
            used[name]=item
    output=root/'paper/generated/main_findings.tex';output.write_text(tex)
    provenance={'operation':'Select and format already recorded values; no new aggregation or statistical inference',
      'source_run':'reviewed fixed-panel historical/replay evidence',
      'inputs':{SOURCE:digest(source)},'renderer':'scripts/build_main_findings.py',
      'renderer_sha256':digest(Path(__file__)),'values':used,'output_sha256':digest(output),
      'model_execution':False,'fitting':False,'resampling':False,'new_statistics':False,'independent_confirmation':False}
    (root/'paper/generated/main_findings.source.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('MAIN_FINDINGS_RENDERED_FROM_EXISTING_VALUES',len(used))
if __name__=='__main__':build()
