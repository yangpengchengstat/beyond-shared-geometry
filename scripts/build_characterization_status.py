#!/usr/bin/env python3
"""Render status-only manuscript slots; never promote numerical results."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]

def build(root=ROOT):
    status_path=root/'release/current_study_status.json'
    status=json.loads(status_path.read_text())
    if status['paper_eligible'] or status['automatic_promotion']:
        raise ValueError('Status-only renderer cannot accept publication promotion.')
    complete=status['study_status']=='complete_limited_precision_characterization_compact_audit_complete'
    if not complete and status['study_status']!='source_frozen_independent_results_awaiting_completion_and_audit':
        raise ValueError('Unknown study status; refusing a stale or invented table')
    content={
    'prospective_results.tex':r'''% Status-only reserved slots. No independent estimate has been promoted.
\begin{table}[t]
\centering\footnotesize
\caption{Independent characterization slots. ``Awaiting audit'' is not a zero, an estimate or a scientific pass. Development analyses do not fill these entries.}
\label{tab:prospective}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lll}
\toprule Domain & Comparison / endpoint & Status \\
\midrule
$C_7$ & Full fields; prediction/context & Awaiting audit \\
$D_4$ & Full fields; prediction/context & Awaiting audit \\
$C_7/D_4$ & Generator/cycle/order & Awaiting audit \\
$C_7/D_4$ & Coordinate-storage curves & Awaiting audit \\
$S_3$ & Joint/context/entity strata & Awaiting audit \\
\bottomrule
\end{tabular}
\end{table}
''',
    'prospective_status.tex':r'''% Process status, not new model measurements or confirmatory effects.
\begin{table}[t]\centering\footnotesize
\caption{Status of earlier stages and the bounded current study. Completion describes execution, not an evidential upgrade.}\label{tab:prospective-stages}
\begin{tabular}{ll}\toprule Stage & Recorded status \\
\midrule
Revised CPU calibration & Completed; passed \\
Excluded structural pilot & Completed; 72 cells \\
Structural power & Limited precision \\
$S_3$ power planning & Planning target met \\
Original native preparation & Stopped: clean competence \\
Input-only/generator development & Completed; excluded \\
Independent characterization & Source frozen; awaiting audit \\
Native/peer/cross-model extensions & Outside current run \\
\bottomrule\end{tabular}\end{table}
'''}
    if complete:
        content['prospective_results.tex']=content['prospective_results.tex'].replace('Awaiting audit','Audited scope').replace('Independent characterization slots.', 'Characterization review scope; numerical aggregates are separately curated.').replace('``Audited scope\'\' is not a zero, an estimate or a scientific pass. Development analyses do not fill these entries.', 'Raw/PCA and optimizer limitations remain; review is not a scientific pass.')
        content['prospective_status.tex']=content['prospective_status.tex'].replace('Source frozen; awaiting audit','Complete; compact audit bounded')
    manifest_path=root/'release/artifact_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    for name,text in content.items():
        path=root/'paper/generated'/name;path.write_text(text)
        key=str(path.relative_to(root))
        entry={'path':key,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
          'source_run':status['study_id']+'-status-only',
          'source_path':'release/current_study_status.json','role':'reserved-result-slots-and-recorded-process-status'}
        existing=next((i for i,e in enumerate(manifest['artifacts']) if e['path']==key),None)
        if existing is None:manifest['artifacts'].append(entry)
        else:manifest['artifacts'][existing]=entry
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    return {'status_tables':len(content),'numerical_results_promoted':False}

if __name__=='__main__':print(json.dumps(build(),indent=2))
