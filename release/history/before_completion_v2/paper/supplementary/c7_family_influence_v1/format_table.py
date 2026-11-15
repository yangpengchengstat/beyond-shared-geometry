#!/usr/bin/env python3
"""Render a table from saved scalar summaries, with no matrix computation."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ORDER=[('meta-llama__Llama-3.2-1B','Llama-3.2-1B'),
       ('meta-llama__Llama-3.2-3B','Llama-3.2-3B'),
       ('meta-llama__Llama-3.1-8B','Llama-3.1-8B'),
       ('Qwen__Qwen2.5-1.5B','Qwen2.5-1.5B'),
       ('Qwen__Qwen2.5-3B','Qwen2.5-3B'),
       ('Qwen__Qwen2.5-7B','Qwen2.5-7B')]

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def number(text):
    value=float(text)
    if value==0:return '$0$'
    mantissa,exponent=f'{value:.3e}'.split('e')
    return '$'+mantissa+r'\times10^{'+str(int(exponent))+'}$'

def render():
    with (HERE/'cell_influence_summary.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    execution=json.loads((HERE/'execution.json').read_text())
    if execution['status']!='complete' or execution['complete']!=1536 or execution['failed']!=0:
        raise ValueError('This presentation requires the recorded complete1536-unit analysis')
    if len(rows)!=24 or len({x['cell'] for x in rows})!=24:raise ValueError('Incomplete saved cell summary')
    lines=[r'\begin{tabular}{@{}lrrrrr@{}}',r'\toprule',
      r'Model & Layer & Baseline $\ell^-$ & Deletion min. & Deletion max. & \shortstack{Above tol.\\(/64)} \\',r'\midrule']
    used=[]
    for model,label in ORDER:
        group=sorted([row for row in rows if row['cell'].startswith(model+'__C7__')],key=lambda x:int(x['cell'].rsplit('__',1)[1]))
        if len(group)!=4:raise ValueError('Unexpected model/readout coverage')
        if used:lines.append(r'\addlinespace[2pt]')
        for j,row in enumerate(group):
            if int(row['planned'])!=64 or int(row['failed'])!=0:raise ValueError('Unexpected deletion coverage')
            used.append(row['cell'])
            cells=[label if j==0 else '',row['cell'].rsplit('__',1)[1],
                number(row['baseline_extra_law_lower']),number(row['minimum_after_deletion']),
                number(row['maximum_after_deletion']),row['positive_count']+'/64']
            lines.append(' & '.join(cells)+r' \\')
    if set(used)!={r['cell'] for r in rows}:raise ValueError('Cell omitted or repeated')
    lines += [r'\bottomrule',r'\end{tabular}']
    return '\n'.join(lines)+'\n'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--receipt',type=Path);args=parser.parse_args()
    text=render()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:f.write(text)
    if args.receipt:
        result={'operation':'format_completed_scalar_summaries_only','new_scientific_analysis':False,
           'input_sha256':{name:sha(HERE/name) for name in ['cell_influence_summary.csv','execution.json']},
           'script_sha256':sha(Path(__file__)), 'output_sha256':sha(args.output),'rows':24,
           'rounding':'four significant digits in displayed endpoints; complete input precision retained'}
        with args.receipt.open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')

if __name__=='__main__':main()
