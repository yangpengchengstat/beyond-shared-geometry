from pathlib import Path
import csv,json,collections,shutil
p=Path(__file__).resolve().parents[1]
main=list(csv.DictReader((p/'dimension_rank_audit_main.csv').open()))
lex=list(csv.DictReader((p/'components/lexical/lexical_dimension_rank_audit.csv').open()))
for r in lex:
 r['run']='afs_d_rank_aware_001';r['singular_min_max_ratio']=r.pop('sigma_retained_ratio');r['sigma_min_all']=r.pop('sigma_min_full');r['rank_rule']=r.pop('threshold_rule');r['EVR_q']=r.pop('evr_q');r['EVR_basis']=r.pop('evr_denominator');r['notes']=r.pop('interpretation')
full=main+lex
extra=p/'components/lexical/original_full_input_rank_audit.csv'
if extra.exists():
 for r in csv.DictReader(extra.open()):full.append(r)
cols=[]
for r in full:
 for k in r:
  if k not in cols:cols.append(k)
for r in full:
 r['status']=r.get('status') or 'unavailable'
with (p/'dimension_rank_audit.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=cols,restval='NA');w.writeheader();w.writerows(full)
for name in ['dimension_sensitivity_existing.csv','dimension_sensitivity_summary.csv','dimension_sensitivity_coverage.csv']:
 shutil.copyfile(p/'components/sensitivity/output'/name,p/name)

# Round only presentation tables; all full precision values remain in CSV.
ss=list(csv.DictReader((p/'dimension_sensitivity_summary.csv').open()))
lines=[]
for domain in ['C7','D4']:
 for method,label in [('independent_ridge','Ridge'),('group_consistent_joint','Joint')]:
  rr=[x for x in ss if x['run']=='main_full_003' and x['bank']=='test' and x['domain']==domain and x['method']==method]
  if not rr:
   rr=[x for x in ss if 'original' in x['run'] and x['bank']=='test' and x['domain']==domain and x['method']==method]
  assert len(rr)==4,(domain,method,len(rr))
  vals={int(x['q']):float(x['E_eq']) for x in rr}
  lines.append('$'+domain.replace('7','_7').replace('4','_4')+'$ & '+label+' & '+' & '.join(f'{vals[q]:.6f}' for q in [8,16,32,64])+r' \\')
tex=r'''\begin{table}[htbp]
\centering\small
\begin{tabular}{@{}llrrrr@{}}
\toprule
Domain & Fit & $q=8$ & $q=16$ (matched) & $q=32$ & $q=64$ \\
\midrule
'''+ '\n'.join(lines)+r'''
\bottomrule
\end{tabular}
\caption{Existing Study~1 test prediction errors on a matched dimension-sensitivity subset, reconstructed from unchanged per-template records. Each domain averages six models, depths 0.4/0.8 and splits 101/303 (24 conditions), with the same six held-out templates per split. The $q=16$ column is this subset, not the larger primary panel. Values retain the original $\Eeq$ normalization; no new dimension was run or selected. The separate computational replay and all recorded endpoints are retained in the audit CSV.}
\label{tab:dimension-sensitivity-audit}
\end{table}
'''
(p/'paper/generated/dimension_sensitivity_audit.tex').write_text(tex)
summary={'rows':len(full),'statuses':dict(collections.Counter(x['status'] for x in full)),'spectral':{},'full':{}}
for bank in ['train','test']:
 rr=[x for x in main if x['run']=='from-zero-characterization' and x['matrix_kind']=='retained_PCA_coordinates' and x['bank']==bank]
 summary['spectral'][bank]={k:[min(float(x[k]) for x in rr),max(float(x[k]) for x in rr)] for k in ['singular_min_max_ratio','sigma_eigen_min','sigma_eigen_max','sigma_eigen_ratio','whitening_residual','EVR_q']}
for run in ['original_full','replay_full']:
 rr=[x for x in main if x['run']==run and x['matrix_kind']=='retained_PCA_coordinates' and x['bank']=='train']
 summary['full'][run]={'count':len(rr),'primary_count':sum(x['panel']=='primary_depth_grid' for x in rr),'EVR_range':[min(float(x['EVR_q']) for x in rr),max(float(x['EVR_q']) for x in rr)],'sratio_range':[min(float(x['singular_min_max_ratio']) for x in rr),max(float(x['singular_min_max_ratio']) for x in rr)]}
(p/'logs/combined_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print('AUDIT_TABLES_READY',summary['rows'],summary['statuses'])
