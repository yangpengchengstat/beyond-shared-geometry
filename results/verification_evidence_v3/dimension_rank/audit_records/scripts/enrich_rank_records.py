"""Add recorded historical checks to derived CSV; no numerical recomputation."""
from pathlib import Path
import csv,json,zipfile,hashlib,argparse
p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.repo);out=Path(a.output)
f=out/'dimension_rank_audit_main.csv';rows=list(csv.DictReader(f.open()))
archive=root/'incoming/tmlr_all_v2/tmlr-all-v2.20261030T203725Z-5489fd.results.zip'
recorded={}
with zipfile.ZipFile(archive) as z:
 for n in z.namelist():
  if n.startswith('witness/units/') and n.endswith('/receipt.json'):
   receipt=json.loads(z.read(n));sp='witness/'+receipt['attempt']+'/summary.json';raw=z.read(sp)
   assert hashlib.sha256(raw).hexdigest()==receipt['files']['summary.json']
   s=json.loads(raw)['train_spectral'];recorded[receipt['unit']]=(s,sp)
 for r in rows:
  if r['run']=='from-zero-characterization' and r['matrix_kind']=='retained_PCA_coordinates':
   r['passes_original_second_moment_rule_on_recheck']=r['historical_rank_pass']
   r['historical_rank_pass']='NA'  # validation has no original recorded spectral check
   if r['bank']=='train':
    s,sp=recorded[r['cell']];r['historical_rank_pass']=s['checks']['second_moment_rank']['passed'];r['historical_whitening_pass']=s['checks']['whitening_identity']['passed'];r['recorded_whitening_residual']=s['checks']['whitening_identity']['residual'];r['historical_check_source']=str(archive.relative_to(root))+'!'+sp+'#train_spectral'
   elif r['bank']=='test':
    sp=next(n for n in z.namelist() if n.startswith('spectral-results-') and n.endswith('/cells/'+r['cell']+'/summary.json'));s=json.loads(z.read(sp));r['historical_rank_pass']=s['checks']['second_moment_rank']['passed'];r['historical_check_source']=str(archive.relative_to(root))+'!'+sp
   else:r['historical_check_source']='NA; diagnostic reapplication only'
  if r['matrix_kind'] in ['PCA_basis','Ridge_operator','Joint_operator'] or r['matrix_kind'].startswith('witness_'):
   for k in ['sigma_eigen_min','sigma_eigen_max','sigma_eigen_ratio']:
    r['diagnostic_gram_'+k]=r[k];r[k]='NA'
   r['notes']+=' Gram values for this operator/basis are NOT observation-bank Sigma.'
  if r.get('EVR_q') not in ('NA','',None):r['EVR_scope']='original training PCA, even when displayed with validation/test coordinates'
cols=list(dict.fromkeys(k for r in rows for k in r))
with f.open('w',newline='') as h:
 w=csv.DictWriter(h,fieldnames=cols,restval='NA');w.writeheader();w.writerows(rows)
before=json.loads((out/'provenance/matrix_archives.before.json').read_text());checks=[]
for k in ['parent','followup']:
 fp=root/before[k]['path'];side=fp.with_name(fp.name+'.sha256');expected=side.read_text().split()[0];assert before[k]['sha256']==expected;checks.append({'archive':before[k]['path'],'sidecar_sha256':expected,'passed':True})
(out/'logs/archive_sidecar_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
print('CHECK_FIELDS_ENRICHED',len(rows),'ARCHIVE_SIDECARS_VERIFIED',len(checks))
