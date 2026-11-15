#!/usr/bin/env python3
"""Read-only diagnostics of immutable saved arrays. No model or fitting imports.
Inputs opened read-only; only --output receives files. All original coordinates,
means, PCA bases, operators and metrics remain untouched. A new numerical rank
rule is fixed in audit_protocol.json before invocation; historical rules are
reported separately. CSV NA means unavailable/not applicable, never zero.
"""
import argparse, collections, csv, hashlib, io, json, platform, sys, tarfile, zipfile
from pathlib import Path
import numpy as np

def sha_bytes(b): return hashlib.sha256(b).hexdigest()
def file_sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,x): p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
class Reader:
 def __init__(self,path,root):
  self.path=path;self.rel=str(path.relative_to(root));self.kind='zip' if zipfile.is_zipfile(path) else 'tar'
  self.used={};self.cache={}
  if self.kind=='zip':
   self.f=zipfile.ZipFile(path);self.names=self.f.namelist()
  else:
   self.f=None;self.names=[]
   cache_root=out/'_input_cache'/path.stem;cache_root.mkdir(parents=True,exist_ok=True)
   with tarfile.open(path,'r|gz') as tf:
    for m in tf:
     self.names.append(m.name)
     selected=m.isfile() and ('/legacy_full/' in m.name) and (m.name.endswith('/run_ledger.json') or ('/checkpoints/structural_' in m.name and m.name.endswith(('.npz','.csv'))))
     if selected:
      cache=cache_root/sha_bytes(m.name.encode())
      cache.write_bytes(tf.extractfile(m).read());self.cache[m.name]=cache
 def read(self,n):
  b=self.f.read(n) if self.kind=='zip' else self.cache[n].read_bytes()
  self.used[n]=sha_bytes(b);return b
 def json(self,n):return json.loads(self.read(n))
 def arrays(self,n):
  b=self.read(n)
  with np.load(io.BytesIO(b),allow_pickle=False) as z:return {k:z[k] for k in z.files}
 def uri(self,n):return self.rel+'!'+n
 def close(self):
  if self.f is not None:self.f.close()

P=argparse.ArgumentParser();P.add_argument('--repo',required=True);P.add_argument('--output',required=True);a=P.parse_args()
root=Path(a.repo).resolve();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
protocol=json.loads((out/'audit_protocol.json').read_text());dump(out/'logs'/'matrix_runtime.json',{'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),'protocol_sha256':file_sha(out/'audit_protocol.json')})
paths={
 'original_full':'runpod_output_by_date/result_9_6_2026/run_outputs_9_4_2026_full3.zip',
 'replay_full':'submission/archive/runpod-from-zero-v1/2026-10-22-full-completed/full-results-v1.tar.gz',
 'parent':'incoming/rank_recovery_v1/from-zero-v1-rank-recovery.results.zip',
 'followup':'incoming/tmlr_all_v2/tmlr-all-v2.20261030T203725Z-5489fd.results.zip'}
before={k:{'path':v,'bytes':(root/v).stat().st_size,'sha256':file_sha(root/v)} for k,v in paths.items()}
dump(out/'provenance'/'matrix_archives.before.json',before)
rows=[];checks=[];missing=[];readers=[]

def diag(A,meta,*,expected=None,centering='none; saved coordinates unchanged',evr=None):
 raw=np.asarray(A);M=np.asarray(raw,dtype=np.float64);q=M.shape[1];s=np.linalg.svd(M,compute_uv=False)
 threshold=max(M.shape)*np.finfo(np.float64).eps*s[0];rank=int(np.count_nonzero(s>threshold));want=q if expected is None else expected
 smallest=float(s[want-1]) if 0<want<=len(s) else 0.;ratio=smallest/float(s[0]) if s[0]>0 else 0.
 Sigma=M.T@M/len(M);ev=np.linalg.eigvalsh((Sigma+Sigma.T)/2)
 status='verified_deficient' if rank<want else ('borderline_or_ill_conditioned' if ratio<=1e-8 else 'verified_supported')
 row=dict(study='',run='',model='',domain='',readout='',split='',bank='',cell='',pca_identity='',input_identity='',matrix_kind='',evidence_source='',evidence_sha256='',shape=str(list(M.shape)),centering=centering,weights='uniform rows',q=want,numerical_rank=rank,sigma_max=float(s[0]),sigma_min_retained=smallest,sigma_min_all=float(s[-1]),singular_min_max_ratio=ratio,input_dtype=str(raw.dtype),compute_dtype='float64',tau_abs=0.,tau_rel=max(M.shape)*np.finfo(np.float64).eps,threshold=threshold,rank_rule='SVD: s > max(shape)*eps64*s_max; posthoc diagnostic',sigma_eigen_min=float(ev[0]),sigma_eigen_max=float(ev[-1]),sigma_eigen_ratio=float(ev[0]/ev[-1]),historical_rule='NA',historical_rank='NA',historical_rank_pass='NA',historical_whitening_pass='NA',whitening_residual='NA',EVR_q=evr if evr is not None else 'NA',EVR_basis='NA',status=status,notes='')
 row.update(meta);rows.append(row);return row

def full_archive(key):
 rd=Reader(root/paths[key],root);readers.append(rd)
 ledger_n=next(n for n in rd.names if n.endswith('/run_ledger.json') and ('main_full_003/' in n if key=='original_full' else '/legacy_full/' in n));ledger=rd.json(ledger_n);prefix=ledger_n.rsplit('/',1)[0]
 count=0
 for unit,info in sorted(ledger['units'].items()):
  kind,uid=unit.split(':',1);f=info.get('fields',{})
  if kind not in ('structural_base','structural_joint') or f.get('latent_dim')!=16:continue
  stem=f'{prefix}/checkpoints/{kind}/{uid}';n=stem+'.npz'
  if n not in rd.names:
   missing.append({'run':key,'unit':unit,'status':info['status'],'missing':n});continue
  ar=rd.arrays(n);sha=rd.used[n];expected_sha=next((x.get('sha256') for x in info.get('outputs',[]) if x['path'].endswith('/'+uid+'.npz')),None)
  if expected_sha and sha!=expected_sha:raise ValueError('ledger NPZ hash mismatch '+n)
  q=ar['z'].shape[1];assert q==16
  meta={'study':'Study 1','run':key,'model':f['model'],'domain':f['group'],'readout':f['layer'],'depth_slot':f['depth_slot'],'split':f['split_seed'],'cell':uid,'pca_identity':rd.uri(n)+'#pca_mean,pca_components','input_identity':sha,'evidence_source':rd.uri(n),'evidence_sha256':sha,'panel':'primary_depth_grid' if f['depth_slot'] in [.2,.4,.6,.8] else 'terminal_secondary'}
  csv_n=stem+'.csv';records=list(csv.DictReader(io.StringIO(rd.read(csv_n).decode())))
  held={int(x['template_id']) for x in records};assert len(held)==12
  states=7 if f['group']=='C7' else 4;ids=np.repeat(np.arange(40),states);assert len(ids)==len(ar['z'])
  masks={'train':~np.isin(ids,list(held)),'validation':np.isin(ids,sorted({int(x['template_id']) for x in records if x['data_split']=='validation'})),'test':np.isin(ids,sorted({int(x['template_id']) for x in records if x['data_split']=='test'}))}
  assert masks['train'].sum()==int(ar['pca_fit_rows'])==28*states
  # Original code's pca_explained is retained / COMPLETE training spectrum energy.
  evr=float(np.sum(ar['pca_explained']))
  if kind=='structural_base':
   for bank,mask in masks.items():
    r=diag(ar['z'][mask],dict(meta,bank=bank,matrix_kind='retained_PCA_coordinates',EVR_basis='saved pca_explained ratios from full training SVD; src/bsg/representations/pca.py:25-29'),evr=evr)
    r['notes']='Training mask is complement of recorded validation/test template IDs in complete 40-template bank; row order verified against dataset builder. No recentering. Full H_c rank not inferred from q.'
   r=diag(ar['pca_components'].T,dict(meta,bank='train',matrix_kind='PCA_basis',weights='not applicable; basis matrix',notes='Stored basis; not refitted'),evr=None)
  else:
   base_uid=next((k.split(':')[1] for k,v in ledger['units'].items() if k.startswith('structural_base:') and all(v.get('fields',{}).get(x)==f.get(x) for x in ['model','group','layer','latent_dim','split_seed','depth_slot'])),None)
   if base_uid:
    base_n=f'{prefix}/checkpoints/structural_base/{base_uid}.npz';base=rd.arrays(base_n)
    same=all(np.array_equal(ar[x],base[x]) for x in ['z','pca_mean','pca_components','pca_explained'])
    checks.append({'run':key,'cell':uid,'check':'joint_shares_saved_base_coordinates','passed':same,'base_unit':base_uid})
    if not same:raise ValueError('Joint coordinates differ '+n)
  for g,R in enumerate(ar['operators']):
   diag(R,dict(meta,bank='train_fit',matrix_kind=('Ridge_operator' if kind=='structural_base' else 'Joint_operator'),action_index=g,weights='not applicable; operator matrix',notes='Matrix rank diagnoses invertibility, not predictive validity; all elements including identity retained'),evr=None)
  count+=1
  if count%100==0:print('FULL_DIAG',key,count,flush=True)
 print('FULL_FINISHED',key,count,flush=True)

for k in ['original_full','replay_full']:full_archive(k)

parent=Reader(root/paths['parent'],root);follow=Reader(root/paths['followup'],root);readers.extend([parent,follow])
ns=follow.names
prep_prefix=next(n.rsplit('/',1)[0] for n in ns if n.startswith('extension-prepared-') and n.endswith('/prepared_manifest.json'))
prep=follow.json(prep_prefix+'/prepared_manifest.json')
spec_prefix=next(n.rsplit('/',1)[0] for n in ns if n.startswith('spectral-results-') and n.endswith('/input_manifest.json'))
manifest=follow.json(spec_prefix+'/input_manifest.json')
context_prefix=next(n.split('/')[0] for n in ns if n.startswith('context-') and '/inputs/' in n)
for item in manifest['cells']:
 cell=item['cell'];base='parent-from-zero-v1/characterization/analysis/'+cell
 pca_n=base+'/pca.npz';legacy_n=base+'/legacy.json';coord_n=base+'/coordinates.npz'
 pca=parent.arrays(pca_n);legacy=parent.json(legacy_n);rec=next(x for x in legacy if x['test_split']=='test');pc=rec['pca']
 spectrum=np.array(pc['singular_values'],dtype=float);assert len(spectrum)==min(pc['fit_rows'],len(pca['center']))
 historical=int(np.sum(spectrum>spectrum[0]*1e-12));assert historical==pc['observed_rank']
 evr=float(np.sum(spectrum[:16]**2)/np.sum(spectrum**2));assert abs(evr-pc['explained_variance'])<1e-12
 coords=parent.arrays(coord_n)['z'];coord_sha=parent.used[coord_n]
 expected_members=item['members'];assert expected_members[coord_n]==coord_sha;assert expected_members[pca_n]==parent.used[pca_n]
 meta={'study':'Study 2 and spectral follow-up','run':'from-zero-characterization','model':item['model_id'],'domain':item['domain'],'readout':item['layer'],'split':'fixed authored banks','cell':cell,'pca_identity':parent.uri(pca_n),'input_identity':parent.used[pca_n],'EVR_basis':'full saved centered training spectrum in '+parent.uri(legacy_n)+'[test].pca'}
 # Saved full spectrum reports original centered training rank without rerunning PCA.
 tol=max(pc['fit_rows'],len(pca['center']))*np.finfo(float).eps*spectrum[0];rank=int(np.sum(spectrum>tol))
 rows.append(dict(meta,bank='train',matrix_kind='H_centered_saved_full_spectrum',shape=str([pc['fit_rows'],len(pca['center'])]),q=16,numerical_rank=rank,sigma_max=float(spectrum[0]),sigma_min_retained=float(spectrum[15]),sigma_min_all=float(spectrum[-1]),singular_min_max_ratio=float(spectrum[15]/spectrum[0]),input_dtype='float64 recorded SVD',compute_dtype='float64',tau_abs=0,tau_rel=max(pc['fit_rows'],len(pca['center']))*np.finfo(float).eps,threshold=float(tol),rank_rule='posthoc rank from retained complete SVD spectrum',historical_rule='s > 1e-12*s_max; original PCA feasibility rule',historical_rank=historical,historical_rank_pass=historical>=16,centering='original training mean already applied for stored spectrum',weights='uniform training rows',EVR_q=evr,evidence_source=parent.uri(legacy_n),evidence_sha256=parent.used[legacy_n],status='verified_supported' if historical>=16 else 'verified_deficient',notes='Full raw training activations need not be reconstructed; saved spectrum and PCA metadata establish feasibility.'))
 n=prep_prefix+'/cells/'+cell+'/banks.npz';bank=follow.arrays(n);assert follow.used[n]==prep['files']['cells/'+cell+'/banks.npz']
 assert np.array_equal(bank['test'],coords[np.array(item['row_indices'],dtype=int)])
 # Development file binds full train/validation arrays to the retained original coordinates.
 dev_n=context_prefix+'/inputs/'+cell+'/development.npz';dev=follow.arrays(dev_n)
 assert np.array_equal(dev['train'].reshape(-1,16),bank['train'].reshape(-1,16))
 summary_n=spec_prefix+'/cells/'+cell+'/summary.json';summary=follow.json(summary_n)
 diag_n=spec_prefix+'/cells/'+cell+'/diagnostics.npz';saved=follow.arrays(diag_n)
 for name,A in [('train',bank['train']),('validation',dev['validation']),('test',bank['test'])]:
  source=dev_n if name=='validation' else n
  r=diag(A.reshape(-1,16),dict(meta,bank=name,matrix_kind='retained_PCA_coordinates',evidence_source=follow.uri(source),evidence_sha256=follow.used[source],EVR_basis=meta['EVR_basis']),evr=evr)
  r['historical_rule']='Sigma lambda_min > 1e-10*lambda_max; no additional centering'
  r['historical_rank_pass']=r['sigma_eigen_min']>1e-10*r['sigma_eigen_max']
  if name=='test':
   assert summary['status']=='complete'
   Sigma=A.reshape(-1,16).T@A.reshape(-1,16)/np.prod(A.shape[:-1])
   sigma_difference=float(np.max(np.abs(Sigma-saved['Sigma'])))
   X=A.reshape(-1,16)@saved['whitening'];res=float(np.linalg.norm(X.T@X/len(X)-np.eye(16),'fro')/4)
   r.update(historical_whitening_pass=summary['checks']['whitening_identity']['passed'],whitening_residual=res,recorded_whitening_residual=summary['checks']['whitening_identity']['residual'],recorded_sigma_ratio=summary['diagnostics']['sigma_min_max_ratio'])
   checks.append({'cell':cell,'check':'test_bank_and_whitening_match_saved_spectral_input','passed':sigma_difference<1e-10 and res<=1e-8,'sigma_max_abs_difference':sigma_difference,'whitening_residual':res})
   assert r['historical_rank_pass']==summary['checks']['second_moment_rank']['passed']
 diag(pca['basis'],dict(meta,bank='train',matrix_kind='PCA_basis',evidence_source=parent.uri(pca_n),evidence_sha256=parent.used[pca_n],weights='not applicable; basis'),evr=None)
 # Read ONLY the committed witness attempt, not an arbitrary first attempt.
 unit_n='witness/units/'+cell+'/receipt.json';unit=follow.json(unit_n)
 # Follow the committed receipt and verify saved array/summary bytes.
 assert unit['status']=='complete'
 wn='witness/'+unit['attempt']+'/diagnostics.npz';wa=follow.arrays(wn)
 assert follow.used[wn]==unit['files']['diagnostics.npz']
 ws_n=wn.replace('diagnostics.npz','summary.json');ws=follow.json(ws_n)
 assert follow.used[ws_n]==unit['files']['summary.json']
 Z=bank['train'].reshape(-1,16);X=Z@wa['train_whitening'];res=float(np.linalg.norm(X.T@X/len(X)-np.eye(16),'fro')/4)
 checks.append({'cell':cell,'check':'saved_train_whitening_identity','passed':res<=1e-8,'whitening_residual':res})
 trainrow=next(r for r in reversed(rows) if r.get('cell')==cell and r.get('bank')=='train' and r.get('matrix_kind')=='retained_PCA_coordinates')
 trainrow.update(whitening_residual=res,historical_whitening_pass='see preserved witness summary',witness_summary_source=follow.uri(wn.replace('diagnostics.npz','summary.json')))
 for key in ['ridge_operators','ols_operators','rho','raw_coordinate_operators']:
  for g,R in enumerate(wa[key]):diag(R,dict(meta,bank='train_fit',matrix_kind='witness_'+key,action_index=g,evidence_source=follow.uri(wn),evidence_sha256=follow.used[wn],weights='not applicable; operator matrix',notes='No fitting performed; raw and whitened actions reported separately.'),evr=None)
 print('SPECTRAL_DIAG',cell,flush=True)

cols=[]
for row in rows:
 for k in row:
  if k not in cols:cols.append(k)
with (out/'dimension_rank_audit_main.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=cols,restval='NA');w.writeheader();w.writerows(rows)
dump(out/'logs'/'matrix_checks.json',checks);dump(out/'logs'/'matrix_missing.json',missing)
inputs={r.rel:r.used for r in readers};dump(out/'provenance'/'matrix_members.sha256.json',inputs)
for r in readers:r.close()
after={k:{'path':v,'bytes':(root/v).stat().st_size,'sha256':file_sha(root/v)} for k,v in paths.items()}
dump(out/'provenance'/'matrix_archives.after.json',after)
if before!=after:raise ValueError('Input changed during audit')
if not all(x['passed'] for x in checks):raise ValueError('A binding or whitening check failed')
summary={'rows':len(rows),'statuses':dict(collections.Counter(r['status'] for r in rows)),'by_kind':dict(collections.Counter(r['matrix_kind'] for r in rows)),'checks_passed':len(checks),'missing':missing,'input_archives_unchanged':True}
dump(out/'logs'/'matrix_summary.json',summary);print(json.dumps(summary,indent=2),flush=True)
