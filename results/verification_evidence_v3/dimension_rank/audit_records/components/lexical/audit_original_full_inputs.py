"""SVD diagnostics of original FULL input matrices and saved PCA identity checks.

No PCA or operator fitting. Only evaluates the already-saved mean and basis.
"""
import argparse, csv, hashlib, io, json, zipfile
from pathlib import Path
import numpy as np

def sha(b): return hashlib.sha256(b).hexdigest()
def fsha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()

p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
protocol=json.loads((a.output/'full_input_protocol.json').read_text());assert protocol['relative_frobenius_tolerance']==1e-10
before=fsha(a.archive);assert before=='5fe7234572120073d8dfab2e8dc0d3373a741f8d48abadc92288f98ab16721f4'
rows=[];checks=[];used={};cache={};unavailable=[]
archive_label='runpod_packages/Beyond_Shared_Geometry_afs_d_rank_aware_immutable_inputs.zip!immutable/run_outputs_9_4_2026_full3.zip'
with zipfile.ZipFile(a.archive) as z:
 def read(n):
  b=z.read(n);used[n]=sha(b);return b
 def arrays(n):
  with np.load(io.BytesIO(read(n)),allow_pickle=False) as ar:return {k:ar[k] for k in ar.files}
 for n in z.namelist():
  if '/activation_cache/' not in n or not n.endswith('.npz'):continue
  b=z.read(n)
  with np.load(io.BytesIO(b),allow_pickle=False) as ar:
   m=json.loads(str(ar['metadata'].item()))
   if not m['group_task'].startswith('structural_'):continue
   key=(m['model_id'],m['group_task'].replace('structural_',''),int(m['layer']))
   if key in cache:raise ValueError('Ambiguous input cache '+str(key))
   assert m['prompt_template_split']['kind']=='template_pool' and m['template_ids']==list(range(40))
   cache[key]=(n,ar['activations'],m);used[n]=sha(b)
 ledger_n='run_outputs/main_experiment/main_full_003/run_ledger.json'
 ledger=json.loads(read(ledger_n));prefix=ledger_n.rsplit('/',1)[0]
 for key,u in sorted(ledger['units'].items()):
  f=u.get('fields',{})
  if not key.startswith('structural_base:') or f.get('latent_dim')!=16:continue
  uid=key.split(':')[1];n=f'{prefix}/checkpoints/structural_base/{uid}.npz';stem=n[:-4]
  cachekey=(f['model'],f['group'],int(f['layer']))
  base=dict(study='Study 1',run='original_full',model=f['model'],domain=f['group'],readout=f['layer'],depth_slot=f['depth_slot'],split=f['split_seed'],bank='train',cell=uid,panel='primary_depth_grid' if f['depth_slot'] in [.2,.4,.6,.8] else 'terminal_secondary',q=16,matrix_kind='raw_training_H_centered')
  if cachekey not in cache:
   rows.append(dict(base,status='unavailable',notes='No unambiguous complete raw activation cache'));unavailable.append(base);continue
  cn,H,cm=cache[cachekey];ar=arrays(n)
  histsha=next(x['sha256'] for x in u['outputs'] if x['path'].endswith(uid+'.npz'));assert used[n]==histsha
  records=list(csv.DictReader(io.StringIO(read(stem+'.csv').decode())))
  held=sorted({int(x['template_id']) for x in records});assert len(held)==12
  states=7 if f['group']=='C7' else 4;ids=np.repeat(np.arange(40),states);assert len(ids)==len(H)
  train=~np.isin(ids,held);assert train.sum()==int(ar['pca_fit_rows'])==28*states
  M=np.asarray(H[train],dtype=np.float64);mean=M.mean(axis=0)
  centered=M-mean;s=np.linalg.svd(centered,compute_uv=False,full_matrices=False)
  tol=max(centered.shape)*np.finfo(np.float64).eps*s[0];rank=int(np.sum(s>tol));ratio=float(s[15]/s[0])
  total=float(np.sum(centered*centered));evr=float(np.dot(s[:16],s[:16])/total)
  # Verification of ORIGINAL saved PCA, not fitting or replacing its basis.
  projected=(np.asarray(H,dtype=np.float64)-ar['pca_mean'])@ar['pca_components'].T
  proj_res=float(np.linalg.norm(projected-ar['z'],'fro')/max(1.,np.linalg.norm(ar['z'],'fro')))
  mean_res=float(np.linalg.norm(mean-ar['pca_mean'])/max(1.,np.linalg.norm(ar['pca_mean'])))
  evr_delta=abs(evr-float(np.sum(ar['pca_explained'])))
  passed=max(proj_res,mean_res,evr_delta)<=1e-10
  checks.append(dict(cell=uid,model=f['model'],domain=f['group'],layer=f['layer'],split=f['split_seed'],passed=passed,raw_cache_member=cn,fit_member=n,
                     raw_to_saved_Z_relative_residual=proj_res,training_mean_relative_residual=mean_res,EVR_absolute_difference=evr_delta,threshold=1e-10))
  row=dict(base,pca_identity=archive_label+'!'+n+'#pca_mean,pca_components',input_identity=used[cn],evidence_source=archive_label+'!'+cn,evidence_sha256=used[cn],shape=str(list(centered.shape)),centering='original arithmetic training-row mean; verified against saved PCA mean',weights='uniform training state-template rows',
           numerical_rank=rank,sigma_max=float(s[0]),sigma_min_retained=float(s[15]),sigma_min_all=float(s[-1]),singular_min_max_ratio=ratio,
           input_dtype=str(H.dtype),compute_dtype='float64',tau_abs=0.,tau_rel=max(centered.shape)*np.finfo(np.float64).eps,threshold=float(tol),rank_rule='SVD: s > max(shape)*eps64*s_max; read-only diagnostic',
           EVR_q=evr,EVR_basis='full original centered training matrix Frobenius energy',historical_rule='source PCASubspace full SVD; no modern observed-rank guard claimed for original FULL',
           status='verified_deficient' if rank<16 else ('borderline_or_ill_conditioned' if ratio<=1e-8 else 'verified_supported'),
           notes='Existing saved PCA formula agrees with saved coordinates; no fitting. Full input rank is distinct from retained dimension.',raw_to_saved_Z_relative_residual=proj_res,raw_to_saved_PCA_binding_pass=passed)
  rows.append(row)
  if len(rows)%25==0:print('FULL_RAW_INPUT',len(rows),flush=True)
 if len(rows)!=300:raise RuntimeError('Expected 300 q16 base fits')
keys=list(dict.fromkeys(k for r in rows for k in r))
with (a.output/'original_full_input_rank_audit.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=keys,restval='NA');w.writeheader();w.writerows(rows)
after=fsha(a.archive);assert before==after
for name,obj in [('full_input_bindings.json',checks),('full_input_members.sha256.json',used),('full_input_preservation.json',{'archive_sha256_before':before,'archive_sha256_after':after,'unchanged':before==after,'checks':len(checks),'all_bindings_pass':all(x['passed'] for x in checks),'unavailable':unavailable,'new_pca_fits':0,'new_model_forwards':0})]:
 (a.output/name).write_text(json.dumps(obj,indent=2)+'\n')
assert all(x['passed'] for x in checks)
print('COMPLETE',len(rows),'raw-input diagnostics;',len(checks),'saved-PCA bindings verified',flush=True)
