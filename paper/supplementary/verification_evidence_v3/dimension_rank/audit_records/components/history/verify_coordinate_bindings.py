#!/usr/bin/env python3
"""Supplementary byte/row binding checks. No SVD, refits, model imports or writes to inputs."""
from pathlib import Path
import argparse, ast, hashlib, io, json, subprocess, zipfile
import numpy as np

def sha(b):return hashlib.sha256(b).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();root=a.repo.resolve();a.out.mkdir(parents=True,exist_ok=True)
 records=[];used={}
 def read(z,n):
  b=z.read(n);used[str(Path(z.filename).relative_to(root))+'!'+n]=sha(b);return b
 def jread(z,n):return json.loads(read(z,n))
 def arrays(z,n):
  with np.load(io.BytesIO(read(z,n)),allow_pickle=False) as f:return {k:f[k] for k in f.files}
 parent_path=root/'incoming/rank_recovery_v1/from-zero-v1-rank-recovery.results.zip'
 follow_path=root/'incoming/tmlr_all_v2/tmlr-all-v2.20261030T203725Z-5489fd.results.zip'
 with zipfile.ZipFile(parent_path) as parent,zipfile.ZipFile(follow_path) as follow:
  names=follow.namelist();ep=next(n.rsplit('/',1)[0] for n in names if n.startswith('extension-prepared-') and n.endswith('/prepared_manifest.json'))
  cp=next(n.rsplit('/',1)[0] for n in names if n.startswith('context-') and n.endswith('/input_manifest.json'))
  em=jread(follow,ep+'/prepared_manifest.json');cm=jread(follow,cp+'/input_manifest.json');assert len(cm['cells'])==48
  for cell in cm['cells']:
   assert cell['status']=='prepared';name=cell['cell'];base='parent-from-zero-v1/characterization/analysis/'+name
   rawrows=read(parent,base+'/rows.json');rawcoords=read(parent,base+'/coordinates.npz')
   assert sha(rawrows)==cell['input_members'][base+'/rows.json'];assert sha(rawcoords)==cell['input_members'][base+'/coordinates.npz']
   rows=json.loads(rawrows)['rows']
   with np.load(io.BytesIO(rawcoords),allow_pickle=False) as f:z=f['z']
   assert len(rows)==len(z);assert z.shape[1]==16
   bn=ep+'/cells/'+name+'/banks.npz';rn=ep+'/cells/'+name+'/rows.json'
   bank=arrays(follow,bn);rr=jread(follow,rn)
   assert used[str(follow_path.relative_to(root))+'!'+bn]==em['files']['cells/'+name+'/banks.npz']
   assert used[str(follow_path.relative_to(root))+'!'+rn]==em['files']['cells/'+name+'/rows.json']
   for split in ('train','test'):
    assert np.array_equal(bank[split],z[np.asarray(rr[split+'_indices'],dtype=int)])
   dn=cp+'/'+cell['paths']['development.npz'];dev=arrays(follow,dn)
   assert used[str(follow_path.relative_to(root))+'!'+dn]==cm['prepared_files'][cell['paths']['development.npz']]
   n={'C7':7,'D4':4}[cell['domain']]
   details={}
   for split,nfamilies in [('train',48),('validation',16)]:
    d={}
    for i,r in enumerate(rows):
     assert r['domain']==cell['archive_row_domain']
     if r['split']==split:
      key=(r['family_id'],r['paraphrase_id'],r['state_id']);assert key not in d;d[key]=i
    fs=[f'structural_{split}_{i:03d}' for i in range(nfamilies)]
    keys=[(family,para,state) for family in fs for para in (0,1) for state in range(n)]
    assert set(d)==set(keys)
    ids=np.asarray([d[k] for k in keys],dtype=int);expected=z[ids].reshape(nfamilies,2,n,16)
    assert np.array_equal(expected,dev[split])
    if split=='train':assert np.array_equal(expected.reshape(-1,n,16),bank['train'])
    details[split]={'shape':list(expected.shape),'source_rows':len(ids),'exact_array_equal':True,'indices_sha256':sha(ids.tobytes())}
   records.append({'cell':name,'all_prepared_member_hashes_match':True,'parent_member_hashes_match':True,'prepared_train_and_test_match_parent_indices':True,'development_arrays_match_parent_rows_sorted_family_para_state':details})
  print('COORDINATE_BINDINGS_VERIFIED',len(records),flush=True)
 # Original FULL row-order source is tied to the actual saved run's clean Git identity.
 full_path=root/'runpod_output_by_date/result_9_6_2026/run_outputs_9_4_2026_full3.zip'
 with zipfile.ZipFile(full_path) as f:
  manifest=jread(f,'run_outputs/main_experiment/main_full_003/manifest.json');assert manifest['git_dirty'] is False
  commit=manifest['git_commit'];assert commit=='30f3c6369a185f43e94b17ca03b0499cd5d65a78'
 old={}
 for name in ['src/bsg/data/c7.py','src/bsg/data/d4.py','src/bsg/representations/extract.py','src/bsg/representations/pca.py','src/bsg/experiments/canonical_units.py']:
  b=subprocess.run(['git','show',commit+':'+name],cwd=root,capture_output=True,check=True).stdout
  old[name]=b
 # Replay FULL source archive is tied to recorded source_manifest SHA in identity.json.
 archive_dir=root/'submission/archive/runpod-from-zero-v1/2026-10-22-full-completed'
 identity=json.loads((archive_dir/'evidence/identity.json').read_text())
 sourcezip=archive_dir/'source/bsg-from-zero-v1.zip';source_data=sourcezip.read_bytes()
 recorded_zip_sha=(archive_dir/'source/bsg-from-zero-v1.zip.sha256').read_text().split()[0];assert sha(source_data)==recorded_zip_sha
 with zipfile.ZipFile(sourcezip) as s:
  sm_bytes=read(s,'beyond-shared-geometry/studies/reproduction_v1/source_manifest.json');assert sha(sm_bytes)==identity['source_sha256'];sm=json.loads(sm_bytes)
  sources=[]
  for name,b in old.items():
   rb=read(s,'beyond-shared-geometry/'+name);assert sha(rb)==sm['files'][name]
   item={'path':name,'original_commit':commit,'original_sha256':sha(b),'replay_sha256':sha(rb),'same_bytes':b==rb,'replay_manifest_verified':True}
   if name.endswith(('/c7.py','/d4.py')):
    for label,data in [('original',b),('replay',rb)]:
     tree=ast.parse(data);constants={}
     for x in tree.body:
      if isinstance(x,ast.Assign) and len(x.targets)==1 and isinstance(x.targets[0],ast.Name) and x.targets[0].id in ('TEMPLATES','STATES','DAYS'):constants[x.targets[0].id]=ast.literal_eval(x.value)
     func=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='build_dataset');lc=next(x for x in ast.walk(func) if isinstance(x,ast.ListComp));generators=[ast.unparse(x.iter) for x in lc.generators]
     assert generators[0]=='enumerate(TEMPLATES)';assert generators[1] in ('enumerate(DAYS)','enumerate(STATES)');assert len(constants['TEMPLATES'])==40
     states=constants.get('DAYS',constants.get('STATES'));assert len(states)==(7 if name.endswith('c7.py') else 4)
     item[label+'_row_order']={'templates':40,'states':len(states),'rows':40*len(states),'outer_generator':generators[0],'inner_generator':generators[1],'template_major_state_minor':True}
   if name.endswith('/extract.py'):
    original_functions={x.name:ast.dump(x,include_attributes=False) for x in ast.parse(b).body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef))}
    replay_functions={x.name:ast.dump(x,include_attributes=False) for x in ast.parse(rb).body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef))}
    same_functions=all(replay_functions.get(k)==v for k,v in original_functions.items());assert same_functions
    item['original_extraction_function_ASTs_unchanged']=same_functions
    item['replay_added_functions']=sorted(set(replay_functions)-set(original_functions))
   sources.append(item)
  print('FULL_ROW_ORDER_SOURCE_VERIFIED',len(sources),flush=True)
 result={'status':'all_checks_passed','no_fitting_no_SVD_no_models':True,'coordinate_cells':records,'full_source_binding':{'original_manifest_git_clean':True,'original_commit':commit,'replay_source_archive_sha256':sha(source_data),'replay_source_manifest_sha256':sha(sm_bytes),'files':sources},'read_member_sha256':used}
 (a.out/'coordinate_binding_checks.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'status':result['status'],'coordinate_cells':len(records),'source_files':len(sources)},indent=2))
if __name__=='__main__':main()
