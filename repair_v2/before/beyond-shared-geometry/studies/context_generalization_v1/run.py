#!/usr/bin/env python3
"""New-context transfer, using old PCA/train/validation and opt-in model execution."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import sys
import time
import zipfile
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SPECTRAL=ROOT/'studies/spectral_finite_bank_v1'
FROZEN=ROOT/'studies/characterization_v1/frozen_run/src'
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(SPECTRAL))
sys.path.insert(0,str(FROZEN))
from bank import build_bank, validate_bank
from analysis import blocks, fit_predictors, evaluate, aggregate
_spec=importlib.util.spec_from_file_location('bsg_spectral_input_adapter',SPECTRAL/'runner.py')
input_adapter=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(input_adapter)
from gamma_r import group_spec
sha=input_adapter.sha
file_sha=input_adapter.file_sha
canonical=input_adapter.canonical
read_json=lambda p:input_adapter._json(Path(p).read_bytes())


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.partial')
    with temporary.open('w') as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)


def save_npz(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.partial.npz')
    np.savez_compressed(temporary,**arrays);os.replace(temporary,path)


@contextmanager
def lock(run):
    import fcntl
    run=Path(run)
    if run.is_symlink():raise ValueError('Run directory cannot be a symlink')
    run.mkdir(parents=True,exist_ok=True)
    with (run/'.writer.lock').open('a+') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as e:raise RuntimeError('Another study process owns this run') from e
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)


def source_lock():
    record=read_json(HERE/'source_manifest.json')
    actual={str(p.relative_to(HERE)) for p in HERE.rglob('*') if p.is_file()
            and p.name!='source_manifest.json' and '__pycache__' not in p.parts
            and '.pytest_cache' not in p.parts and p.suffix not in ('.pyc','.pyo')}
    if actual!=set(record['files']):raise ValueError('Study source coverage changed')
    for base,key in ((HERE,'files'),(ROOT,'dependencies')):
        for name,digest in record.get(key,{}).items():
            input_adapter.safe_member(name)
            path=base/name
            if path.is_symlink() or not path.is_file() or file_sha(path)!=digest:
                raise ValueError('Study source identity changed: '+name)
    input_adapter.source_lock()
    return file_sha(HERE/'source_manifest.json')


def spec():return read_json(HERE/'protocol.json')


def _receipt(run):
    run=Path(run);record=read_json(run/'input_manifest.json')
    if (run/'input_manifest.sha256').read_text().strip()!=file_sha(run/'input_manifest.json'):
        raise ValueError('Prepared input manifest changed')
    if record['source_manifest_sha256']!=source_lock() or record['protocol']!=spec():
        raise ValueError('Study source/protocol changed since preparation')
    bank=read_json(run/'bank.json')['rows'];validate_bank(bank)
    if sha(canonical(bank))!=record['bank_sha256'] or bank!=build_bank():raise ValueError('Frozen bank changed')
    for name,digest in record['prepared_files'].items():
        input_adapter.safe_member(name)
        if file_sha(run/name)!=digest:raise ValueError('Prepared input changed: '+name)
    return record,bank


def prepare(archive,expected_sha256,output):
    output=Path(output)
    if output.exists():raise ValueError('Preparation refuses an existing output; use a fresh namespace')
    identity=source_lock();protocol=spec();spectral=input_adapter.protocol()
    audited=input_adapter.inspect_archive(Path(archive),expected_sha256,spectral)
    bank=build_bank();bank_sha=sha(canonical(bank))
    output.mkdir(parents=True)
    atomic(output/'bank.json',{'rows':bank,'authorship':'assistant_authored_fixed_frames',
                              'human_semantic_review':'pending','bank_sha256':bank_sha})
    review_lines=['Human review of 64 paired frames per domain',
                  'Check both renderings preserve the same stated fact. These are new narrative wrappers around fixed factual sentences.',
                  'Check the target span, no competing state token, and no implied change of the stated weekday/corner. Review every frame; do not use model output.', '']
    for domain in ('C7','D4'):
        for family in range(64):
            chosen=[r for r in bank if r['domain']==domain and r['family_id']==f'new_context_{family:03d}' and r['state_id']==0]
            review_lines.append(domain+' / '+f'new_context_{family:03d}')
            review_lines.extend(str(r['paraphrase_id'])+': '+r['text'] for r in chosen)
            review_lines.append('')
    (output/'BANK_REVIEW.txt').write_text('\n'.join(review_lines)+'\n')
    atomic(output/'source_archive_audit.json',audited)
    atomic(output/'protocol.json',protocol)
    files={'BANK_REVIEW.txt':file_sha(output/'BANK_REVIEW.txt'),'bank.json':file_sha(output/'bank.json'),'source_archive_audit.json':file_sha(output/'source_archive_audit.json'),
           'protocol.json':file_sha(output/'protocol.json')}
    cells=[];overlap=set()
    with zipfile.ZipFile(archive) as z:
        index=input_adapter.zip_index(z)
        for item in audited['cells']:
            cell={k:item[k] for k in ('cell','domain','model_id','layer')}
            if item['status']!='prepared_metadata_only':
                cells.append({**cell,'status':'not_evaluable_input','reason':item['reason']});continue
            try:
                prefix=spectral['archive_prefix']+'/analysis/'+item['cell']
                raw={n:input_adapter.read_member(z,index,prefix+'/'+n,input_adapter.MAX_NPZ_BYTES if n.endswith('.npz') else input_adapter.MAX_JSON_BYTES)
                     for n in ('coordinates.npz','pca.npz','rows.json')}
                if any(sha(raw[n])!=item['members'][prefix+'/'+n] for n in raw):raise ValueError('Archive changed during use')
                rows=input_adapter._json(raw['rows.json'])['rows']
                if any(not isinstance(r.get('text'),str) for r in rows):raise ValueError('Prompt text missing; overlap cannot be checked')
                oldtexts={r['text'] for r in rows};overlap|=oldtexts&{r['text'] for r in bank}
                with np.load(io.BytesIO(raw['coordinates.npz']),allow_pickle=False) as f:zcoords=f['z']
                with np.load(io.BytesIO(raw['pca.npz']),allow_pickle=False) as f:center=f['center'];basis=f['basis']
                if not np.isfinite(zcoords).all() or not np.isfinite(center).all() or not np.isfinite(basis).all():raise ValueError('Nonfinite old coordinate/PCA')
                if np.linalg.norm(basis.T@basis-np.eye(16),'fro')>1e-6:raise ValueError('Saved PCA basis is not orthonormal')
                train,_=blocks(zcoords,rows,item['domain'],'train',[f'structural_train_{i:03d}' for i in range(48)])
                validation,_=blocks(zcoords,rows,item['domain'],'validation',[f'structural_validation_{i:03d}' for i in range(16)])
                folder=output/'inputs'/item['cell'];folder.mkdir(parents=True)
                save_npz(folder/'development.npz',train=train,validation=validation)
                save_npz(folder/'pca.npz',center=center,basis=basis)
                mergedraw=input_adapter.read_member(z,index,item['merged_manifest'])
                if sha(mergedraw)!=item['merged_manifest_sha256']:raise ValueError('Old extraction identity changed')
                oldruntime=input_adapter._json(mergedraw)['model']
                atomic(folder/'old_runtime.json',oldruntime)
                paths={n:str((folder/n).relative_to(output)) for n in ('development.npz','pca.npz','old_runtime.json')}
                files.update({v:file_sha(output/v) for v in paths.values()})
                cells.append({**cell,'status':'prepared','paths':paths,'input_members':item['members'],
                              'revision':item['revision'],'tokenizer_revision':item['tokenizer_revision']})
            except (ValueError,KeyError,TypeError,zipfile.BadZipFile) as e:
                cells.append({**cell,'status':'not_evaluable_input','reason':str(e)})
    if file_sha(archive)!=expected_sha256:raise ValueError('Archive changed during preparation')
    if overlap:raise ValueError('New bank overlaps old prompt text; preparation cannot proceed')
    record={'schema_version':1,'protocol':protocol,'source_manifest_sha256':identity,'bank_sha256':bank_sha,
            'parent_archive_sha256':expected_sha256,'prepared_files':files,'cells':cells,
            'exact_prompt_overlap_count':0,'human_semantic_review':'pending',
            'no_old_test_rows_saved_for_fitting':True,'new_model_execution':False}
    atomic(output/'input_manifest.json',record)
    (output/'input_manifest.sha256').write_text(file_sha(output/'input_manifest.json')+'\n')
    atomic(output/'state.json',{'stage':'prepared','prepared_cells':sum(c['status']=='prepared' for c in cells),
                               'planned_cells':48,'human_semantic_review':'pending','paper_results_available':False})
    print('PREPARED_BANK_SHA256',bank_sha,flush=True)
    print('REVIEW bank.json before recording an attestation. No model was loaded.',flush=True)
    return record


def review(run,bank_sha256,attest_reviewed=False):
    if not attest_reviewed:raise ValueError('Explicit --attest-reviewed is required after actual review')
    record,bank=_receipt(run)
    if bank_sha256!=record['bank_sha256']:raise ValueError('Reviewed bank hash differs')
    path=Path(run)/'human_review.json'
    value={'bank_sha256':bank_sha256,'source_manifest_sha256':record['source_manifest_sha256'],
           'attestation':'User confirms checking all 64 paired frames for each domain, meaning preservation, unique target spans, and no answer leakage.',
           'mechanism':'explicit operator attestation, not an authenticated digital signature',
           'authorship':'assistant_authored_and_user_reviewed','approval_before_model_execution':True}
    if path.exists() and read_json(path)!=value:raise ValueError('Existing review differs')
    atomic(path,value)
    return value


def _check_review(run,record):
    p=Path(run)/'human_review.json'
    if not p.exists():raise ValueError('Human semantic review is pending; inspect bank.json and run review')
    review=read_json(p)
    if review.get('bank_sha256')!=record['bank_sha256'] or review.get('source_manifest_sha256')!=record['source_manifest_sha256']:
        raise ValueError('Human review is for a different bank/source')


def _verify_artifacts(run,record):
    for name,digest in record['artifacts'].items():
        input_adapter.safe_member(name)
        if file_sha(Path(run)/name)!=digest:raise ValueError('Committed artifact changed: '+name)


def _commit(run,path,binding,artifacts,extra=None):
    result={'binding':binding,'artifacts':{str(Path(p).relative_to(run)):file_sha(p) for p in artifacts},**(extra or {})}
    atomic(path,result);return result


def _binding(record):return sha(canonical({'input':record,'protocol':spec()}))


def fit(run,execute_analysis=False):
    if not execute_analysis:raise ValueError('Fit requires --execute-analysis; no fitting was done')
    run=Path(run);record,_=_receipt(run);_check_review(run,record);binding=_binding(record)
    if any(c['status']!='prepared' for c in record['cells']):raise ValueError('Input panel incomplete; no cells silently omitted')
    if (run/'extraction').exists() and not (run/'fits_manifest.json').exists():raise ValueError('New data already exist without frozen fits')
    summaries=[]
    for c in record['cells']:
        folder=run/'fits'/c['cell'];receipt=folder/'complete.json'
        if receipt.exists():
            saved=read_json(receipt)
            if saved['binding']!=binding:raise ValueError('Fit identity changed')
            _verify_artifacts(run,saved);summaries.append(saved);continue
        folder.mkdir(parents=True,exist_ok=True)
        with np.load(run/c['paths']['development.npz'],allow_pickle=False) as f:
            params,metadata=fit_predictors(f['train'],f['validation'],c['domain'],record['protocol']['lambda_grid'])
        save_npz(folder/'parameters.npz',**params);atomic(folder/'fit.json',metadata)
        saved=_commit(run,receipt,binding,[folder/'parameters.npz',folder/'fit.json'],{'cell':c['cell']})
        summaries.append(saved);print('FIT_COMPLETE',c['cell'],flush=True)
    atomic(run/'fits_manifest.json',{'binding':binding,'cells':summaries,'new_test_outputs_seen':False,
                                    'review_sha256':file_sha(run/'human_review.json')})
    return summaries


def _fits(run,record):
    frozen=read_json(Path(run)/'fits_manifest.json')
    if frozen['binding']!=_binding(record) or frozen['review_sha256']!=file_sha(Path(run)/'human_review.json'):
        raise ValueError('Frozen fit contract changed')
    if {x['cell'] for x in frozen['cells']}!={x['cell'] for x in record['cells']}:raise ValueError('Frozen fit coverage incomplete')
    for x in frozen['cells']:_verify_artifacts(run,x)
    return frozen


def extract(run,execute_models=False,allow_download=False):
    if not execute_models:raise ValueError('Extraction requires --execute-models; no model was loaded')
    run=Path(run);record,bank=_receipt(run);_check_review(run,record);frozen=_fits(run,record)
    # Environment must be fixed before importing torch through the model runtime.
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER','0')
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    from bsg_next.hf import ModelRuntime
    from bsg_next.extraction import extract_features,read_cache
    binding=sha(canonical({'inputs':_binding(record),'fits':frozen}))
    begun=run/'extraction_contract.json'
    if begun.exists():
        if read_json(begun)['binding']!=binding:raise ValueError('Extraction/frozen predictor contract changed')
    else:atomic(begun,{'binding':binding,'bank_sha256':record['bank_sha256'],
                      'fit_manifest_sha256':file_sha(run/'fits_manifest.json')})
    for model in input_adapter.protocol()['models']:
        selected=[c for c in record['cells'] if c['model_id']==model['id']]
        old=read_json(run/selected[0]['paths']['old_runtime.json'])
        if any(read_json(run/c['paths']['old_runtime.json'])!=old for c in selected):raise ValueError('Old runtime differs across domains/layers')
        runtime=ModelRuntime(model['id'],revision=model['revision'],dtype=old['dtype'].replace('torch.',''),
                             device_map=old['device_map'],batch_size=old['batch_size'],
                             local_files_only=not allow_download,chat_template=old.get('chat_template',False),
                             allow_cpu_offload=old.get('allow_cpu_offload',False),
                             deterministic=old.get('deterministic',True),attention_implementation=old.get('attention_implementation','eager'))
        current=runtime.identity()
        if current!=old:
            mismatch={k:{'old':old.get(k),'new':current.get(k)} for k in set(old)|set(current) if old.get(k)!=current.get(k)}
            atomic(run/'environment_mismatch.json',mismatch)
            raise ValueError('Runtime differs from parent cache; inspect environment_mismatch.json and install pinned dependencies')
        cache=run/'extraction'/input_adapter.model_key(model['id'])
        try:
            print('EXTRACT_OR_RESUME',model['id'],len(bank),'rows',flush=True)
            extract_features(runtime,bank,model['layers'],'span_mean',cache,resume=True)
            rows,arrays,manifest=read_cache(cache)
            if rows!=bank:raise ValueError('Extraction rows differ from frozen bank')
            for c in selected:
                folder=run/'new_coordinates'/c['cell'];receipt=folder/'complete.json'
                if receipt.exists():
                    saved=read_json(receipt)
                    if saved['binding']!=binding:raise ValueError('Projected coordinates belong to another run')
                    _verify_artifacts(run,saved);continue
                indices=[i for i,r in enumerate(rows) if r['domain']==c['domain']]
                with np.load(run/c['paths']['pca.npz'],allow_pickle=False) as pca:
                    z=(np.asarray(arrays[c['layer']][indices],dtype=np.float64)-pca['center'])@pca['basis']
                if not np.isfinite(z).all():raise ValueError('Nonfinite projected test coordinates')
                folder.mkdir(parents=True,exist_ok=True);save_npz(folder/'coordinates.npz',z=z)
                atomic(folder/'rows.json',{'rows':[rows[i] for i in indices]})
                _commit(run,receipt,binding,[folder/'coordinates.npz',folder/'rows.json'],
                        {'cell':c['cell'],'cache_manifest_sha256':file_sha(cache/'manifest.json'),
                         'saved_pca_sha256':file_sha(run/c['paths']['pca.npz'])})
                print('PROJECT_COMPLETE',c['cell'],flush=True)
        finally:runtime.close()
    atomic(run/'extraction_complete.json',{'binding':binding,'planned_cells':48,'status':'complete'})
    return {'status':'complete'}


def analyze(run,execute_analysis=False):
    if not execute_analysis:raise ValueError('Analysis requires --execute-analysis; no scientific endpoint computed')
    run=Path(run);record,bank=_receipt(run);_check_review(run,record);frozen=_fits(run,record)
    binding=sha(canonical({'inputs':_binding(record),'fits':frozen}))
    families=[f'new_context_{i:03d}' for i in range(64)];cells=[]
    if (run/'analysis_complete.json').exists():
        prior=read_json(run/'analysis_complete.json')
        if prior['binding']!=binding:raise ValueError('Analysis contract changed')
        _verify_artifacts(run,prior)
        for c in record['cells']:
            receipt=read_json(run/'new_coordinates'/c['cell']/'complete.json')
            if receipt['binding']!=binding:raise ValueError('Completed analysis test coordinate binding differs')
            _verify_artifacts(run,receipt)
        return read_json(run/'summary.json')
    for c in record['cells']:
        folder=run/'evaluations'/c['cell'];folder.mkdir(parents=True,exist_ok=True)
        try:
            receipt=read_json(run/'new_coordinates'/c['cell']/'complete.json')
            if receipt['binding']!=binding:raise ValueError('Test coordinate binding differs')
            _verify_artifacts(run,receipt)
            with np.load(run/'new_coordinates'/c['cell']/'coordinates.npz',allow_pickle=False) as f:z=f['z']
            rows=read_json(run/'new_coordinates'/c['cell']/'rows.json')['rows']
            if rows!=[r for r in bank if r['domain']==c['domain']]:raise ValueError('New test identities changed')
            test,names=blocks(z,rows,c['domain'],'test',families)
            with np.load(run/'fits'/c['cell']/'parameters.npz',allow_pickle=False) as f:params={k:f[k] for k in f.files}
            metadata=read_json(run/'fits'/c['cell']/'fit.json')
            result=evaluate(params,metadata,test,c['domain'],names)
        except (OSError,ValueError,KeyError,TypeError) as e:
            result={'status':'not_evaluable','reason':str(e)}
        result.update(cell=c['cell'],domain=c['domain'],model_id=c['model_id'],layer=c['layer'])
        atomic(folder/'result.json',result);cells.append(result)
        print('EVALUATE',c['cell'],result['status'],flush=True)
    cfg=record['protocol']['bootstrap']
    summary=aggregate(cells,families,draws=cfg['draws'],seed=cfg['seed'])
    summary.update(bank_sha256=record['bank_sha256'],human_review_sha256=file_sha(run/'human_review.json'),
                   model_execution_required=True,automatic_paper_promotion=False)
    atomic(run/'summary.json',summary)
    csvpath=run/'cell_summary.csv'
    with csvpath.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['cell','domain','status','identity','coordinate','difference','coordinate_nativeval','reason']);writer.writeheader()
        for c in cells:writer.writerow({k:c.get(k,'') for k in ('cell','domain','status','reason')}|c.get('means',{}))
    artifacts=[run/'summary.json',csvpath]+list((run/'evaluations').glob('*/result.json'))
    if summary['status']=='complete':
        _commit(run,run/'analysis_complete.json',binding,artifacts,{'status':summary['status']})
    else:
        # An early inspection must not permanently freeze a missing-cell panel.
        # Retain the entire failed attempt before allowing a later explicit retry.
        attempt=run/'analysis_attempts'/str(time.time_ns())
        for artifact in artifacts:
            archived=attempt/artifact.relative_to(run)
            archived.parent.mkdir(parents=True,exist_ok=True)
            archived.write_bytes(artifact.read_bytes())
        _commit(run,attempt/'receipt.json',binding,
                [p for p in attempt.rglob('*') if p.is_file()],{'status':summary['status']})
    atomic(run/'state.json',{'stage':'analyzed','status':summary['status'],'paper_results_available':summary['status']=='complete',
                            'automatic_manuscript_promotion':False})
    return summary


def _export_validation(run,record):
    """Report damaged/incomplete receipts without preventing preservation exports."""
    run=Path(run);fit_binding=_binding(record)
    checks={'scope':'artifact hashes and frozen bindings only; not scientific validity',
            'scientific_pass':False,'fit_receipts':{},'new_coordinate_receipts':{}}
    def inspect(path,binding):
        if not path.is_file():return {'status':'missing'}
        try:
            receipt=read_json(path)
            if not isinstance(receipt.get('artifacts'),dict) or not receipt['artifacts']:
                raise ValueError('Receipt has no artifact hashes')
            _verify_artifacts(run,receipt)
            if binding is None:raise ValueError('Frozen fits are unverified; cannot verify downstream binding')
            if receipt.get('binding')!=binding:raise ValueError('Receipt binding differs')
            return {'status':'verified','receipt_sha256':file_sha(path)}
        except (OSError,ValueError,KeyError,TypeError,AttributeError) as e:
            return {'status':'invalid','reason':str(e)}
    cells=record.get('cells',[])
    for c in cells:
        checks['fit_receipts'][c['cell']]=inspect(run/'fits'/c['cell']/'complete.json',fit_binding)
    frozen=None
    if not (run/'fits_manifest.json').is_file():checks['fits_manifest']={'status':'missing'}
    else:
        try:
            candidate=_fits(run,record)
            if len(candidate['cells'])!=len(cells):raise ValueError('Frozen fit count differs')
            for item in candidate['cells']:
                if checks['fit_receipts'][item['cell']]['status']!='verified':
                    raise ValueError('A committed fit receipt is invalid or missing')
                if read_json(run/'fits'/item['cell']/'complete.json')!=item:
                    raise ValueError('Fit manifest disagrees with committed receipt')
            frozen=candidate
            checks['fits_manifest']={'status':'verified','sha256':file_sha(run/'fits_manifest.json')}
        except (OSError,ValueError,KeyError,TypeError,AttributeError) as e:
            checks['fits_manifest']={'status':'invalid','reason':str(e)}
    binding=sha(canonical({'inputs':fit_binding,'fits':frozen})) if frozen is not None else None
    for c in cells:
        checks['new_coordinate_receipts'][c['cell']]=inspect(run/'new_coordinates'/c['cell']/'complete.json',binding)
    checks['analysis_receipt']=inspect(run/'analysis_complete.json',binding)
    if checks['analysis_receipt']['status']=='verified':
        try:
            if read_json(run/'analysis_complete.json').get('status')!='complete':
                raise ValueError('Analysis receipt does not declare complete')
            if read_json(run/'summary.json').get('status')!='complete':
                raise ValueError('Analysis summary does not declare complete')
            if not cells or any(x['status']!='verified' for x in checks['new_coordinate_receipts'].values()):
                raise ValueError('Completed analysis has missing or invalid test-coordinate receipts')
        except (OSError,ValueError,KeyError,TypeError,AttributeError) as e:
            checks['analysis_receipt']={'status':'invalid','reason':str(e)}
    return checks


def collect(run,output):
    run=Path(run);record,_=_receipt(run);output=Path(output)
    if output.exists() or output.with_suffix(output.suffix+'.sha256').exists():raise ValueError('Refusing to overwrite result archive')
    # Audit metadata remains readable without retaining raw hidden arrays or old train inputs.
    validation=_export_validation(run,record)
    names=['input_manifest.json','input_manifest.sha256','source_archive_audit.json','protocol.json','bank.json',
           'human_review.json','fits_manifest.json','extraction_contract.json','extraction_complete.json',
           'summary.json','cell_summary.csv','analysis_complete.json','state.json','BANK_REVIEW.txt','environment_mismatch.json']
    files=[run/n for n in names if (run/n).is_file()]
    for folder in ('fits','new_coordinates','evaluations','failures','analysis_attempts'):
        files.extend(p for p in (run/folder).rglob('*') if p.is_file() and '.partial' not in p.name)
    files.extend(p for pattern in ('inputs/*/old_runtime.json','extraction/*/manifest.json')
                 for p in run.glob(pattern) if p.is_file())
    output.parent.mkdir(parents=True,exist_ok=True)
    tmp=output.with_suffix(output.suffix+'.partial')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(files):
            if p.is_symlink():raise ValueError('Output symlink not allowed')
            z.write(p,str(p.relative_to(run)))
        z.writestr('export_manifest.json',json.dumps({
            'files':{str(p.relative_to(run)):file_sha(p) for p in files},
            'status':'exported_preserved_records_not_a_scientific_pass',
            'artifact_validation':validation,
            'extraction_provenance':'manifest metadata only; referenced raw feature chunks are excluded',
            'raw_feature_cache_complete':False,'raw_feature_arrays_included':False,
            'old_training_coordinates_included':False},indent=2,allow_nan=False))
    os.replace(tmp,output)
    output.with_suffix(output.suffix+'.sha256').write_text(file_sha(output)+'  '+output.name+'\n')
    return {'archive':str(output),'sha256':file_sha(output),'private_coordinates_included':True,
            'raw_hidden_arrays_included':False,'old_training_coordinates_included':False}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);subs=p.add_subparsers(dest='command',required=True)
    q=subs.add_parser('prepare');q.add_argument('--archive',type=Path,required=True);q.add_argument('--expected-sha256',required=True);q.add_argument('--output',type=Path,required=True)
    for name in ('review','fit','extract','analyze','collect'):
        q=subs.add_parser(name);q.add_argument('--run',type=Path,required=True)
        if name=='review':q.add_argument('--bank-sha256',required=True);q.add_argument('--attest-reviewed',action='store_true')
        if name in ('fit','analyze'):q.add_argument('--execute-analysis',action='store_true')
        if name=='extract':q.add_argument('--execute-models',action='store_true');q.add_argument('--allow-download',action='store_true')
        if name=='collect':q.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv)
    try:
        if a.command=='prepare':result=prepare(a.archive,a.expected_sha256,a.output)
        else:
            if not a.run.is_dir():raise ValueError('Prepared run does not exist')
            with lock(a.run):
                if a.command=='review':result=review(a.run,a.bank_sha256,a.attest_reviewed)
                elif a.command=='fit':result=fit(a.run,a.execute_analysis)
                elif a.command=='extract':result=extract(a.run,a.execute_models,a.allow_download)
                elif a.command=='analyze':result=analyze(a.run,a.execute_analysis)
                else:result=collect(a.run,a.output)
        print(json.dumps({'command':a.command,'status':'finished','details':result if a.command in ('analyze','collect') else None},indent=2,allow_nan=False))
        return 0
    except (ValueError,OSError,RuntimeError,KeyError) as e:
        print(type(e).__name__+': '+str(e),file=sys.stderr)
        failure_root=getattr(a,'run',None) or (a.output if a.command=='prepare' else None)
        if failure_root and failure_root.is_dir():
            failure=failure_root/'failures'/f'{time.time_ns()}_{a.command}.json'
            atomic(failure,{'command':a.command,'exception_type':type(e).__name__,'message':str(e),'status':'stopped_preserved'})
        return 2

if __name__=='__main__':raise SystemExit(main())
