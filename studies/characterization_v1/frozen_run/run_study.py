#!/usr/bin/env python3
"""Independent, limited-precision characterization; explicit execution only."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import threading
import time
import zipfile

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')


class BudgetExceeded(RuntimeError):
    pass


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path,value):
    from bsg_next.io import jsonable
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.partial')
    with temp.open('w') as f:
        json.dump(jsonable(value),f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    temp.replace(path)


def validate_bundle():
    manifest=read(ROOT/'source_manifest.json')
    for name,expected in manifest.items():
        p=Path(name)
        if p.is_absolute() or '..' in p.parts or (ROOT/p).is_symlink():raise ValueError('Unsafe source manifest')
        if digest(ROOT/p)!=expected:raise ValueError('Frozen source/config changed: '+name)
    return digest(ROOT/'source_manifest.json')


def validate_parent(parent,cfg):
    for name,expected in cfg['parent_pins'].items():
        if digest(parent/name)!=expected:raise ValueError('Parent evidence changed: '+name)
    for domain,seal in cfg['dataset_seals'].items():
        if digest(parent/'outputs/datasets'/domain/'manifest.json')!=seal['manifest_sha256']:
            raise ValueError('Dataset seal changed: '+domain)
    for stage in ('extract_confirmation','structural_confirmation'):
        if (parent/'stage_state'/(stage+'.json')).exists():
            raise ValueError('Parent records prior test work; review independence before a new study')
        folder=parent/'outputs'/stage
        if folder.exists() and any(folder.iterdir()):
            raise ValueError('Parent contains post-pilot artifacts; review access before proceeding')
    if (parent/'confirmation_freeze.json').exists():raise ValueError('Parent already has a confirmation freeze')
    from bsg_next.estimators import numeric_code_identity
    if numeric_code_identity()!=cfg['expected_numeric_code_identity']:raise ValueError('Numerical source mismatch')
    old=read(parent/'study_config.json')
    if cfg['models']!=old['models']:raise ValueError('Pinned model panel changed')
    from importlib.metadata import version
    for package,expected in old['protocol_dependency_versions'].items():
        if package=='python':continue
        if version(package).split('+',1)[0]!=expected.split('+',1)[0]:
            raise ValueError('Dependency base version mismatch: '+package)
    from bsg_next.config import model_key
    oldman={}
    for model in cfg['models']:
        for domain in ('C7','D4','S3'):
            p=parent/'outputs/extract_development'/model_key(model)/domain
            if digest(p/'manifest.json')!=cfg['expected_development_cache_manifest_hashes'][model['id']+'|'+domain]:
                raise ValueError('Development-cache identity changed')
            oldman[model['id']+'|'+domain]=read(p/'manifest.json')
            environments=[v['dependency_build_versions'] for v in oldman[model['id']+'|'+domain].get('telemetry',[])
                          if v.get('stage')=='runtime_environment' and 'dependency_build_versions' in v]
            for environment in environments:
                for package,expected_build in environment.items():
                    if expected_build is not None and version(package)!=expected_build:
                        raise ValueError('Dependency build differs from original extraction: '+package)
    return oldman


def selected_test_rows(rows,domain,cfg):
    from bsg_next.data import select_phase_records
    selected=select_phase_records(rows,'confirmation',cfg['test_context_families'],cfg['test_entity_orbits'])
    if domain=='S3':selected=[r for r in selected if r['split'] in cfg['included_S3_splits']]
    if len(selected)!=cfg['all_test_rows_per_model'][domain]:raise ValueError('Unexpected held-out/support block size')
    if domain=='S3' and Counter(r['split'] for r in selected)!=cfg['S3_new_row_breakdown']:
        raise ValueError('Unexpected S3 block allocation')
    if domain!='S3' and {r['split'] for r in selected}!={'test'}:raise ValueError('Unexpected structural split')
    return selected


def disk_estimate(cfg,oldman):
    total=0
    for spec in cfg['models']:
        for domain in ('C7','D4','S3'):
            m=oldman[spec['id']+'|'+domain]
            dimensions=sum(v['shape'][1] for v in m['arrays'].values())
            if domain=='S3':dimensions+=m['potential_features']['shape'][1]
            new=cfg['all_test_rows_per_model'][domain]
            total+=(2*new+m['n_rows'])*dimensions*4  # New arrays plus a separate merged cache.
    return int(total*1.05)+2**30  # Row metadata, projected exports and filesystem allowance.


def merge_resumable(old,new,merged):
    """Copy verified source rows through the durable cache writer, without a model."""
    import numpy as np
    from bsg_next.extraction import read_cache,extract_features
    sources=[read_cache(p) for p in (old,new)]
    first=sources[0][2]
    for _,_,m in sources:
        if any(m[k]!=first[k] for k in ('model','layers','readout','extraction_code_sha256')):
            raise ValueError('Incompatible merge sources')
    rows=[r for rr,_,_ in sources for r in rr]
    lookup={r['row_id']:(s,i) for s,(rr,_,_) in enumerate(sources) for i,r in enumerate(rr)}
    if len(lookup)!=len(rows):raise ValueError('Duplicate source rows')
    merged.mkdir(parents=True,exist_ok=True)
    descriptor={'source_manifest_sha256':[digest(p/'manifest.json') for p in (old,new)],
                'transform':'exact_source_row_concatenation_without_model_execution'}
    # Keep merge provenance outside the cache folder so its writer can initialize safely.
    prov=merged.parent/(merged.name+'.merge_sources.json')
    if prov.exists() and read(prov)!=descriptor:raise ValueError('Merge source identities changed')
    atomic(prov,descriptor)
    potentials=[np.load(p/m['potential_features']['path'],mmap_mode='r',allow_pickle=False)
                if m.get('potential_features') else None for p,(_,_,m) in zip((old,new),sources)]
    class CopyRuntime:
        telemetry=[]
        def identity(self):return first['model']
        def features(self,records,layers,readout):
            for start in range(0,len(records),256):
                batch=records[start:start+256]
                yield list(range(start,start+len(batch))),{layer:np.stack([
                    sources[lookup[r['row_id']][0]][1][layer][lookup[r['row_id']][1]] for r in batch]) for layer in layers}
        def input_entity_features(self,records):
            return np.stack([potentials[lookup[r['row_id']][0]][lookup[r['row_id']][1]] for r in records])
    return extract_features(CopyRuntime(),rows,first['layers'],first['readout'],merged,
                            potential_features=bool(first.get('potential_features')))


def collect_results(output):
    atomic(output/'collection_status.json',{'summary_status':read(output/'summary.json').get('status') if (output/'summary.json').exists() else 'incomplete_no_summary',
        'artifact_collection_is_not_a_scientific_pass':True,'models_loaded_by_collection':False})
    archive=output.with_suffix('.results.zip');temp=archive.with_suffix('.zip.partial')
    if temp.exists():temp.unlink()  # Regenerable export only, never experiment files.
    print('EXPORT_RESULTS_START; compact coordinates included, full hidden caches excluded.',flush=True)
    count=0
    with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for folder in ('analysis','provenance'):
            base=output/folder
            if base.exists():
                for p in sorted(base.rglob('*')):
                    if p.is_file():
                        z.write(p,output.name+'/'+str(p.relative_to(output)));count+=1
                        if count%25==0:print('EXPORT_PROGRESS files='+str(count),flush=True)
        for name in ('study_contract.json','summary.json','execution_state.json','budget.json','collection_status.json'):
            if (output/name).exists():z.write(output/name,output.name+'/'+name)
    with zipfile.ZipFile(temp) as z:
        if z.testzip() is not None:raise ValueError('Result export CRC failed')
    temp.replace(archive)
    archive.with_suffix('.zip.sha256').write_text(digest(archive)+'  '+archive.name+'\n')
    print('RESULT_ARCHIVE '+str(archive),flush=True)


def run(args):
    import numpy as np
    from bsg_next.config import model_key
    from bsg_next.data import load_sealed_dataset
    from bsg_next.extraction import read_cache,extract_features
    from bsg_next.hf import runtime_from_config
    from bsg_next.structural import fit_train_pca,analyze_cell
    from bsg_next.groups import get_group
    from fields import compare_cell,input_comparisons
    bundle=validate_bundle();cfg=read(ROOT/'study_config.json')
    parent=args.parent.resolve();output=args.output.resolve()
    if output.is_relative_to(parent) or output.is_relative_to(ROOT):raise ValueError('Use a new output outside source and parent')
    output.mkdir(parents=True,exist_ok=True)
    with (output/'worker.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('Another worker is already active; do not start a second copy')
        contract={'study':cfg,'bundle_source_manifest_sha256':bundle,
                  'parent_input_hashes':cfg['parent_pins'],'technical_test':False,
                  'old_study_continued':False,'formal_inference_enabled':False,
                  'holdout_access_assurance':'bounded recorded evidence; not a universal external access certification'}
        frozen=output/'study_contract.json'
        if frozen.exists():
            if read(frozen)!=contract:raise ValueError('Output contract mismatch; preserve it and use a fresh namespace')
        else:
            if any(p.name!='worker.lock' for p in output.iterdir()):raise ValueError('Nonempty output without a frozen contract')
            atomic(frozen,contract)  # Frozen before dataset or activation outcomes are opened.
        oldman=validate_parent(parent,cfg)
        needed=disk_estimate(cfg,oldman)
        reserve=int(cfg['minimum_free_disk_reserve_GiB']*2**30)
        # Reserve worst-case allocation on a first run; retain partial files on resume.
        existing=sum(p.stat().st_size for folder in ('heldout','merged') for p in (output/folder).rglob('*') if p.is_file())
        if shutil.disk_usage(output).free<max(0,needed-existing)+reserve:
            raise ValueError('Insufficient free disk for raw/merged caches and reserve; no extraction started')
        previous=read(output/'budget.json') if (output/'budget.json').exists() else {'spent_seconds':0,'running':False}
        spent=float(previous['spent_seconds'])
        if previous.get('running'):spent+=max(0,time.time()-previous['updated_epoch'])
        remaining=cfg['wall_budget_seconds_total']-spent
        if remaining<=0:raise BudgetExceeded('Cumulative six-hour budget exhausted; outputs retained')
        started=time.monotonic();state=read(output/'execution_state.json') if (output/'execution_state.json').exists() else {'completed':[]}
        def budget(running=True):
            atomic(output/'budget.json',{'spent_seconds':spent+time.monotonic()-started,'updated_epoch':time.time(),
                    'running':running,'limit_seconds':cfg['wall_budget_seconds_total']})
        def stopped(signum,frame):raise BudgetExceeded('Budget or termination signal received; resume requires the same frozen package')
        for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM):signal.signal(sig,stopped)
        signal.setitimer(signal.ITIMER_REAL,remaining)
        budget();status='running';error=None
        heartbeat_stop=threading.Event()
        def heartbeat():
            while not heartbeat_stop.wait(60):
                print('HEARTBEAT elapsed_this_session_seconds='+str(round(time.monotonic()-started))+
                      ' completed_checkpoints='+str(len(state['completed'])),flush=True)
        monitor=threading.Thread(target=heartbeat,daemon=True);monitor.start()
        try:
            provenance=output/'provenance';provenance.mkdir(exist_ok=True)
            for name in ('source_manifest.json','study_config.json'):
                shutil.copyfile(ROOT/name,provenance/name)
            from importlib.metadata import version
            atomic(provenance/'dependency_build_versions.json',{name:version(name) for name in read(parent/'study_config.json')['protocol_dependency_versions'] if name!='python'})
            atomic(provenance/'disk_plan.json',{'estimated_additional_bytes_first_run':needed,'available_bytes':shutil.disk_usage(output).free,
                                              'reserve_bytes':reserve,'not_a_runtime_quote':True})
            # Loading sealed prompts happens only after the amended source/config contract exists.
            datasets={domain:selected_test_rows(load_sealed_dataset(parent/'outputs/datasets'/domain),domain,cfg)
                      for domain in ('C7','D4','S3')}
            def record(label):
                if label not in state['completed']:state['completed'].append(label)
                atomic(output/'execution_state.json',state);budget()
            for spec in cfg['models']:
                key=model_key(spec)
                runtime=runtime_from_config(spec,cfg['runtime'])
                try:
                    for domain in ('C7','D4','S3'):
                        label=key+'/'+domain
                        print('EXTRACT_OR_VERIFY '+label+' rows='+str(len(datasets[domain])),flush=True)
                        if runtime.identity()!=oldman[spec['id']+'|'+domain]['model']:
                            raise ValueError('Runtime identity differs from the pinned development cache')
                        old=parent/'outputs/extract_development'/key/domain
                        new=output/'heldout'/key/domain
                        extract_features(runtime,datasets[domain],spec['layers'],
                            'final_token' if domain=='S3' else 'span_mean',new,potential_features=domain=='S3')
                        record('extract/'+label)
                        merged=output/'merged'/key/domain
                        print('MERGE_OR_VERIFY '+label,flush=True)
                        merge_resumable(old,new,merged)
                        rows,arrays,manifest=read_cache(merged)
                        for role,cachepath in (('development',old),('heldout',new),('merged',merged)):
                            shutil.copyfile(cachepath/'manifest.json',provenance/(key+'__'+domain+'__'+role+'.json'))
                        # The merge is required to contain exactly the same source rows, once each.
                        old_rows,_,_=read_cache(old)
                        if {r['row_id'] for r in rows}!={r['row_id'] for r in old_rows+datasets[domain]}:
                            raise ValueError('Merged row coverage differs from the frozen source blocks')
                        record('merge/'+label)
                        potential=None
                        shared=None
                        if domain=='S3':
                            potential=np.load(merged/manifest['potential_features']['path'],mmap_mode='r',allow_pickle=False)
                            shared=output/'analysis/shared'/(key+'__S3');shared.mkdir(parents=True,exist_ok=True)
                            compact=shared/'compact_export.json'
                            if compact.exists():
                                for name,sha in read(compact).items():
                                    if digest(shared/name)!=sha:raise ValueError('Compact S3 input export changed')
                            else:
                                keys=sorted({(r['orbit_id'],r['state_id']) for r in rows});first={}
                                for i,r in enumerate(rows):first.setdefault((r['orbit_id'],r['state_id']),i)
                                positions={k:i for i,k in enumerate(keys)}
                                inverse=np.array([positions[(r['orbit_id'],r['state_id'])] for r in rows],dtype=np.int32)
                                first_ids=np.array([first[k] for k in keys],dtype=np.int64)
                                np.savez_compressed(shared/'potential_unique.npz',values=np.asarray(potential[first_ids]),inverse=inverse,first_row_indices=first_ids)
                                atomic(shared/'rows.json',{'rows':[{k:r[k] for k in ('row_id','family_id','split','state_id','paraphrase_id','orbit_id','domain')} for r in rows],
                                    'potential_keys':keys,'scope':'Exact first unique orbit/state vectors used by unchanged legacy potential preprocessing; not all full hidden states.'})
                                atomic(compact,{name:digest(shared/name) for name in ('potential_unique.npz','rows.json')})
                        train=np.array([r['split']=='train' for r in rows])
                        group=get_group(domain)
                        splits=('test_joint','test_context','test_entity') if domain=='S3' else ('test',)
                        for layer in spec['layers']:
                            cell=key+'__'+domain+'__'+str(layer)
                            target=output/'analysis'/cell/'legacy.json'
                            extras=output/'analysis'/cell/'fields.json'
                            marker=target.parent/'cell_complete.json'
                            if marker.exists():
                                for name,sha in read(marker).items():
                                    if digest(target.parent/name)!=sha:raise ValueError('Completed cell changed: '+cell)
                                print('RESUME_CELL '+cell,flush=True);continue
                            print('ANALYZE_CELL '+cell,flush=True)
                            center,basis,pca=fit_train_pca(arrays[layer],train,16)
                            z=np.empty((len(rows),16),dtype=np.float64)
                            for start in range(0,len(rows),512):
                                z[start:start+512]=(np.asarray(arrays[layer][start:start+512],float)-center)@basis
                            target.parent.mkdir(parents=True,exist_ok=True)
                            # Fitted coordinate transform retained independently of the source cache.
                            np.savez_compressed(target.parent/'pca.npz',center=center,basis=basis)
                            if not target.exists():
                                values=[]
                                for split in splits:
                                    print('LEGACY_ENDPOINTS '+cell+' '+split,flush=True)
                                    v=analyze_cell(z,rows,group,cfg['structural'],potential,split)
                                    v.update(model=manifest['model'],layer=layer,pca=pca,cache_manifest_sha256=digest(merged/'manifest.json'))
                                    values.append(v)
                                atomic(target,values)
                            if domain!='S3' and not extras.exists():
                                generator=compare_cell(z,rows,group,cfg['structural'],evaluation_split='test')
                                comparisons=input_comparisons(z,rows,group,cfg['structural'],evaluation_split='test')
                                import math
                                legacy=read(target)[0]['mean_losses']
                                for current,expected in (
                                    (generator['mean_losses']['unseen_composite']['generator_ridge'],legacy['generator_unseen_actual_target']),
                                    (comparisons['controls']['primary_operator']['mean_loss'],legacy['primary_operator']),
                                    (comparisons['controls']['oracle_local']['mean_loss'],legacy['local_displacement'])):
                                    if not math.isclose(current,expected,rel_tol=1e-8,abs_tol=1e-10):
                                        raise ValueError('New input comparison does not reproduce unchanged legacy estimator')
                                full=next(v for v in comparisons['storage_curves'] if v['track']=='generator_only' and v['full_training_support'])
                                for kind in ('primitive','unseen_composite'):
                                    if not math.isclose(full['results'][kind]['mean_loss'],generator['mean_losses'][kind]['generator_input_only_field'],rel_tol=1e-10,abs_tol=1e-12):
                                        raise ValueError('Full kernel curve and generator reference disagree')
                                atomic(extras,{'generator':generator,'input_comparisons':comparisons,
                                    'model_id':spec['id'],'group':group.name,'layer':layer,
                                    'technical_test':False,'formal_inference':False,'paper_eligible':False})
                            if domain!='S3':
                                np.savez_compressed(target.parent/'coordinates.npz',z=z)
                                atomic(target.parent/'rows.json',{'rows':[{k:r[k] for k in ('row_id','family_id','split','state_id','paraphrase_id','orbit_id','domain')} for r in rows]})
                            else:
                                np.savez_compressed(target.parent/'coordinates.npz',z=z)
                                atomic(target.parent/'coordinate_export.json',{'rows_and_potential_archive_relative_directory':str(shared.relative_to(output)),
                                    'shared_input_hashes':read(shared/'compact_export.json'),
                                    'reconstruct_potential_features':'potential_unique.values[potential_unique.inverse[row_indices]]; no model forward required',
                                    'full_hidden_state_or_PCA_refit_from_raw_exported':False})
                            required=['legacy.json','pca.npz','coordinates.npz']+(['fields.json','rows.json'] if domain!='S3' else ['coordinate_export.json'])
                            atomic(marker,{name:digest(target.parent/name) for name in required})
                            record('analyze/'+cell)
                            print('DONE_CELL '+cell,flush=True)
                finally:runtime.close()
            status='complete_limited_precision_characterization'
        except Exception as exc:
            status='budget_stopped' if isinstance(exc,BudgetExceeded) else 'error'
            error={'type':type(exc).__name__,'message':str(exc)}
        finally:
            signal.setitimer(signal.ITIMER_REAL,0);heartbeat_stop.set();monitor.join(timeout=1);budget(False)
        legacy=[v for p in sorted((output/'analysis').glob('*/legacy.json')) for v in read(p)]
        fields=[read(p) for p in sorted((output/'analysis').glob('*/fields.json'))]
        if status.startswith('complete') and (len(legacy)!=120 or len(fields)!=48):
            status='error';error={'type':'IncompletePanel','message':'Expected 120 legacy split records and 48 field comparisons'}
        summary={'status':status,'error':error,'study_id':cfg['study_id'],'legacy_split_records':len(legacy),
            'field_cells':len(fields),'technical_test':False,'paper_eligible':False,
            'formal_inference_enabled':False,'old_power_gate_passed':False,'native_run':False,
            'synthetic_calibration_run':False,'scope':'Pilot-informed independent holdout characterization conditional on recorded access evidence; not old-study confirmation.'}
        grouped=defaultdict(list)
        for v in legacy:
            grouped[v['group']+'|'+v['test_split']].append(v)
        summary['descriptive_legacy_means']={key:{name:float(np.mean([r['mean_losses'][name] for r in values]))
            for name in values[0]['mean_losses']} for key,values in grouped.items()}
        summary['descriptive_field_means']={}
        for group in ('C7','D4_corner'):
            records=[r for r in fields if r['group']==group]
            if not records:continue
            summary['descriptive_field_means'][group]={
                'cells':len(records),'generator_unseen':{method:float(np.mean([
                    r['generator']['mean_losses']['unseen_composite'][method] for r in records]))
                    for method in ('generator_ridge','generator_input_only_field')},
                'all_element_full_kernel':float(np.mean([next(v['results']['all_changed']['mean_loss']
                    for v in r['input_comparisons']['storage_curves'] if v['track']=='all_element' and v['full_training_support']) for r in records]))}
        if status.startswith('complete'):
            from uncertainty import summarize
            try:
                sensitivity=summarize(legacy,fields,expected_conditions=24,
                    replicates=cfg['descriptive_uncertainty']['replicates'],seed=cfg['descriptive_uncertainty']['seed'])
                atomic(output/'analysis'/'descriptive_uncertainty.json',sensitivity)
                summary['descriptive_uncertainty']='analysis/descriptive_uncertainty.json; sensitivity only, no new significance decisions'
            except Exception as exc:
                status='error';summary['status']=status;summary['error']={'type':type(exc).__name__,'message':str(exc)}
        atomic(output/'summary.json',summary);collect_results(output)
        budget(False)
        print(json.dumps(summary,indent=2),flush=True)
        return 0 if status.startswith('complete') else 2


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--execute-models',action='store_true')
    mode.add_argument('--self-test',action='store_true')
    mode.add_argument('--collect-existing',action='store_true')
    parser.add_argument('--parent',type=Path,default=Path('/workspace/bsg-runs/icml2027-bf16-f64-v2'))
    parser.add_argument('--output',type=Path,default=Path('/workspace/bsg-runs/icml2027-characterization-v1'))
    args=parser.parse_args()
    if args.self_test:
        from test_study import run_tests
        run_tests();return 0
    if args.collect_existing:
        manifest=validate_bundle();output=args.output.resolve()
        if not (output/'study_contract.json').exists():raise ValueError('No existing study to collect')
        if read(output/'study_contract.json')['bundle_source_manifest_sha256']!=manifest:
            raise ValueError('Collection source contract mismatch')
        with (output/'worker.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            collect_results(output)
        return 0
    return run(args)


if __name__=='__main__':raise SystemExit(main())
