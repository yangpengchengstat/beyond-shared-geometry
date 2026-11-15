#!/usr/bin/env python3
"""Explicit synthetic-test and offline-execution entrypoints; create-only outputs."""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, io, json, os, platform, sys, time, traceback, unittest
from pathlib import Path
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0,str(HERE))
import core
import numpy as np

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def write(path, obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f: json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False); f.write('\n')

def read(path): return json.loads(Path(path).read_text())

def source_hashes():
    manifest=read(HERE/'source_manifest.json')
    for name,sha in manifest.items():
        if digest(PROJECT/name)!=sha: raise ValueError('Source changed since freeze: '+name)
    return manifest

def make_output(path):
    path=Path(path).resolve()
    if path.is_relative_to(PROJECT/'results') or path.is_relative_to(PROJECT/'paper'):
        raise ValueError('Use a fresh run directory outside frozen results and manuscript')
    path.mkdir(parents=True,exist_ok=False)
    return path

def test(output):
    hashes=source_hashes(); out=make_output(output)
    module=core.load('family_synthetic_tests',HERE/'test_core.py')
    suite=unittest.defaultTestLoader.loadTestsFromModule(module)
    with (out/'tests.log').open('x') as stream:
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    receipt={'status':'passed' if result.wasSuccessful() and result.testsRun else 'failed',
       'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
       'skipped':len(result.skipped),'python':sys.version,'numpy':np.__version__,
       'source_hashes':hashes,'real_banks_loaded':False,'real_family_deletions':0}
    write(out/'tests_result.json',receipt); print(json.dumps(receipt|{'source_hashes':'recorded'}),flush=True)
    return 0 if receipt['status']=='passed' and not result.skipped else 2

def same(checks, name,a,b):
    core.orthogonal.Checks(checks).scalar(name,float(a),float(b))

def anchor(result, saved, spec):
    if result['status']!='complete':return result
    try:
        for key in ('eta','L_orth','beta','extra_law_lower','extra_law_upper'):
            same(result['checks'],'saved_orth_'+key,result['metrics'][key],saved['metrics'][key])
        for key in ('gamma','L_lower','L_upper'):
            same(result['checks'],'saved_spectral_'+key,result['metrics'][key],spec['metrics'][key])
        same(result['checks'],'saved_spectral_eta',result['metrics']['eta'],spec['metrics']['eta_LS'])
        for key in ('sigma_min','sigma_max','sigma_min_max_ratio'):
            same(result['checks'],'saved_moment_'+key,result['diagnostics'][key],saved['diagnostics'][key])
        same(result['checks'],'saved_identity',result['metrics']['L_identity'],saved['metrics']['L_identity_direct'])
    except core.orthogonal.ComparatorError as e:
        result.update(status='baseline_anchor_failed',reason=str(e))
    return result

def export_csv(path, records):
    metrics=['eta','L_orth','beta','gamma','L_lower','L_upper','p_minus','p_plus',
        'extra_law_lower','extra_law_upper','extra_law_lower_above_tolerance',
        'extra_law_comparison_tolerance','extra_law_lower_tolerance_multiple',
        'L_identity','delta_type_normalized','total_trivial_energy']
    cols=['cell','model_id','layer','deleted_family','status','reason']+metrics+[
        'sigma_min','sigma_max','sigma_min_max_ratio','sigma_condition_number']
    with Path(path).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
        for record in records:
            row={**record,**record.get('metrics',{}),**record.get('diagnostics',{})}
            w.writerow({k:row.get(k) for k in cols})

def summaries(records, baselines, families):
    bycell={c:[] for c in baselines}
    for record in records: bycell[record['cell']].append(record)
    cells=[]
    tracked=['eta','L_orth','beta','gamma','L_lower','L_upper','p_minus','p_plus',
             'extra_law_lower','extra_law_upper','L_identity','delta_type_normalized']
    for cell, rr in bycell.items():
        good=[r for r in rr if r['status']=='complete']
        b=baselines[cell]['metrics']
        summary={'cell':cell,'planned':64,'completed':len(good),'failed':64-len(good),
          'positive_law_lower_count':sum(r['metrics']['extra_law_lower_above_tolerance'] for r in good),
          'baseline':b,'metrics':{}}
        for key in tracked:
            if good:
                lo=min(good,key=lambda r:r['metrics'][key]); hi=max(good,key=lambda r:r['metrics'][key])
                summary['metrics'][key]={'min':lo['metrics'][key],'min_family':lo['deleted_family'],
                   'max':hi['metrics'][key],'max_family':hi['deleted_family'],
                   'min_change':lo['metrics'][key]-b[key], 'max_change':hi['metrics'][key]-b[key]}
        summary['smallest_sigma_ratio']=min((r['diagnostics']['sigma_min_max_ratio'] for r in good),default=None)
        cells.append(summary)
    def panel(rows, family):
        good=[r for r in rows if r['status']=='complete']
        result={'deleted_family':family,'complete_cells':len(good),'planned_cells':24}
        if len(good)!=24: result['status']='incomplete_no_aggregate';return result
        result.update(status='complete',means={k:float(np.mean([r['metrics'][k] for r in good])) for k in tracked},
           positive_law_lower_count=sum(r['metrics']['extra_law_lower_above_tolerance'] for r in good))
        for key in ('extra_law_lower','p_minus'):
            values=np.array([r['metrics'][key] for r in good]); order=np.argsort(-values);total=float(values.sum())
            result[key]={'median':float(np.median(values)),'min':float(values.min()),'max':float(values.max()),
               'sum':total,'largest_cell':good[int(order[0])]['cell'],
               'largest_share':float(values[order[0]]/total) if total>0 else None,
               'top_two_share':float(values[order[:2]].sum()/total) if total>0 else None}
        return result
    return {'cell_summaries':cells,'baseline_panel':panel(list(baselines.values()),None),
       'family_panels':[panel([r for r in records if r['deleted_family']==f],f) for f in families],
       'interpretation':'Post-hoc complete-family deletion with each reduced bank rewhitened; not fixed-metric additive contribution, population inference, independent replication, or registration analysis.'}

def execute(repo, output, tests):
    hashes=source_hashes(); receipt=read(Path(tests)/'tests_result.json')
    if receipt['status']!='passed' or receipt['skipped'] or receipt['source_hashes']!=hashes or receipt['numpy']!=np.__version__ or receipt['python']!=sys.version:
        raise ValueError('Current-source tests in the same runtime must pass before real inputs')
    out=make_output(output); started=time.time(); protocol=read(HERE/'protocol.json')
    write(out/'protocol.json',protocol);write(out/'test_receipt.json',receipt)
    write(out/'source_hashes.json',hashes)
    write(out/'runtime.json',{'python':sys.version,'executable':sys.executable,'numpy':np.__version__,
       'platform':platform.platform(),'model_forward':False,'blas_threads_requested':1})
    old=core.load('old_orthogonal_run',PROJECT/'studies/orthogonal_comparator_v1/run.py')
    inputs=core.load('family_archive_inputs',PROJECT/'studies/spectral_dimension_sensitivity_v1/archive_inputs.py')
    _,_,historical=old.source_contract()
    orth_evidence=read(PROJECT/'results/orthogonal_comparator_v1/EVIDENCE_SHA256.json')
    families=historical['test_families']
    if families!=[f'structural_test_{i:03d}' for i in range(64)]:raise ValueError('Family panel differs')
    baselines={}; records=[]; banks={}; bindings={}
    try:
        with inputs.Study2Inputs(repo,historical) as reader:
            print('VERIFY_ORIGINAL_ARCHIVES_AND_INPUTS',flush=True)
            preflight=reader.preflight();write(out/'input_preflight.json',preflight)
            selected=sorted([c for c in reader.cells if c['domain']=='C7'],key=lambda c:c['cell'])
            if len(selected)!=24:raise ValueError('C7 panel incomplete')
            write(out/'plan.json',{'cells':selected,'families':families,'baseline_units':24,'deletion_units':1536})
            oldconfig=read(PROJECT/'analysis/rank_sensitivity_v1/config.json')
            snapshots={k:reader.followup.json(k,v) for k,v in oldconfig['metadata_hashes'].items()}
            snapshot_cells={c['cell']:c for c in snapshots['alternatives/input_snapshot/inputs.json']['cells']}
            snapmanifest=snapshots['alternatives/input_snapshot/prepared_manifest.json']['files']
            for c in selected:
                cell=c['cell']; Z=reader.load_test_bank(cell); contexts=c['contexts']
                if Z.shape!=(128,7,16) or Z.dtype!=np.float64 or set(core.validate_contexts(Z,contexts))!=set(families):raise ValueError('Production bank/family shape differs')
                snap=snapshot_cells[cell]; rowsrel=str(Path(snap['bank_path']).with_name('rows.json'))
                rows=reader.followup.json('alternatives/input_snapshot/'+rowsrel,snapmanifest[rowsrel])
                if rows['test_contexts']!=contexts or rows['test_indices']!=c['row_indices'] or snap['source_members']!=c['members']:raise ValueError('Snapshot/parent mapping mismatch')
                raw=reader.followup.read('alternatives/input_snapshot/'+snap['bank_path'],snapmanifest[snap['bank_path']])
                with np.load(io.BytesIO(raw),allow_pickle=False) as a:
                    if not np.array_equal(a['test'],Z):raise ValueError('Snapshot and original coordinates not identical')
                savedfile=PROJECT/'results/orthogonal_comparator_v1/units'/(cell+'__r16')/'summary.json'
                if digest(savedfile)!=orth_evidence[str(savedfile.relative_to(PROJECT/'results/orthogonal_comparator_v1'))]:
                    raise ValueError('Original orthogonal evidence hash mismatch')
                result=anchor(core.analyze_bank(Z),read(savedfile),reader.semantic_baselines[cell])
                binding={'original_cell':c,'snapshot_bank_sha256':snapmanifest[snap['bank_path']],
                    'snapshot_rows_sha256':snapmanifest[rowsrel], 'baseline_orthogonal_sha256':digest(savedfile),
                    'pca_unchanged':True,'same_snapshot_and_original_values':True}
                result.update(cell=cell,model_id=c['model_id'],layer=c['layer'],deleted_family=None)
                write(out/'baseline'/cell/'summary.json',result);write(out/'bindings'/(cell+'.json'),binding)
                baselines[cell]=result; banks[cell]=Z;bindings[cell]=binding
                print('BASELINE',cell,result['status'],flush=True)
            export_csv(out/'baseline.csv',list(baselines.values()))
            if any(b['status']!='complete' for b in baselines.values()):raise ValueError('Baseline reproduction failed; no family deletions started')
            print('ALL_24_BASELINES_VERIFIED',flush=True)
            for c in selected:
                cell=c['cell']; Z=banks[cell]
                before=hashlib.sha256(Z.tobytes()).hexdigest()
                for family in families:
                    try:
                        remaining,deletion=core.delete_family(Z,c['contexts'],family)
                        if remaining.shape!=(126,7,16):raise ValueError('Deletion shape differs')
                        result=core.analyze_bank(remaining);result['deletion']=deletion
                    except Exception as exc:
                        result={'status':'not_evaluable_exception','reason':repr(exc),'traceback':traceback.format_exc(),'metrics':{},'diagnostics':{}}
                    result.update(cell=cell,model_id=c['model_id'],layer=c['layer'],deleted_family=family,
                        input_binding_file='bindings/'+cell+'.json')
                    write(out/'units'/cell/(family+'.json'),result);records.append(result)
                    if len(records)%16==0: print('PROGRESS',len(records),1536,'last_status='+result['status'],f'elapsed={time.time()-started:.1f}s',flush=True)
                if hashlib.sha256(Z.tobytes()).hexdigest()!=before:raise ValueError('Input array mutated')
            export_csv(out/'deletions.csv',records)
            after={key:inputs.file_sha(getattr(reader,key).path) for key in ('parent','followup')}
            unchanged=all(after[k]==inputs.ARCHIVES[k][1] for k in after) and source_hashes()==hashes
            write(out/'input_preservation.json',{'unchanged':unchanged,'archive_sha256_after':after,
               'used_member_sha256':{'parent':reader.parent.used,'followup':reader.followup.used},'source_unchanged':source_hashes()==hashes})
            if not unchanged:
                raise ValueError('Input preservation failed; no aggregate summary published')
            write(out/'summary.json',summaries(records,baselines,families))
            status='complete' if len(records)==1536 and all(r['status']=='complete' for r in records) and unchanged else 'complete_with_failures'
            write(out/'execution.json',{'status':status,'planned':1536,'recorded':len(records),
              'complete':sum(r['status']=='complete' for r in records),'failed':sum(r['status']!='complete' for r in records),
              'elapsed_seconds':time.time()-started,'paper_modified':False,'original_results_modified':False})
            print('EXECUTION_FINISHED',status,flush=True)
            return 0 if status=='complete' else 3
    except BaseException as exc:
        write(out/'technical_failure.json',{'error':repr(exc),'traceback':traceback.format_exc(),
            'deletions_recorded':len(records),'baselines_recorded':len(baselines),'elapsed_seconds':time.time()-started})
        raise

def main():
    p=argparse.ArgumentParser(); p.add_argument('command',choices=['test','execute']);p.add_argument('--output',required=True)
    p.add_argument('--repo');p.add_argument('--tests');p.add_argument('--execute-offline',action='store_true');a=p.parse_args()
    if a.command=='test':return test(a.output)
    if not (a.execute_offline and a.repo and a.tests):p.error('Execution needs --execute-offline, --repo and --tests')
    return execute(a.repo,a.output,a.tests)

if __name__=='__main__':raise SystemExit(main())
