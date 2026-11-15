#!/usr/bin/env python3
"""Read a pinned coordinate ZIP and run new CPU-only exploratory comparisons."""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
import fcntl
import hashlib
import io
import json
import multiprocessing as mp
from pathlib import Path, PurePosixPath
import platform
import sys
import time
import zipfile
import numpy as np
from common import make_cell, development_subset

ROOT = Path(__file__).resolve().parent
PREFIX = 'icml2027-characterization-v1/'
EXPECTED_MODELS = {
    'meta-llama__Llama-3.2-1B': (3,6,10,13),
    'meta-llama__Llama-3.2-3B': (6,11,17,22),
    'meta-llama__Llama-3.1-8B': (6,13,19,26),
    'Qwen__Qwen2.5-1.5B': (6,11,17,22),
    'Qwen__Qwen2.5-3B': (7,14,22,29),
    'Qwen__Qwen2.5-7B': (6,11,17,22),
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.partial')
    with temp.open('w', encoding='utf-8') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    temp.replace(path)


def source_identity():
    manifest_path = ROOT/'source_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for name, expected in manifest['files'].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe source manifest path')
        path = ROOT/name
        if path.is_symlink() or digest(path) != expected:
            raise ValueError('Source changed: '+name+'; use a new protocol/package and output')
    return digest(manifest_path)


def safe_members(zf):
    names = zf.namelist()
    if len(names) != len(set(names)):
        raise ValueError('Duplicate ZIP member')
    for item in zf.infolist():
        path = PurePosixPath(item.filename)
        if path.is_absolute() or '..' in path.parts or '\\' in item.filename:
            raise ValueError('Unsafe ZIP member')
        if ((item.external_attr >> 16) & 0o170000) == 0o120000:
            raise ValueError('Symlink ZIP member')
        if not item.filename.startswith(PREFIX):
            raise ValueError('Unexpected archive root')
    return names


def read_cell(zf, name, *, development_only=False, registration='semantic'):
    base = PREFIX+'analysis/'+name+'/'
    names = set(zf.namelist())
    if base+'rows.json' in names:
        row_name = base+'rows.json'
    else:
        export = json.loads(zf.read(base+'coordinate_export.json'))
        reference = PurePosixPath(export['rows_and_potential_archive_relative_directory'])
        if reference.is_absolute() or '..' in reference.parts:
            raise ValueError('Unsafe shared metadata reference')
        row_name = PREFIX+str(reference)+'/rows.json'
    rows = json.loads(zf.read(row_name))['rows']
    with np.load(io.BytesIO(zf.read(base+'coordinates.npz')), allow_pickle=False) as archive:
        z = archive['z']
    if z.dtype != np.float64 or z.shape != (len(rows),16) or not np.isfinite(z).all():
        raise ValueError('Expected finite float64 q16 coordinates')
    domains = {r['domain'] for r in rows}
    if len(domains) != 1:
        raise ValueError('Mixed coordinate domains')
    if development_only:
        z, rows = development_subset(z, rows)
    return make_cell(z, rows, domains.pop(), registration)


def preflight(archive, config):
    actual = digest(archive)
    if actual != config['parent_archive_sha256']:
        raise ValueError('Input archive SHA256 mismatch; no fit started')
    cells = []
    with zipfile.ZipFile(archive) as zf:
        names = safe_members(zf)
        bad = zf.testzip()
        if bad:
            raise ValueError('ZIP CRC failure: '+bad)
        observed = sorted(n[len(PREFIX+'analysis/'):-len('/coordinates.npz')]
                          for n in names if n.startswith(PREFIX+'analysis/') and n.endswith('/coordinates.npz'))
        expected = sorted(f'{model}__{domain}__{layer}' for model,layers in EXPECTED_MODELS.items()
                          for domain in ('C7','D4','S3') for layer in layers)
        if observed != expected:
            raise ValueError('Expected the exact six-model/four-readout/three-domain 72-cell panel')
        for name in observed:
            cell = read_cell(zf,name)
            cells.append({'name':name,'domain':cell.domain,'rows':len(cell.rows),
                          'test_splits':list(cell.test_splits),
                          'train':int(cell.train.sum()),'validation':int(cell.validation.sum())})
    return {'archive_sha256':actual,'cells':cells,'coordinate_cells':len(cells),
            'old_calibration_required':False,'model_weights_required':False,
            'raw_forward_replay_supported':False,'full_hidden_states_required_for_these_comparisons':False}


def jobs_for(cells,config,mode):
    if mode == 'pilot':
        chosen = {}
        for c in cells:
            # Fixed before fitting: Qwen1.5B first readout for each domain.
            if c['name'] in ('Qwen__Qwen2.5-1.5B__C7__6', 'Qwen__Qwen2.5-1.5B__D4__6', 'Qwen__Qwen2.5-1.5B__S3__6'):
                chosen[c['domain']] = c
        if len(chosen) != 3:
            raise ValueError('Fixed development benchmark cells missing')
        cells = [chosen[d] for d in ('C7','D4_corner','S3_role')]
    jobs = []
    for c in cells:
        for registration in (['semantic'] if mode == 'pilot' else config['registrations']):
            for component,domains in config['components'].items():
                if c['domain'] in domains:
                    jobs.append({'cell':c['name'],'domain':c['domain'], 'registration':registration,'component':component})
    return jobs


def job_id(job):
    return job['cell']+'--'+job['registration']+'--'+job['component']


def execute_job(archive,job,config,mode):
    start = time.monotonic()
    with zipfile.ZipFile(archive) as zf:
        cell = read_cell(zf,job['cell'],development_only=mode=='pilot',registration=job['registration'])
    splits = ['pilot'] if mode == 'pilot' else list(cell.test_splits)
    if job['component'] == 'compact':
        from compact import evaluate_compact
        result = evaluate_compact(cell,{**config,'technical_test':mode=='pilot'},splits)
    else:
        from context import evaluate_context
        result = evaluate_context(cell,{**config,'technical_test':mode=='pilot'},splits)
    # Fail before returning a checkpoint if any code emitted NaN/Infinity.
    canonical(result)
    return {'job':job,'result':result,'elapsed_seconds':time.monotonic()-start,
            'evaluation_splits':splits,'scientific_role':'excluded_development_benchmark' if mode=='pilot' else 'exploratory_existing_test_inspected',
            'paper_eligible':False,'confirmation_authorized':False}


@contextmanager
def output_lock(output):
    with (output/'.lock').open('a+') as f:
        try:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('Another process owns this output directory') from exc
        try: yield
        finally: fcntl.flock(f,fcntl.LOCK_UN)


def verify_checkpoint(path, job, identity_sha):
    data = json.loads(Path(path).read_text())
    expected = data.pop('payload_sha256', None)
    if expected != hashlib.sha256(canonical(data)).hexdigest():
        raise ValueError('Checkpoint payload is corrupt: '+job_id(job))
    if data.get('identity_sha256') != identity_sha or data.get('job') != job:
        raise ValueError('Checkpoint identity mismatch')
    return data


def status(output):
    state = json.loads((output/'state.json').read_text())
    print(json.dumps(state,indent=2))


def collect(output):
    state = json.loads((output/'state.json').read_text())
    if state['status'] != 'complete':
        raise ValueError('Collection requires complete outputs; partial files remain in output')
    identity = json.loads((output/'identity.json').read_text())
    identity_sha = hashlib.sha256(canonical(identity)).hexdigest()
    expected_ids = set(identity['job_ids'])
    if set(state['checkpoints']) != expected_ids or state['completed_jobs'] != len(expected_ids):
        raise ValueError('Complete state does not cover all planned jobs')
    for identifier, receipt in state['checkpoints'].items():
        path = output/'checkpoints'/(identifier+'.json')
        if digest(path) != receipt['sha256']:
            raise ValueError('Checkpoint file changed before collection: '+identifier)
        data = json.loads(path.read_text())
        if job_id(data['job']) != identifier:
            raise ValueError('Wrong checkpoint job name')
        verify_checkpoint(path, data['job'], identity_sha)
    destination = output.with_name(output.name+'.results.zip')
    partial = destination.with_name(destination.name+'.partial')
    files = [p for p in output.rglob('*') if p.is_file() and p.name != '.lock' and not p.name.endswith('.partial')]
    with zipfile.ZipFile(partial,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as zf:
        for p in sorted(files):
            if p.is_symlink(): raise ValueError('Unexpected output symlink')
            zf.write(p,output.name+'/'+str(p.relative_to(output)))
    with zipfile.ZipFile(partial) as check:
        if check.testzip(): raise ValueError('Result ZIP integrity check failed')
    partial.replace(destination)
    sidecar = destination.with_name(destination.name+'.sha256')
    sidecar.write_text(digest(destination)+'  '+destination.name+'\n')
    print('RESULT_ARCHIVE '+str(destination),flush=True)
    print('No Pod stop/delete is performed by this program.',flush=True)
    return destination


def run(args,config,source_sha,info):
    output = Path(args.output).absolute()
    if output.is_symlink(): raise ValueError('Output cannot be a symlink')
    if output.resolve().is_relative_to(ROOT):
        raise ValueError('Output must be outside the source package')
    if (set(output.parts) | set(output.resolve().parts)) & {'icml2027-characterization-v1','icml2027-bf16-f64-v2','icml2027-bf16-v1','frozen_run','results','incoming','imports'}:
        raise ValueError('Output cannot be inside a preserved input or historical results tree')
    if not output.exists(): output.mkdir(parents=True)
    if not output.is_dir(): raise ValueError('Output must be a directory')
    # Never use the old study's output or the parent raw input tree.
    if output.name in ('icml2027-characterization-v1','icml2027-bf16-f64-v2','icml2027-bf16-v1'):
        raise ValueError('A fresh compact/context output directory is required')
    jobs = jobs_for(info['cells'],config,args.mode)
    identity = {'study_id':config['study_id'],'source_manifest_sha256':source_sha,
                'protocol_sha256':digest(ROOT/'protocol.json'),'input_archive_sha256':info['archive_sha256'],
                'mode':args.mode,'numpy_version':np.__version__,'python':platform.python_version(),
                'job_ids':[job_id(j) for j in jobs]}
    identity_sha = hashlib.sha256(canonical(identity)).hexdigest()
    with output_lock(output):
        if (output/'identity.json').exists():
            if json.loads((output/'identity.json').read_text()) != identity:
                raise ValueError('Existing output identity differs; use a fresh output path')
        elif any(p.name != '.lock' for p in output.iterdir()):
            raise ValueError('Unrecognized nonempty output; no files overwritten')
        else:
            atomic_json(output/'identity.json',identity)
            atomic_json(output/'protocol.json',config)
            atomic_json(output/'source_manifest.json',json.loads((ROOT/'source_manifest.json').read_text()))
            atomic_json(output/'input_inventory.json',info)
        completed = {}; pending=[]
        (output/'checkpoints').mkdir(exist_ok=True)
        for j in jobs:
            path=output/'checkpoints'/(job_id(j)+'.json')
            if path.exists():
                data = verify_checkpoint(path,j,identity_sha)
                completed[job_id(j)]={'sha256':digest(path),'elapsed_seconds':data['elapsed_seconds']}
            else: pending.append(j)
        start=time.monotonic(); previous=0.
        if (output/'state.json').exists(): previous=float(json.loads((output/'state.json').read_text()).get('total_wall_seconds',0))
        def save_state(label, error=None):
            value={'status':label,'study_id':config['study_id'],'mode':args.mode,
                   'completed_jobs':len(completed),'planned_jobs':len(jobs),
                   'elapsed_this_session_seconds':round(time.monotonic()-start,2),
                   'total_wall_seconds':round(previous+time.monotonic()-start,2),
                   'checkpoints':completed,'error':error,'paper_eligible':False,'confirmation_authorized':False,
                   'calibration_repeated':False,'model_forwards':0}
            atomic_json(output/'state.json',value)
        save_state('running')
        print(f'START {args.mode}: {len(completed)}/{len(jobs)} already complete; workers={args.workers}; no calibration, no model loading',flush=True)
        active={}; cursor=0; stopped=False
        try:
            with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn')) as pool:
                while cursor<len(pending) or active:
                    expired=args.max_seconds>0 and time.monotonic()-start>=args.max_seconds
                    while not expired and len(active)<args.workers and cursor<len(pending):
                        job=pending[cursor];cursor+=1
                        print('START_JOB '+job_id(job),flush=True)
                        future=pool.submit(execute_job,str(Path(args.archive).absolute()),job,config,args.mode)
                        active[future]=job
                    if not active:
                        stopped=cursor<len(pending);break
                    done,_=wait(active,timeout=30,return_when=FIRST_COMPLETED)
                    for future in done:
                        job=active.pop(future);result=future.result();result['identity_sha256']=identity_sha
                        path=output/'checkpoints'/(job_id(job)+'.json')
                        result['payload_sha256']=hashlib.sha256(canonical(result)).hexdigest()
                        atomic_json(path,result)
                        completed[job_id(job)]={'sha256':digest(path),'elapsed_seconds':result['elapsed_seconds']}
                        print(f'DONE_JOB {job_id(job)} ({result["elapsed_seconds"]:.1f}s) {len(completed)}/{len(jobs)}',flush=True)
                    save_state('running')
                    if not done:
                        print(f'HEARTBEAT elapsed={time.monotonic()-start:.0f}s completed={len(completed)}/{len(jobs)} active={len(active)}',flush=True)
            if source_identity() != source_sha:
                raise ValueError('Source manifest changed during execution; use a fresh output')
            if digest(args.archive) != info['archive_sha256']:
                raise ValueError('Input archive changed during execution; results cannot be collected')
            save_state('paused_budget' if stopped else 'complete')
        except BaseException as exc:
            save_state('failed',repr(exc));raise
        if stopped:
            print('PAUSED_BUDGET: finished in-flight jobs; repeat the same command to resume. This does not stop Pod billing.',flush=True)
            return 75
        from compare import make_comparison
        atomic_json(output/'validation_comparability.json',make_comparison(output, config['validation_comparability_factor']))
        collect(output)
        return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',help='Exact original characterization results ZIP')
    parser.add_argument('--output',help='Fresh output directory; matching checkpoints resume')
    parser.add_argument('--mode',choices=['preflight','pilot','full','status','collect'],default='preflight')
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--max-seconds',type=float,default=0,help='0=no time ceiling. Otherwise stop scheduling after this session budget, finish in-flight jobs, and resume on next invocation.')
    args=parser.parse_args()
    if args.mode in ('status','collect'):
        if not args.output: parser.error('--output required')
        if args.mode=='status': status(Path(args.output));return 0
        with output_lock(Path(args.output)): collect(Path(args.output))
        return 0
    if not args.archive: parser.error('--archive required')
    if args.workers<1 or args.workers>8 or args.max_seconds<0: parser.error('workers 1..8; max-seconds >= 0')
    config=json.loads((ROOT/'protocol.json').read_text());source_sha=source_identity()
    if np.__version__!=config['numpy_version']:
        raise ValueError(f'Use NumPy {config["numpy_version"]}; found {np.__version__}. Create a separate environment; do not replace the old study environment.')
    print('VERIFY_SOURCE_AND_INPUT',flush=True)
    info=preflight(args.archive,config)
    print('INPUT_READY 72 q16 cells; original ZIP read-only; no full hidden states or model weights needed',flush=True)
    if args.mode=='preflight':
        print(json.dumps({k:v for k,v in info.items() if k!='cells'},indent=2));return 0
    if not args.output: parser.error('--output required')
    return run(args,config,source_sha,info)

if __name__=='__main__':
    try: sys.exit(main())
    except Exception as error:
        print('COMPACT_CONTEXT_ERROR '+str(error),file=sys.stderr,flush=True);sys.exit(2)
