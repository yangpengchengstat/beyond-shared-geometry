#!/usr/bin/env python3
"""Sequential launcher for the five frozen TMLR follow-ups; never historical replay."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time
import uuid
import zipfile

TRUSTED_INPUT = '6a4c38237bf1368d4c77c97504f666d0d933bfebed9417d173fbf7441c0fecef'
STUDIES = ('spectral_finite_bank_v1', 'spectral_extensions_v1', 'context_generalization_v1')
STEPS = ('context_fit', 'context_extract', 'context_analyze', 'spectral_prepare',
         'spectral_analyze', 'extension_prepare', 'witness', 'simulate', 'alternatives')
MIN_FREE = 10 * 2**30


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.partial')
    with tmp.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def relative_file(root, name):
    p = Path(name)
    if p.is_absolute() or '..' in p.parts or not p.parts:
        raise ValueError('Unsafe recorded relative path')
    target = root / p
    if target.is_symlink() or not target.resolve().is_relative_to(root.resolve()):
        raise ValueError('Linked/escaping evidence path: ' + name)
    return target


def tree(root):
    result = {}
    for p in sorted(Path(root).rglob('*')):
        if p.is_symlink():
            raise ValueError('Symlink in evidence: ' + str(p))
        if p.is_file() and not p.name.endswith('.lock') and p.name != 'state.json':
            result[str(p.relative_to(root))] = sha(p)
    return result


def verify(root, files):
    if not files:
        raise ValueError('Empty evidence receipt')
    for name, digest in files.items():
        p = relative_file(Path(root), name)
        if not p.is_file() or sha(p) != digest:
            raise ValueError('Changed or missing completed evidence: ' + str(p))


def sources(source):
    """Verify inherited locks without importing research or GPU modules."""
    source = Path(source)
    files = {}
    for name in STUDIES:
        base = source / 'studies' / name
        manifest = base / 'source_manifest.json'
        record = read(manifest)
        actual = {str(p.relative_to(base)) for p in base.rglob('*') if p.is_file()
                  and p.name != 'source_manifest.json' and '__pycache__' not in p.parts
                  and '.pytest_cache' not in p.parts and p.suffix not in ('.pyc', '.pyo')}
        if actual != set(record['files']):
            raise ValueError('Frozen study coverage differs: ' + name)
        for prefix, mapping in ((base, record['files']), (source, record.get('dependencies', {}))):
            verify(prefix, mapping) if mapping else None
            for rel in mapping:
                p = relative_file(prefix, rel)
                files[str(p.relative_to(source))] = sha(p)
        files[str(manifest.relative_to(source))] = sha(manifest)
    for rel in ('pyproject.toml', 'studies/reproduction_v1/requirements.in'):
        files[rel] = sha(source / rel)
    return files


@contextmanager
def lock(root):
    with (Path(root) / '.batch.lock').open('a+') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Another batch controller is active; do not launch twice')
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def checkpoint(root, state):
    atomic(root / 'state.json', state)


def environment():
    versions = {}
    for p in ('numpy', 'scipy', 'torch', 'transformers', 'huggingface_hub', 'tokenizers', 'accelerate'):
        try:
            versions[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            versions[p] = None
    return {'python': platform.python_version(), 'platform': platform.platform(), 'packages': versions}


def child(command, log, name, root):
    log.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    print('START_STAGE', name, 'log=' + str(log), flush=True)
    with log.open('xb') as out:
        proc = subprocess.Popen(command, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=True)
        try:
            while True:
                try:
                    code = proc.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    print('HEARTBEAT', name, 'elapsed_seconds=' + str(int(time.monotonic()-start)), flush=True)
        except BaseException:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL); proc.wait()
            except ProcessLookupError:
                pass
            raise
    if code:
        print(log.read_text(errors='replace')[-6000:], flush=True)
        raise RuntimeError(f'{name} exited {code}; evidence/logs preserved at {root}')
    print('DONE_STAGE', name, 'elapsed_seconds=' + str(round(time.monotonic()-start, 1)), flush=True)


def stage(root, state, name, output, command, *, resume=False, validate=None):
    old = state['stages'].get(name, {})
    if old.get('status') == 'complete':
        if sha(root / old['receipt']) != old['receipt_sha256']:
            raise ValueError('Completed stage receipt changed: ' + name)
        receipt = read(root / old['receipt'])
        verify(root / old['output'], receipt['files'])
        print('VERIFIED_RESUME', name, flush=True)
        return root / old['output']
    if shutil.disk_usage(root).free < MIN_FREE:
        raise ValueError('Less than 10 GiB free; preserve data and free/extend storage before continuing')
    attempt = uuid.uuid4().hex
    log = root / 'logs' / (name + '-' + attempt + '.log')
    previous = list(old.get('attempts', []))
    previous.append({'attempt': attempt, 'started': now(), 'log': str(log.relative_to(root))})
    # Nonresumable frozen stages always use a new attempt directory. Interrupted outputs stay intact.
    if not resume:
        output = output.with_name(output.name + '-' + attempt)
    state.update(status='running', active_stage=name)
    state['stages'][name] = {'status': 'running', 'output': str(output.relative_to(root)), 'attempts': previous}
    checkpoint(root, state)
    try:
        child(command(output), log, name, root)
        if validate:
            validate(output)
        files = tree(output)
        receipt_path = root / 'receipts' / (name + '-' + attempt + '.json')
        atomic(receipt_path, {'stage': name, 'files': files, 'finished': now(), 'output': str(output.relative_to(root)),
                              'scientific_pass_assessed': False})
        state['stages'][name].update(status='complete', receipt=str(receipt_path.relative_to(root)),
                                    receipt_sha256=sha(receipt_path))
        checkpoint(root, state)
        return output
    except BaseException as e:
        failure = root / 'failures' / (name + '-' + attempt + '.json')
        atomic(failure, {'utc': now(), 'stage': name, 'type': type(e).__name__, 'error': str(e),
                         'log': str(log.relative_to(root)), 'output': str(output.relative_to(root))})
        state['stages'][name].update(status='interrupted' if isinstance(e, KeyboardInterrupt) else 'technical_failure')
        state.update(status='stopped', active_stage=name)
        checkpoint(root, state)
        raise


def script(source, name):
    file = 'run.py' if name == 'context_generalization_v1' else 'runner.py'
    return [sys.executable, '-u', str(source / 'studies' / name / file)]


def prepare(source, archive, root):
    source, archive, root = source.resolve(), archive.resolve(), root.resolve()
    if root.exists():
        raise ValueError('Preparation requires a fresh batch directory; use run to resume an existing batch')
    if root.is_relative_to(source) or source.is_relative_to(root) or archive.is_relative_to(root):
        raise ValueError('Keep source, input archive, and batch outputs in separate directories')
    bound = sources(source)
    if sha(archive) != TRUSTED_INPUT:
        raise ValueError('Original archive checksum differs')
    root.mkdir(parents=True)
    with lock(root):
        atomic(root/'identity.json', {'archive_path': str(archive), 'archive_sha256': TRUSTED_INPUT,
               'source_path': str(source), 'source_files': bound, 'launcher_sha256': sha(Path(__file__)),
               'planned_stages': list(STEPS), 'scope': 'five TMLR supplements; no historical replay or calibration'})
        for rel in bound:
            dst = root/'source_snapshot'/rel; dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source/rel, dst)
        shutil.copyfile(__file__, root/'launcher.py')
        atomic(root/'environment.json', environment())
        state = {'status': 'preparing', 'stages': {}, 'created': now(), 'scientific_pass_assessed': False}
        cp = stage(root, state, 'context_prepare', root/'context',
                   lambda out: script(source, 'context_generalization_v1') + ['prepare', '--archive', str(archive),
                   '--expected-sha256', TRUSTED_INPUT, '--output', str(out)], validate=check_context_prepared)
        state.update(status='awaiting_human_review', context=str(cp.relative_to(root)), active_stage=None)
        checkpoint(root,state)
        print('REVIEW_FILE', cp/'BANK_REVIEW.txt', flush=True)
        print('BANK_SHA256', read(cp/'input_manifest.json')['bank_sha256'], flush=True)
        print('PREPARATION_FINISHED_NO_MODELS_LOADED', flush=True)


def check_context_prepared(p):
    d = read(p/'input_manifest.json')
    if len(d['cells']) != 48 or any(c['status'] != 'prepared' for c in d['cells']):
        raise ValueError('Prepared context panel incomplete; inspect retained input records')


def load(root, source=None, archive=None):
    if root.is_symlink() or not (root/'identity.json').is_file():
        raise ValueError('Not a prepared batch directory')
    identity = read(root/'identity.json')
    source = (source or Path(identity['source_path'])).resolve()
    archive = (archive or Path(identity['archive_path'])).resolve()
    if identity['launcher_sha256'] != sha(Path(__file__)) or sources(source) != identity['source_files']:
        raise ValueError('Launcher or frozen scientific source changed; do not mix versions')
    verify(root/'source_snapshot', identity['source_files'])
    if sha(archive) != identity['archive_sha256']:
        raise ValueError('Original archive changed or missing')
    return source, archive, read(root/'state.json')


def context_path(root, state):
    rel = state.get('context') or state['stages']['context_prepare']['output']
    return root / rel


def require_review(context):
    m = read(context/'input_manifest.json')
    if sha(context/'input_manifest.json') != (context/'input_manifest.sha256').read_text().strip():
        raise ValueError('Context input manifest changed')
    r = read(context/'human_review.json') if (context/'human_review.json').exists() else {}
    if (r.get('bank_sha256') != m['bank_sha256'] or r.get('source_manifest_sha256') != m['source_manifest_sha256']
            or r.get('approval_before_model_execution') is not True):
        raise ValueError('Human review pending. Read BANK_REVIEW.txt and record actual review before running')


def review(root, bank_sha256, attest):
    if not attest:
        raise ValueError('An actual human review is required; no automatic attestation')
    with lock(root):
        source, archive, state = load(root)
        cp = context_path(root,state)
        log = root/'logs'/('review-'+uuid.uuid4().hex+'.log')
        child(script(source,'context_generalization_v1')+['review','--run',str(cp),
              '--bank-sha256',bank_sha256,'--attest-reviewed'],log,'record_human_review',root)
        require_review(cp)
        state.update(status='ready',context=str(cp.relative_to(root)))
        checkpoint(root,state)


def completed_spectrum(p):
    m = read(p/'output_manifest.json'); verify(p,m['files'])
    v = read(p/'validation.json')
    if v['reported'] != 48 or v['planned'] != 48:
        raise ValueError('Incomplete spectral execution')


def completed_extension(p, expected):
    d = read(p/'state.json')
    if d.get('execution_status') != 'complete':
        raise ValueError('Extension has not finished all planned units')
    if len(read(p/'plan.json')['units']) != expected:
        raise ValueError('Extension plan changed')


def run(root, execute_models, execute_analysis, allow_download, source=None, archive=None):
    if not execute_models or not execute_analysis:
        raise ValueError('Full suite requires explicit --execute-models and --execute-analysis')
    with lock(root):
        source, archive, state = load(root,source,archive)
        oldenv = read(root/'environment.json'); current = environment()
        if oldenv['python'] != current['python'] or oldenv['packages'] != current['packages']:
            raise ValueError('Runtime versions changed since preparation; restore the recorded environment')
        cp = context_path(root,state); require_review(cp)
        base = script(source,'context_generalization_v1')
        # Underlying fit/extract/analyze support checksum-verified resume. They share one prepared directory.
        for action in ('fit','extract','analyze'):
            flags = ['--execute-models'] + (['--allow-download'] if allow_download else []) if action=='extract' else ['--execute-analysis']
            stage(root,state,'context_'+action,cp,
                  lambda out,a=action,f=flags:base+[a,'--run',str(out)]+f,resume=True)
        if read(cp/'summary.json').get('status') != 'complete':
            raise ValueError('New-context panel incomplete; do not report full-suite completion')
        spec = script(source,'spectral_finite_bank_v1')
        sp = stage(root,state,'spectral_prepare',root/'spectral-prepared',
                   lambda out:spec+['prepare','--archive',str(archive),'--expected-sha256',TRUSTED_INPUT,'--output',str(out)])
        sa = stage(root,state,'spectral_analyze',root/'spectral-results',
                   lambda out:spec+['analyze','--prepared',str(sp),'--archive',str(archive),'--output',str(out),'--execute-analysis'],
                   validate=completed_spectrum)
        ext = script(source,'spectral_extensions_v1')
        ep = stage(root,state,'extension_prepare',root/'extension-prepared',
                   lambda out:ext+['prepare','--spectral-prepared',str(sp),'--archive',str(archive),'--output',str(out)])
        for action, count in (('witness',48),('simulate',36),('alternatives',2952)):
            stage(root,state,action,root/action,
                  lambda out,a=action:ext+[a,'--output',str(out),'--execute-analysis']+
                      (['--prepared',str(ep)] if a!='simulate' else [])+(['--resume'] if out.exists() else []),
                  resume=True, validate=lambda out,n=count:completed_extension(out,n))
        # Recheck every saved stage receipt after downstream stages. State/lock files are deliberately excluded.
        for name, item in state['stages'].items():
            if item['status']!='complete':raise ValueError('Unfinished stage: '+name)
            if sha(root/item['receipt'])!=item['receipt_sha256']:raise ValueError('Receipt changed: '+name)
            verify(root/item['output'],read(root/item['receipt'])['files'])
        state.update(status='completed_pending_scientific_review', active_stage=None, finished=now(),
                     scientific_pass_assessed=False, automatic_manuscript_promotion=False)
        checkpoint(root,state)
        result = collect_unlocked(root)
        print('RESULT_ARCHIVE',result,flush=True)
        print('ALL_STAGES_FINISHED; review all outcomes; Pod remains running and billable.',flush=True)


def collect_unlocked(root):
    paths = []
    for p in root.rglob('*'):
        if p.is_symlink():raise ValueError('Symlink export refused')
        if p.is_file() and not p.name.endswith('.lock'):paths.append(p)
    size = sum(p.stat().st_size for p in paths)
    if shutil.disk_usage(root).free < size + 2*2**30:
        raise ValueError('Need additional free space for verified export; completed results remain in place')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:6]
    output = root.parent / (root.name+'.'+stamp+'.results.zip')
    tmp = output.with_name(output.name+'.partial')
    inventory = {str(p.relative_to(root)):sha(p) for p in paths}
    with zipfile.ZipFile(tmp,'x',zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for i,p in enumerate(sorted(paths)):
            z.write(p,str(p.relative_to(root)))
            if i % 250 == 0:print('EXPORT_PROGRESS',i,len(paths),flush=True)
        z.writestr('batch_export_manifest.json',json.dumps({'files':inventory,'source_archive_sha256':TRUSTED_INPUT,
            'original_input_archive_included':False,'model_weights_included':False,
            'partial':read(root/'state.json').get('status')!='completed_pending_scientific_review',
            'scientific_pass_assessed':False},sort_keys=True))
    with zipfile.ZipFile(tmp) as z:
        for name,digest in inventory.items():
            h=hashlib.sha256()
            with z.open(name) as f:
                for b in iter(lambda:f.read(2**20),b''):h.update(b)
            if h.hexdigest()!=digest:raise ValueError('Export member checksum mismatch: '+name)
    os.replace(tmp,output)
    output.with_name(output.name+'.sha256').write_text(sha(output)+'  '+output.name+'\n')
    return output


def status(root):
    state=read(root/'state.json');print(json.dumps(state,indent=2))
    for name in ('witness','simulate','alternatives'):
        p=root/name/'state.json'
        if p.is_file():print(name,json.dumps(read(p)))
    active=state.get('active_stage')
    if active and active in state['stages']:
        logfile=root/state['stages'][active]['attempts'][-1]['log']
        if logfile.exists():print('\nRECENT_STAGE_LOG\n'+'\n'.join(logfile.read_text(errors='replace').splitlines()[-12:]))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    q=s.add_parser('prepare');q.add_argument('--source',type=Path,required=True);q.add_argument('--archive',type=Path,required=True);q.add_argument('--run',type=Path,required=True)
    q=s.add_parser('review');q.add_argument('--run',type=Path,required=True);q.add_argument('--bank-sha256',required=True);q.add_argument('--attest-reviewed',action='store_true')
    q=s.add_parser('run');q.add_argument('--run',type=Path,required=True);q.add_argument('--source',type=Path);q.add_argument('--archive',type=Path)
    q.add_argument('--execute-models',action='store_true');q.add_argument('--execute-analysis',action='store_true');q.add_argument('--allow-download',action='store_true')
    for name in ('status','collect'):
        q=s.add_parser(name);q.add_argument('--run',type=Path,required=True)
    a=p.parse_args(argv)
    try:
        if a.command=='prepare':prepare(a.source,a.archive,a.run)
        elif a.command=='review':review(a.run,a.bank_sha256,a.attest_reviewed)
        elif a.command=='run':run(a.run,a.execute_models,a.execute_analysis,a.allow_download,a.source,a.archive)
        elif a.command=='status':status(a.run)
        else:
            with lock(a.run):print('RESULT_ARCHIVE',collect_unlocked(a.run),flush=True)
        return 0
    except (Exception,KeyboardInterrupt) as e:
        print('BATCH_STOP:',type(e).__name__,str(e),file=sys.stderr,flush=True)
        print('Evidence preserved. Pod remains running and billable.',file=sys.stderr,flush=True)
        return 2


if __name__=='__main__':
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt('Termination requested')))
    raise SystemExit(main())
