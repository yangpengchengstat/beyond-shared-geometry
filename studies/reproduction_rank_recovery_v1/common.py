"""Independent, content-bound continuation; the original run remains read-only."""
from pathlib import Path
import importlib.util
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / 'studies/reproduction_v1'
sys.path.insert(0, str(BASE))
from receipt import read, write, sha, identity, lock, safe_output, source_identity, verify_inventory


def load_base(name, filename):
    spec = importlib.util.spec_from_file_location(name, BASE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


historical = load_base('rank_recovery_original_historical', 'historical.py')
runner = load_base('rank_recovery_original_controller', 'run.py')


def recovery_identity():
    manifest = read(HERE / 'source_manifest.json')
    if source_identity() != manifest['parent_source_manifest_sha256']:
        raise ValueError('Recovery requires the exact parent source release')
    for name, digest in manifest['files'].items():
        rel = Path(name)
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('Unsafe recovery source member')
        path = HERE / rel
        if path.is_symlink() or not path.is_file() or sha(path) != digest:
            raise ValueError('Recovery source mismatch: ' + name)
    return sha(HERE / 'source_manifest.json')


def write_once(path, value):
    path = Path(path)
    if path.exists():
        if identity(read(path)) != identity(value):
            raise ValueError('Immutable continuation identity changed: ' + str(path))
    else:
        write(path, value)


def validate_parent(parent):
    parent = Path(parent).resolve()
    original_id = read(parent / 'identity.json')
    if original_id['source_sha256'] != source_identity() or original_id['plan'] != runner.plan():
        raise ValueError('Parent source/queue identity mismatch')
    state = read(parent / 'state.json')
    for name in ('characterization', 'compact_context'):
        if state['stages'].get(name, {}).get('status') != 'complete':
            raise ValueError('Completed parent mainline required: ' + name)
    old = parent / 'historical'
    receipt = read(old / 'source_receipt.json')
    if receipt != {'study_id': 'bsg-historical-design-reproduction-v1',
                   'source': historical.source_snapshot(ROOT), 'environment': historical.environment(),
                   'reproduction_not_new_confirmation': True}:
        raise ValueError('Historical source/environment changed')
    stages = {}
    for stage in ('legacy_pre_full', 'legacy_full'):
        path = old / 'stage_state' / (stage + '.json')
        value = read(path)
        if value.get('status') != 'complete' or value['source_receipt_sha256'] != sha(old / 'source_receipt.json'):
            raise ValueError('Verified completed parent stage required: ' + stage)
        historical.verify_records(old / 'runs' / stage, value['outputs'])
        historical.validate_successful_manifest(old / 'runs' / stage, tier='pre_full' if stage=='legacy_pre_full' else 'full')
        historical.source_binding(old / 'runs' / stage)
        stages[stage] = {'state_sha256': sha(path), 'outputs': value['outputs']}
    failed = old / 'stage_state/afs_pre_full.json'
    failure = read(failed)
    message = failure.get('error', {}).get('message', '')
    if failure.get('status') != 'failed' or failure.get('error', {}).get('type') != 'ValueError' or message != 'requested latent_dim=16 exceeds observed centered training rank 6':
        raise ValueError('This recovery is restricted to the observed legacy q16 embedding rank failure')
    ledger_path = old / 'runs/afs_pre_full/run_ledger.json'
    ledger = read(ledger_path)
    failures = [(key, value) for key, value in ledger['units'].items() if value.get('status') == 'FAILED']
    if len(failures) != 1 or not failures[0][0].startswith('embedding_analysis:') or failures[0][1].get('message') != message:
        raise ValueError('Unexpected failed unit; do not hide unrelated failures')
    descriptor = {'parent_root': str(parent), 'parent_identity_sha256': sha(parent / 'identity.json'),
        'parent_historical_source_receipt_sha256': sha(old / 'source_receipt.json'),
        'imported_stages': stages, 'failed_stage_sha256': sha(failed),
        'failed_ledger_sha256': sha(ledger_path), 'failed_unit': failures[0],
        'recovery_source_sha256': recovery_identity(), 'scientific_pass_claimed': False,
        'original_q16_D': 'INFEASIBLE_RANK_DEFICIENT_PRESERVED',
        'D_followup': 'unchanged frozen rank-aware C7=6,D4=3 design, separately named'}
    return descriptor
