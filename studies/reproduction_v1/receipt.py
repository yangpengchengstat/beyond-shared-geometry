"""Content-bound receipts for the fresh replication namespace."""
from __future__ import annotations
import contextlib, fcntl, hashlib, json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()

def identity(value):
    return hashlib.sha256(canonical(value)).hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.partial')
    with temp.open('w') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
        stream.flush(); os.fsync(stream.fileno())
    temp.replace(path)

def safe_output(path):
    path = Path(path).absolute()
    if path.is_symlink() or path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('Use a fresh output outside the publication source tree')
    blocked={'frozen_run','incoming','imports','icml2027-bf16-v1','icml2027-bf16-f64-v2','icml2027-characterization-v1'}
    if set(path.resolve().parts)&blocked:
        raise ValueError('Preserved historical namespaces cannot be outputs')
    return path

def source_identity():
    manifest = ROOT/'studies/reproduction_v1/source_manifest.json'
    data = read(manifest)
    for name, expected in data['files'].items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or '\\' in name:
            raise ValueError('Unsafe source path')
        file = ROOT/path
        if file.is_symlink() or not file.is_file() or sha(file)!=expected:
            raise ValueError('Changed/missing frozen replication source: '+name)
    return sha(manifest)

@contextlib.contextmanager
def lock(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+') as stream:
        try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:raise ValueError('Another writer owns '+str(path)) from exc
        try:yield
        finally:fcntl.flock(stream,fcntl.LOCK_UN)

def inventory(directory):
    directory=Path(directory);result={}
    if not directory.exists():return result
    for path in sorted(directory.rglob('*')):
        if path.is_symlink():raise ValueError('Symlink in experiment output')
        if not path.is_file() or path.name.startswith('.') or path.name.endswith('.partial'):continue
        result[str(path.relative_to(directory))]=sha(path)
    return result

def verify_inventory(directory, files):
    directory=Path(directory)
    for name, expected in files.items():
        rel=Path(name)
        if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe receipt member')
        path=directory/rel
        if path.is_symlink() or not path.is_file() or sha(path)!=expected:
            raise ValueError('Output receipt mismatch: '+name)
    return True
