#!/usr/bin/env python3
"""Verify prepared TMLR study sources without opening private research data."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

IGNORED = {'__pycache__', '.pytest_cache', 'local_results', 'incoming', 'imports', '.venv'}

def verify(root: Path, study_name: str = 'context_generalization_v1') -> dict:
    if study_name not in {'context_generalization_v1', 'spectral_extensions_v1'}:
        raise ValueError('unknown TMLR study')
    study = root / 'studies' / study_name
    manifest_file = study / 'source_manifest.json'
    manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
    expected = manifest.get('files', {})
    if not expected:
        raise ValueError('empty source manifest: ' + study_name)
    actual = {}
    for p in sorted(study.rglob('*')):
        rel = p.relative_to(study)
        if any(part in IGNORED for part in rel.parts) or p.suffix in {'.pyc', '.pyo'} or p == manifest_file:
            continue
        if p.is_symlink():
            raise ValueError('study symlink is not allowed: ' + rel.as_posix())
        if p.is_file():
            actual[rel.as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    if actual != expected:
        changed = sorted(k for k in set(actual) | set(expected) if actual.get(k) != expected.get(k))
        raise ValueError('study source mismatch: ' + ', '.join(changed))
    dependencies = manifest.get('dependencies', {})
    if not dependencies:
        raise ValueError('missing inherited dependency identities')
    for name, digest in dependencies.items():
        rel = Path(name)
        target = root / rel
        if rel.is_absolute() or '..' in rel.parts or '\\' in name or target.is_symlink() or not target.resolve().is_relative_to(root.resolve()):
            raise ValueError('unsafe dependency path: ' + name)
        if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise ValueError('inherited dependency mismatch: ' + name)
    return {'files': len(actual), 'dependencies': len(dependencies), 'protocol_id': manifest.get('protocol_id'),
            'model_execution': False, 'research_data_opened': False}

def verify_all(root: Path) -> dict:
    metadata = json.loads((root / 'release/release_manifest.json').read_text(encoding='utf-8'))
    requested = []
    if metadata.get('context_generalization_implementation_added'):
        requested.append('context_generalization_v1')
    if metadata.get('spectral_extensions_implementation_added'):
        requested.append('spectral_extensions_v1')
    if not requested:
        raise ValueError('no prepared TMLR studies declared')
    return {name: verify(root, name) for name in requested}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(verify_all(args.root), indent=2, sort_keys=True))
