"""Recover full archived prompts by exact row ID; never regenerate prompts.

Coordinate row metadata is a compact index, not a full dataset. The dataset's
storage order can differ from extraction order. Restore extraction order and
verify both canonical dataset identity and serialized extraction-row identity.
"""
from __future__ import annotations
import hashlib
import json


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode()


def restore_rows(adapter, archive, index, spec, item, metadata, merged):
    prefix = spec['archive_prefix'] + '/datasets/' + item['domain']
    manifest_name, rows_name = prefix + '/manifest.json', prefix + '/rows.jsonl'
    manifest_raw = adapter.read_member(archive, index, manifest_name)
    raw = adapter.read_member(archive, index, rows_name)
    manifest = adapter._json(manifest_raw)
    if manifest.get('status') != 'sealed' or manifest.get('rows_path') != 'rows.jsonl':
        raise ValueError('Archived dataset is not a sealed rows.jsonl dataset')
    if _sha(raw) != manifest.get('rows_sha256'):
        raise ValueError('Archived prompt file checksum differs')
    rows = [adapter._json(line) for line in raw.splitlines() if line.strip()]
    if _sha(_canonical(rows)) != manifest.get('dataset_sha256'):
        raise ValueError('Archived dataset identity differs')
    if not all(isinstance(r, dict) and isinstance(r.get('row_id'), str)
               and isinstance(r.get('text'), str) and r['text'] for r in rows):
        raise ValueError('Archived prompt text or row identity missing')
    lookup = {r['row_id']: r for r in rows}
    if len(lookup) != len(rows):
        raise ValueError('Duplicate archived prompt identity')
    ids = [r['row_id'] for r in metadata]
    if len(ids) != len(set(ids)) or set(ids) != set(lookup):
        raise ValueError('Coordinate and archived prompt row sets differ')
    ordered = [lookup[rid] for rid in ids]
    if any(any(full.get(k) != v for k, v in meta.items())
           for meta, full in zip(metadata, ordered)):
        raise ValueError('Coordinate metadata differs from archived prompt')
    digest = _sha(_canonical(ordered))
    if digest != merged.get('dataset_sha256') or digest != merged.get('descriptor', {}).get('dataset_sha256'):
        raise ValueError('Restored prompt order differs from extraction dataset identity')
    serialized = b''.join(_canonical(row) + b'\n' for row in ordered)
    if _sha(serialized) != merged.get('rows_sha256') or len(ordered) != merged.get('n_rows'):
        raise ValueError('Restored prompt serialization differs from extraction rows')
    return ordered, {
        'members': {manifest_name: _sha(manifest_raw), rows_name: _sha(raw)},
        'alignment': 'exact row IDs; all compact metadata matched; no coordinate reordering',
        'extraction_dataset_sha256': digest,
        'extraction_rows_sha256': _sha(serialized),
        'prompt_regeneration': False,
    }
