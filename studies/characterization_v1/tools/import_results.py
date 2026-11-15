#!/usr/bin/env python3
"""Validate and privately import frozen characterization-v1 result archives.

No incoming Python is imported or executed. No model, network or GPU operation
exists here. CLI input paths are relative to the current directory and must stay inside the
same research project. This validates transport, source contract, artifact
coverage and optional coordinate integrity; it is not a scientific refit or a
manuscript-promotion tool.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import struct
import sys
import tempfile
import zipfile

_LOCATION = Path(__file__).resolve()
# Installed research checkout and standalone public release use the same files.
# Choose by this script's own path components; never search personal folders.
if _LOCATION.parents[4].name == 'submission' and _LOCATION.parents[3].name == 'icml2027':
    PROJECT = _LOCATION.parents[5]
    STUDY_REL = Path('submission/icml2027/studies/characterization_v1')
else:
    PROJECT = _LOCATION.parents[3]
    STUDY_REL = Path('studies/characterization_v1')
SOURCE_REL = STUDY_REL / 'frozen_run'
MANIFEST_SHA256 = '5832afe9f3d31b4962ab30ba6c717f151bcb99a14b23ae25f146108a3396f739'
SOURCE_SHA256 = 'ebbc9915ee622bd19c26228c78d0167f89acb2f073553cb54c4ca2dec29c2c89'
SOURCE_PREFIX = 'bsg-characterization-v1/'
RESULT_PREFIX = 'icml2027-characterization-v1/'
STUDY_ID = 'icml2027-characterization-v1'
COMPLETE_STATUS = 'complete_limited_precision_characterization'
HEX = re.compile(r'[0-9a-f]{64}\Z')
MAX_FILES = 10000
MAX_FILE = 1024**3
MAX_TOTAL = 12 * 1024**3
MAX_RATIO = 1500
MAX_JSON = 64 * 1024**2
MAX_NPZ = 512 * 1024**2
MAX_HEADER = 65536


class ValidationError(ValueError):
    pass


def require(test, message):
    if not test:
        raise ValidationError(message)


def boundary(path, project=PROJECT):
    """Reject outside paths before stat/open, including symlink escapes."""
    project = Path(project).absolute()
    path = Path(path)
    if not path.is_absolute():
        path = project / path
    lexical = Path(os.path.abspath(path))
    require(lexical.is_relative_to(project), 'Path is outside the authorized research project')
    # Do not follow any project-child symlink, even one that happens to reenter.
    cursor = project
    for part in lexical.relative_to(project).parts:
        cursor = cursor / part
        require(not cursor.is_symlink(), 'Symlinks are not accepted for input/output paths')
    resolved = lexical.resolve()
    require(resolved.is_relative_to(project.resolve()), 'Resolved path escapes research project')
    return resolved


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def no_duplicate_keys(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, 'Duplicate JSON key: ' + str(key))
        value[key] = item
    return value


def parse_json(data, context='JSON'):
    require(len(data) <= MAX_JSON, context + ' exceeds JSON size budget')
    try:
        return json.loads(data.decode('utf-8'), object_pairs_hook=no_duplicate_keys,
                          parse_constant=lambda s: (_ for _ in ()).throw(ValidationError('Nonfinite JSON constant')))
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValidationError('Malformed ' + context) from exc


def safe_name(name):
    require(isinstance(name, str) and name and '\\' not in name and '\x00' not in name,
            'Unsafe archive path')
    require(not name.startswith('/') and ':' not in name, 'Absolute/drive archive path rejected')
    trimmed = name[:-1] if name.endswith('/') else name
    parts = trimmed.split('/')
    require(all(p not in ('', '.', '..') for p in parts), 'Traversal/noncanonical archive path rejected')
    require(str(PurePosixPath(trimmed)) == trimmed, 'Noncanonical archive path')
    return trimmed


def inspect_zip(z, *, max_total=MAX_TOTAL, max_file=MAX_FILE, max_files=MAX_FILES):
    infos = z.infolist()
    require(len(infos) <= max_files, 'ZIP member-count budget exceeded')
    seen, total = set(), 0
    for item in infos:
        name = safe_name(item.filename)
        require(name not in seen, 'Duplicate ZIP member: ' + name)
        seen.add(name)
        mode = item.external_attr >> 16
        file_type = stat.S_IFMT(mode)
        require(file_type in (0, stat.S_IFREG, stat.S_IFDIR), 'Symlink/special ZIP member rejected')
        require(not (mode & (stat.S_ISUID | stat.S_ISGID)), 'Privileged ZIP mode rejected')
        require(not item.flag_bits & 1, 'Encrypted ZIP member rejected')
        require(item.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED), 'Unsupported ZIP compression')
        require(0 <= item.file_size <= max_file and item.compress_size >= 0, 'ZIP member-size budget exceeded')
        if item.file_size:
            require(item.compress_size > 0 and item.file_size / item.compress_size <= MAX_RATIO,
                    'ZIP compression-ratio budget exceeded')
        total += item.file_size
        require(total <= max_total, 'ZIP total-size budget exceeded')
        # Files may not be parents of other archive entries.
    files = {safe_name(i.filename) for i in infos if not i.is_dir()}
    for name in files:
        require(not any(str(p) in files for p in PurePosixPath(name).parents if str(p) != '.'),
                'ZIP file/directory collision rejected')
    return {'members': len(infos), 'uncompressed_bytes': total, 'files': sorted(files)}


def check_sidecar(archive, sidecar):
    require(Path(sidecar).stat().st_size <= 4096, 'Checksum sidecar too large')
    text = Path(sidecar).read_text(encoding='utf-8').strip()
    match = re.fullmatch(r'([0-9a-fA-F]{64})[ \t]+\*?([^\r\n]+)', text)
    require(match is not None, 'Checksum must contain exactly one sha256sum-format entry')
    require(match[2] == Path(archive).name, 'Checksum filename differs from supplied ZIP')
    actual = sha_file(archive)
    require(actual == match[1].lower(), 'Result ZIP SHA256 mismatch')
    return actual


@dataclass
class Frozen:
    config: dict
    manifest: dict
    manifest_sha256: str
    zip_sha256: str
    verification_mode: str = 'pinned_zip_sha256_and_manifest'


def load_frozen(path):
    path = Path(path)
    if path.is_dir():
        manifest_path = path / 'source_manifest.json'
        require(not manifest_path.is_symlink() and sha_file(manifest_path) == MANIFEST_SHA256,
                'Expanded frozen source manifest differs from pinned v1')
        manifest = parse_json(manifest_path.read_bytes(), 'expanded source manifest')
        require(isinstance(manifest, dict) and manifest, 'Missing expanded source map')
        for name, expected in manifest.items():
            safe_name(name)
            require(HEX.fullmatch(expected), 'Invalid expanded source checksum')
            source_file = path / name
            cursor = path
            for part in Path(name).parts:
                cursor /= part
                require(not cursor.is_symlink(), 'Expanded source symlink rejected')
            require(source_file.is_file() and sha_file(source_file) == expected, 'Expanded frozen source checksum mismatch: ' + name)
        config = parse_json((path / 'study_config.json').read_bytes(), 'expanded study config')
        require(config['study_id'] == STUDY_ID and len(config['models']) == 6 and
                all(len(v['layers']) == 4 for v in config['models']), 'Frozen fixed panel invalid')
        return Frozen(config, manifest, MANIFEST_SHA256, SOURCE_SHA256, 'pinned_expanded_manifest_and_all_listed_file_bytes')
    require(sha_file(path) == SOURCE_SHA256, 'Frozen execution ZIP differs from the pinned v1 source')
    try:
        with zipfile.ZipFile(path) as z:
            inspect_zip(z)
            require(z.testzip() is None, 'Frozen source ZIP CRC mismatch')
            manifest_bytes = z.read(SOURCE_PREFIX + 'source_manifest.json')
            manifest = parse_json(manifest_bytes, 'source manifest')
            expected = {SOURCE_PREFIX + 'source_manifest.json'}
            require(isinstance(manifest, dict) and manifest, 'Missing source manifest map')
            for name, digest in manifest.items():
                safe_name(name)
                require(HEX.fullmatch(digest) is not None, 'Invalid source file checksum')
                full = SOURCE_PREFIX + name
                require(sha_bytes(z.read(full)) == digest, 'Frozen source checksum mismatch: ' + name)
                expected.add(full)
            require(set(i.filename for i in z.infolist() if not i.is_dir()) == expected,
                    'Unexpected or missing frozen source member')
            config = parse_json(z.read(SOURCE_PREFIX + 'study_config.json'), 'frozen study config')
    except (zipfile.BadZipFile, KeyError, EOFError) as exc:
        raise ValidationError('Frozen execution ZIP is invalid') from exc
    require(config['study_id'] == STUDY_ID and len(config['models']) == 6
            and all(len(v['layers']) == 4 for v in config['models']), 'Frozen six-model/four-layer panel invalid')
    return Frozen(config, manifest, sha_bytes(manifest_bytes), SOURCE_SHA256)


class Archive:
    def __init__(self, z):
        self.z = z
        self.inventory = inspect_zip(z)
        self.names = set(self.inventory['files'])
        require(all(n.startswith(RESULT_PREFIX) for n in self.names), 'Unexpected result ZIP root')
        self.names = {n[len(RESULT_PREFIX):] for n in self.names}
        root_files = {'study_contract.json', 'summary.json', 'execution_state.json', 'budget.json', 'collection_status.json'}
        for n in self.names:
            require(n in root_files or (n.startswith(('analysis/', 'provenance/')) and n.endswith(('.json', '.npz', '.json.partial', '.npz.partial'))),
                    'Unexpected result payload type/path: ' + n)
        self.hashes = {}

    def raw(self, name):
        require(name in self.names, 'Missing artifact: ' + name)
        return self.z.read(RESULT_PREFIX + name)

    def js(self, name):
        return parse_json(self.raw(name), name)

    def digest(self, name):
        if name not in self.hashes:
            h = hashlib.sha256()
            with self.z.open(RESULT_PREFIX + name) as f:
                for block in iter(lambda: f.read(2**20), b''):
                    h.update(block)
            self.hashes[name] = h.hexdigest()
        return self.hashes[name]

    def verify_map(self, directory, mapping, expected):
        require(isinstance(mapping, dict) and set(mapping) == set(expected), 'Artifact-checksum key set mismatch: ' + directory)
        for name, checksum in mapping.items():
            safe_name(name)
            require('/' not in name and isinstance(checksum, str) and HEX.fullmatch(checksum), 'Unsafe artifact checksum entry')
            require(self.digest(directory + '/' + name) == checksum, 'Artifact checksum mismatch: ' + directory + '/' + name)


def npy_header(data):
    stream = io.BytesIO(data)
    require(stream.read(6) == b'\x93NUMPY', 'Invalid NPY magic')
    version = stream.read(2)
    require(version in (b'\x01\x00', b'\x02\x00', b'\x03\x00'), 'Unsupported NPY format')
    size_bytes = stream.read(2 if version == b'\x01\x00' else 4)
    require(len(size_bytes) == (2 if version == b'\x01\x00' else 4), 'Truncated NPY length')
    size = int.from_bytes(size_bytes, 'little')
    require(0 < size <= MAX_HEADER, 'NPY header-size budget exceeded')
    try:
        header = ast.literal_eval(stream.read(size).decode('utf-8' if version == b'\x03\x00' else 'latin1').strip())
    except (ValueError, SyntaxError, UnicodeError, MemoryError, RecursionError) as exc:
        raise ValidationError('Malformed NPY header') from exc
    require(isinstance(header, dict) and set(header) == {'descr', 'fortran_order', 'shape'}, 'Invalid NPY header keys')
    dtype, shape = header['descr'], header['shape']
    require(isinstance(dtype, str) and re.fullmatch(r'[<>=|][fiu][1248]', dtype), 'Object/structured/unsupported NPY dtype rejected')
    require(isinstance(shape, tuple) and len(shape) in (1, 2) and
            all(type(v) is int and 0 < v <= 10000000 for v in shape), 'Invalid NPY shape')
    require(type(header['fortran_order']) is bool, 'Invalid NPY array-order flag')
    count = math.prod(shape)
    require(count <= 100000000 and len(data) == stream.tell() + count * int(dtype[2:]), 'NPY payload-size mismatch')
    return {'shape': list(shape), 'dtype': dtype, 'bytes': len(data), 'offset': stream.tell(), 'fortran_order': header['fortran_order']}


def npz_arrays(a, path, expected, numerical):
    raw = a.raw(path)
    require(len(raw) <= MAX_NPZ, 'NPZ compressed-size budget exceeded')
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            inv = inspect_zip(z, max_total=MAX_NPZ, max_file=MAX_NPZ, max_files=12)
            require(set(inv['files']) == {key + '.npy' for key in expected}, 'NPZ array key set mismatch: ' + path)
            require(z.testzip() is None, 'NPZ CRC mismatch')
            headers = {key: npy_header(z.read(key + '.npy')) for key in expected}
        arrays = None
        if numerical:
            import numpy as np  # Explicitly optional. No Torch or study source imports.
            with np.load(io.BytesIO(raw), allow_pickle=False) as values:
                arrays = {key: values[key] for key in expected}
            require(all(np.isfinite(v).all() for v in arrays.values()), 'Nonfinite NPZ coordinate/PCA value')
        return headers, arrays
    except (zipfile.BadZipFile, EOFError, KeyError) as exc:
        raise ValidationError('Malformed NPZ: ' + path) from exc


def npz_integers(a, path, keys):
    # Compact mapping indices are always checked, even without NumPy.
    output = {}
    with zipfile.ZipFile(io.BytesIO(a.raw(path))) as z:
        for key in keys:
            raw = z.read(key + '.npy')
            h = npy_header(raw)
            require(len(h['shape']) == 1 and h['dtype'][-2:] in ('i4', 'i8'), 'Expected signed compact index vector')
            endian = h['dtype'][0]
            prefix = '<' if endian == '<' else '>' if endian == '>' else '='
            code = 'i' if h['dtype'][-1] == '4' else 'q'
            output[key] = [v[0] for v in struct.iter_unpack(prefix + code, raw[h['offset']:])]
    return output


def model_key(spec):
    return re.sub(r'[^A-Za-z0-9_.-]', '__', spec['id'])


def model_identity(identity, spec):
    require(isinstance(identity, dict) and identity.get('model_id') == spec['id']
            and identity.get('model_revision') == spec['revision']
            and identity.get('tokenizer_revision') == spec['tokenizer_revision'], 'Pinned model/tokenizer identity mismatch')


def finite_metrics(value):
    require(isinstance(value, dict) and value, 'Missing mean-loss metrics')
    require(all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in value.values()),
            'Malformed/nonfinite mean-loss metric')


def read_rows(a, path, domain, manifest):
    obj = a.js(path)
    rows = obj.get('rows')
    require(isinstance(rows, list) and len(rows) == manifest['n_rows'], 'Exported row count differs from merged cache')
    keys = ('row_id', 'family_id', 'split', 'state_id', 'paraphrase_id', 'orbit_id', 'domain')
    ids, observations = set(), set()
    states = {'C7': 7, 'D4': 4, 'S3': 6}[domain]
    for row in rows:
        require(isinstance(row, dict) and set(row) == set(keys), 'Reduced row metadata schema mismatch')
        require(all(isinstance(row[k], str) and row[k] for k in ('row_id', 'family_id', 'split', 'orbit_id', 'domain')),
                'Invalid reduced row identifier')
        require(row['domain'] == {'C7': 'C7', 'D4': 'D4_corner', 'S3': 'S3_role'}[domain], 'Wrong row domain')
        require(type(row['state_id']) is int and 0 <= row['state_id'] < states and type(row['paraphrase_id']) is int
                and row['paraphrase_id'] in (0, 1), 'Invalid state/paraphrase coordinate')
        require(row['row_id'] not in ids, 'Duplicate reduced row ID')
        ids.add(row['row_id'])
        observation = tuple(row[k] for k in ('family_id', 'split', 'paraphrase_id', 'orbit_id', 'state_id'))
        require(observation not in observations, 'Duplicate reduced row observation')
        observations.add(observation)
    # State-complete observations are required by all three finite group tasks.
    assigned_splits = defaultdict(set)
    for row in rows:
        unit = row['family_id'] if domain != 'S3' else (row['family_id'], row['orbit_id'])
        assigned_splits[unit].add(row['split'])
    require(all(len(v) == 1 for v in assigned_splits.values()),
            'Context family leaks across splits' if domain != 'S3' else 'S3 family/entity orbit leaks across splits')
    blocks = defaultdict(set)
    for r in rows:
        blocks[(r['family_id'], r['split'], r['paraphrase_id'], r['orbit_id'])].add(r['state_id'])
    require(all(s == set(range(states)) for s in blocks.values()), 'Incomplete state orbit in exported rows')
    return obj, rows


def validate_s3_frozen_row_bank(rows, cfg):
    """Accept only the exact S3 bank selected by the immutable v1 worker.

    Historical ``select_phase_records(..., 'development')`` retained the
    train-context/validation-entity and validation-context/train-entity blocks.
    Their 9,216 reserved rows are metadata in the merged export, not additional
    test observations or permission to use reserved targets for fitting. The
    new extraction still contains only the four named heldout/support blocks.
    Validate the entire Cartesian bank rather than allow any 'reserved' label.
    Row/compact-index consistency is checked separately by the inverse/first-index mapping.
    """
    families = int(cfg['test_context_families'])
    orbits = int(cfg['test_entity_orbits'])
    blocks = (
        ('train', 48, 'train', 24, 'train'),
        ('validation', 16, 'validation', 8, 'validation'),
        ('train', 48, 'validation', 8, 'reserved'),
        ('validation', 16, 'train', 24, 'reserved'),
        ('pilot', 12, 'pilot', 8, 'pilot'),
        ('train', 48, 'test', orbits, 'test_entity'),
        ('test', families, 'train', 24, 'test_context'),
        ('test', families, 'test', orbits, 'test_joint'),
        ('support', 2, 'test', orbits, 'support'),
    )
    expected = set()
    for context, nf, entity, no, split in blocks:
        for fi in range(nf):
            family = f'structural_{context}_{fi:03d}'
            for oi in range(no):
                orbit = f"{'pilot' if entity == 'pilot' else 'main'}_triple_{entity}_{oi:03d}"
                for paraphrase in (0, 1):
                    for state in range(6):
                        row_id = f'S3_role/{family}/p{paraphrase}/{orbit}/s{state}'
                        expected.add((row_id, family, orbit, paraphrase, state, split, 'S3_role'))
    keys = ('row_id', 'family_id', 'orbit_id', 'paraphrase_id', 'state_id', 'split', 'domain')
    actual = {tuple(row[key] for key in keys) for row in rows}
    require(len(actual) == len(rows) and actual == expected,
            'S3 complete frozen development/reserved/test/support metadata bank differs')


def validate_result(archive, frozen, *, numerical=False):
    """Validate all present artifacts; missing outputs remain explicit, never fabricated."""
    cfg = frozen.config
    with zipfile.ZipFile(archive) as z:
        a = Archive(z)
        require(z.testzip() is None, 'Result ZIP CRC mismatch')
        contract = a.js('study_contract.json')
        require(contract.get('study') == cfg and contract.get('bundle_source_manifest_sha256') == frozen.manifest_sha256
                and contract.get('parent_input_hashes') == cfg['parent_pins'], 'Study/source contract mismatch')
        require(contract.get('technical_test') is False and contract.get('old_study_continued') is False
                and contract.get('formal_inference_enabled') is False, 'Study contract scientific scope changed')
        missing, warnings, checked = [], [], []
        for name, value in [('provenance/source_manifest.json', frozen.manifest), ('provenance/study_config.json', cfg)]:
            if name in a.names:
                require(a.js(name) == value, 'Provenance differs from frozen source/config: ' + name)
                if name.endswith('source_manifest.json'):
                    require(a.digest(name) == frozen.manifest_sha256, 'Source manifest byte identity mismatch')
            else:
                missing.append(name)
        summary = a.js('summary.json') if 'summary.json' in a.names else {}
        if not summary:
            missing.append('summary.json')
        else:
            require(summary.get('study_id') == cfg['study_id'], 'Summary study ID mismatch')
            require(summary.get('paper_eligible') is False and summary.get('formal_inference_enabled') is False
                    and summary.get('old_power_gate_passed') is False and summary.get('native_run') is False
                    and summary.get('synthetic_calibration_run') is False, 'Summary attempts unsupported scientific promotion')
        if 'budget.json' in a.names:
            budget = a.js('budget.json')
            require(budget.get('limit_seconds') == cfg['wall_budget_seconds_total'] and
                    isinstance(budget.get('spent_seconds'), (int, float)) and math.isfinite(budget['spent_seconds'])
                    and budget['spent_seconds'] >= 0, 'Malformed budget record')
            if budget.get('running'):
                warnings.append('Archive budget record still marks a running invocation; review worker state.')
        else:
            missing.append('budget.json')
        if 'collection_status.json' in a.names:
            collection = a.js('collection_status.json')
            require(collection.get('artifact_collection_is_not_a_scientific_pass') is True and
                    collection.get('models_loaded_by_collection') is False, 'Collection scope flags changed')
            require(collection.get('summary_status') == summary.get('status', 'incomplete_no_summary'), 'Collection and summary status differ')
        else:
            missing.append('collection_status.json')
        versions = None
        if 'provenance/dependency_build_versions.json' not in a.names:
            missing.append('provenance/dependency_build_versions.json')
        else:
            versions = a.js('provenance/dependency_build_versions.json')
            require(isinstance(versions, dict) and versions and all(isinstance(v, str) and v for v in versions.values()),
                    'Invalid dependency-build provenance')
        markers, legacy, fields, projected = [], [], [], []
        expected_dirs, expected_checkpoints = set(), set()
        numerical_scope = []
        validated_s3_banks = []
        for spec in cfg['models']:
            key = model_key(spec)
            for domain in ('C7', 'D4', 'S3'):
                group = {'C7': 'C7', 'D4': 'D4_corner', 'S3': 'S3_role'}[domain]
                checkpoints = {'extract/' + key + '/' + domain, 'merge/' + key + '/' + domain}
                expected_checkpoints |= checkpoints
                manifests = {}
                for role in ('development', 'heldout', 'merged'):
                    name = 'provenance/' + key + '__' + domain + '__' + role + '.json'
                    if name not in a.names:
                        missing.append(name)
                        continue
                    m = a.js(name)
                    require(m.get('status') == 'complete' and type(m.get('n_rows')) is int and m['n_rows'] > 0,
                            'Invalid cache completion manifest: ' + name)
                    model_identity(m.get('model'), spec)
                    require(list(map(str, m.get('layers', []))) == list(map(str, spec['layers'])), 'Cache layer panel differs from source')
                    require(m.get('readout') == ('final_token' if domain == 'S3' else 'span_mean'), 'Readout provenance mismatch')
                    require(set(m.get('arrays', {})) == set(map(str, spec['layers'])), 'Cache array panel mismatch')
                    for info in m['arrays'].values():
                        require(info.get('shape') == [m['n_rows'], spec['hidden_dimension']] and info.get('dtype') == 'float32'
                                and isinstance(info.get('sha256'), str) and HEX.fullmatch(info['sha256']), 'Invalid raw-cache array provenance')
                    if 'descriptor' in m:
                        descriptor = m['descriptor']
                        require(sha_bytes(canonical(descriptor)) == m.get('cache_key') and
                                all(m.get(k) == descriptor.get(k) for k in ('dataset_sha256', 'model', 'layers', 'readout', 'extraction_code_sha256')),
                                'Cache descriptor/content identity mismatch')
                    if domain == 'S3':
                        potential = m.get('potential_features')
                        require(isinstance(potential, dict) and potential.get('shape') == [m['n_rows'], 3 * spec['hidden_dimension']]
                                and potential.get('dtype') == 'float32' and isinstance(potential.get('sha256'), str)
                                and HEX.fullmatch(potential['sha256']), 'S3 potential-cache provenance missing')
                    if role == 'development':
                        require(a.digest(name) == cfg['expected_development_cache_manifest_hashes'][spec['id'] + '|' + domain],
                                'Original development-cache provenance changed')
                        for event in m.get('telemetry', []):
                            builds = event.get('dependency_build_versions', {}) if event.get('stage') == 'runtime_environment' else {}
                            if versions is not None:
                                require(all(expected is None or versions.get(package) == expected for package, expected in builds.items()),
                                        'Dependency builds differ from pinned development-cache telemetry')
                    if role == 'heldout':
                        require(m['n_rows'] == cfg['all_test_rows_per_model'][domain], 'Heldout/support row count differs from source')
                    manifests[role] = m
                merged = manifests.get('merged')
                if all(role in manifests for role in ('development', 'heldout', 'merged')):
                    require(merged['n_rows'] == manifests['development']['n_rows'] + manifests['heldout']['n_rows'], 'Merged provenance row arithmetic mismatch')
                    require(all(manifests[r]['model'] == merged['model'] and manifests[r].get('extraction_code_sha256') == merged.get('extraction_code_sha256') for r in manifests),
                            'Development/heldout/merged extraction identity mismatch')
                rows, row_obj, compact_arrays = None, None, None
                shared = 'analysis/shared/' + key + '__S3'
                if domain == 'S3' and shared + '/compact_export.json' in a.names:
                    a.verify_map(shared, a.js(shared + '/compact_export.json'), {'potential_unique.npz', 'rows.json'})
                    require(merged is not None, 'S3 compact export lacks merged-cache provenance')
                    row_obj, rows = read_rows(a, shared + '/rows.json', domain, merged)
                    validate_s3_frozen_row_bank(rows, cfg)
                    validated_s3_banks.append(key)
                    keys = row_obj.get('potential_keys')
                    actual_keys = sorted({(r['orbit_id'], r['state_id']) for r in rows})
                    require(keys == [list(k) for k in actual_keys], 'Compact S3 potential key mapping mismatch')
                    h, compact_arrays = npz_arrays(a, shared + '/potential_unique.npz', {'values', 'inverse', 'first_row_indices'}, numerical)
                    require(h['values']['shape'] == [len(keys), merged['potential_features']['shape'][1]],
                            'Compact S3 values shape differs from unique keys')
                    require(h['inverse']['shape'] == [len(rows)] and h['first_row_indices']['shape'] == [len(keys)]
                            and h['values']['dtype'][-2:] == 'f4' and h['inverse']['dtype'][-2:] == 'i4'
                            and h['first_row_indices']['dtype'][-2:] == 'i8', 'Compact S3 NPY mapping schema mismatch')
                    integer_arrays = npz_integers(a, shared + '/potential_unique.npz', ('inverse', 'first_row_indices'))
                    positions = {tuple(k): i for i, k in enumerate(keys)}
                    expected_inverse = [positions[(r['orbit_id'], r['state_id'])] for r in rows]
                    first = {}
                    for i, row in enumerate(rows):
                        first.setdefault((row['orbit_id'], row['state_id']), i)
                    expected_first = [first[tuple(k)] for k in keys]
                    require(integer_arrays['inverse'] == expected_inverse and integer_arrays['first_row_indices'] == expected_first,
                            'Compact S3 inverse/first-occurrence mapping mismatch')
                    if numerical:
                        import numpy as np
                        positions = {tuple(k): i for i, k in enumerate(keys)}
                        inverse = np.array([positions[(r['orbit_id'], r['state_id'])] for r in rows], dtype=np.int64)
                        first = {}
                        for i, row in enumerate(rows):
                            first.setdefault((row['orbit_id'], row['state_id']), i)
                        first_ids = np.array([first[tuple(k)] for k in keys])
                        require(np.array_equal(compact_arrays['inverse'], inverse) and
                                np.array_equal(compact_arrays['first_row_indices'], first_ids), 'Compact S3 inverse/first-occurrence mapping mismatch')
                        numerical_scope.append(shared + ': finite unique vectors; exact inverse/first indices')
                for layer in spec['layers']:
                    cell = key + '__' + domain + '__' + str(layer)
                    directory = 'analysis/' + cell
                    expected_dirs.add(directory)
                    expected_checkpoints.add('analyze/' + cell)
                    marker = directory + '/cell_complete.json'
                    if marker not in a.names:
                        missing.append(marker)
                        # Keep any uncommitted partial data, explicitly without calling it complete.
                        continue
                    require(merged is not None, 'Completed cell lacks merged-cache provenance')
                    required = {'legacy.json', 'pca.npz', 'coordinates.npz'} | ({'fields.json', 'rows.json'} if domain != 'S3' else {'coordinate_export.json'})
                    a.verify_map(directory, a.js(marker), required)
                    markers.append(cell)
                    if domain != 'S3':
                        row_obj, rows = read_rows(a, directory + '/rows.json', domain, merged)
                    else:
                        require(rows is not None, 'Completed S3 cell lacks shared rows/potential export')
                        export = a.js(directory + '/coordinate_export.json')
                        require(export.get('rows_and_potential_archive_relative_directory') == shared and
                                export.get('shared_input_hashes') == a.js(shared + '/compact_export.json') and
                                export.get('full_hidden_state_or_PCA_refit_from_raw_exported') is False, 'S3 coordinate/shared provenance mismatch')
                    counts = Counter(r['split'] for r in rows)
                    new_counts = cfg['S3_new_row_breakdown'] if domain == 'S3' else {'test': cfg['all_test_rows_per_model'][domain]}
                    require(all(counts[k] == n for k, n in new_counts.items()), 'Exported new-test/support split coverage mismatch')
                    require(counts['train'] > 0 and counts['validation'] > 0 and counts['pilot'] > 0, 'Development train/validation/pilot rows missing')
                    allowed = {'train', 'validation', 'pilot'} | set(new_counts)
                    if domain == 'S3':
                        # The shared-row check above already requires the full exact bank.
                        allowed.add('reserved')
                    require(set(counts) <= allowed, 'Reserved/unplanned evaluation rows present')
                    families = {r['family_id'] for r in rows if r['split'] == ('test_joint' if domain == 'S3' else 'test')}
                    require(len(families) == cfg['test_context_families'], 'Test context-family count differs from frozen plan')
                    if domain == 'S3':
                        orbits = {r['orbit_id'] for r in rows if r['split'] == 'test_joint'}
                        require(len(orbits) == cfg['test_entity_orbits'], 'S3 heldout entity-orbit count differs from source')
                    hz, az = npz_arrays(a, directory + '/coordinates.npz', {'z'}, numerical)
                    hp, ap = npz_arrays(a, directory + '/pca.npz', {'center', 'basis'}, numerical)
                    require(hz['z']['shape'] == [len(rows), 16] and hz['z']['dtype'][-2:] == 'f8', 'q16 coordinate shape/dtype mismatch')
                    require(hp['center']['shape'] == [spec['hidden_dimension']] and
                            hp['basis']['shape'] == [spec['hidden_dimension'], 16] and
                            all(v['dtype'][-2:] == 'f8' for v in hp.values()), 'PCA transform shape/dtype mismatch')
                    if numerical:
                        import numpy as np
                        train = np.array([r['split'] == 'train' for r in rows])
                        require(np.allclose(ap['basis'].T @ ap['basis'], np.eye(16), rtol=1e-7, atol=1e-7), 'PCA basis is not orthonormal')
                        require(np.allclose(az['z'][train].mean(0), 0, rtol=0, atol=1e-7 * max(1., float(np.abs(az['z'][train]).max()))),
                                'Training projection is not centered')
                        scale = float(np.mean(np.sum(az['z'][train] ** 2, axis=1)))
                    values = a.js(directory + '/legacy.json')
                    splits = {'test_joint', 'test_context', 'test_entity'} if domain == 'S3' else {'test'}
                    require(isinstance(values, list) and len(values) == len(splits) and
                            {v.get('test_split') for v in values} == splits, 'Legacy split panel mismatch: ' + cell)
                    for v in values:
                        require(v.get('group') == group and str(v.get('layer')) == str(layer) and
                                v.get('model') == merged['model'] and v.get('coordinate_dimension') == 16 and
                                v.get('cache_manifest_sha256') == a.digest('provenance/' + key + '__' + domain + '__merged.json'),
                                'Legacy condition/cache identity mismatch: ' + cell)
                        finite_metrics(v.get('mean_losses'))
                        if numerical:
                            require(math.isclose(v['training_scale'], max(scale, 1e-12), rel_tol=1e-7, abs_tol=1e-10), 'Training energy differs from q16 export')
                        expected_cluster_keys = {r['family_id'] + '|' + r['orbit_id'] for r in rows if r['split'] == v['test_split']}
                        for field in ('family_orbit_prediction', 'family_orbit_context'):
                            values_by_unit = v.get(field)
                            require(isinstance(values_by_unit, dict) and set(values_by_unit) == expected_cluster_keys, 'Family/orbit contrast coverage differs from evaluation rows')
                            require(all(isinstance(t, (int, float)) and math.isfinite(t) for t in values_by_unit.values()), 'Nonfinite cluster contrast')
                        legacy.append(v)
                    if domain != 'S3':
                        field = a.js(directory + '/fields.json')
                        require(field.get('model_id') == spec['id'] and field.get('group') == group and
                                str(field.get('layer')) == str(layer) and field.get('paper_eligible') is False and
                                field.get('formal_inference') is False and field.get('technical_test') is False,
                                'Field condition/scope identity mismatch')
                        generator = field.get('generator', {})
                        require(generator.get('fit_action_count') == (1 if domain == 'C7' else 2) and
                                generator.get('intermediate_true_state_used') is False and
                                generator.get('intermediate_projection_or_snapping') is False, 'Generator supervision/information track changed')
                        for metrics in generator.get('mean_losses', {}).values():
                            finite_metrics(metrics)
                        require(set(generator.get('mean_losses', {})) == {'primitive', 'unseen_composite'}, 'Generator endpoint panel missing')
                        fields.append(field)
                    projected.append({'cell': cell, 'rows': len(rows), 'coordinate_dimension': 16, 'split_counts': dict(sorted(counts.items()))})
                    checked.append(directory)
        actual_dirs = {str(PurePosixPath(n).parent) for n in a.names if n.endswith('/cell_complete.json')}
        require(actual_dirs <= expected_dirs, 'Unexpected model/layer/domain completion marker')
        for n in a.names:
            if n.startswith('analysis/') and len(PurePosixPath(n).parts) >= 3:
                directory = '/'.join(PurePosixPath(n).parts[:2])
                require(directory in expected_dirs or directory == 'analysis/shared', 'Unexpected analysis condition directory')
            if n.endswith('.partial'):
                missing.append('uncommitted_partial_artifact: ' + n)
        if 'execution_state.json' in a.names:
            state = a.js('execution_state.json').get('completed')
            require(isinstance(state, list) and len(state) == len(set(state)) and set(state) <= expected_checkpoints, 'Duplicate/unknown execution checkpoint')
            for marker in markers:
                require('analyze/' + marker in state, 'Cell marker lacks execution completion record')
            if set(state) != expected_checkpoints:
                missing.append('execution_state: expected 108 unique extraction/merge/analysis checkpoints')
        else:
            missing.append('execution_state.json')
        require(len(markers) <= 72 and len(legacy) <= 120 and len(fields) <= 48, 'Panel cardinality overflow')
        if summary:
            actual_legacy_count = sum(len(a.js(n)) for n in a.names if re.fullmatch(r'analysis/[^/]+/legacy\.json', n))
            actual_field_count = sum(bool(re.fullmatch(r'analysis/[^/]+/fields\.json', n)) for n in a.names)
            require(summary.get('legacy_split_records') == actual_legacy_count and summary.get('field_cells') == actual_field_count,
                    'Summary counts differ from actual retained files')
        if summary.get('status') == COMPLETE_STATUS and (summary.get('legacy_split_records') != 120 or summary.get('field_cells') != 48):
            missing.append('summary: fixed 120 legacy records / 48 field cells not complete')
        if len(markers) != 72 or len(legacy) != 120 or len(fields) != 48:
            missing.append('fixed_panel: 72 committed cells / 120 legacy split records / 48 field cells')
        if summary.get('status') == COMPLETE_STATUS and summary.get('error') is not None:
            missing.append('summary claims completion but retains an error')
        if summary.get('status') != COMPLETE_STATUS:
            missing.append('run_status: ' + str(summary.get('status', 'no_summary')))
        if summary.get('status') == COMPLETE_STATUS:
            for field in ('descriptive_legacy_means', 'descriptive_field_means'):
                if field not in summary:
                    missing.append('summary.' + field)
        # Independently check reported legacy aggregate means from committed unique identities.
        grouped = defaultdict(list)
        for v in legacy:
            grouped[v['group'] + '|' + v['test_split']].append(v)
        aggregates = {}
        for group, records in sorted(grouped.items()):
            metric_names = set(records[0]['mean_losses'])
            require(all(set(r['mean_losses']) == metric_names for r in records), 'Inconsistent legacy metric keys')
            aggregates[group] = {m: math.fsum(r['mean_losses'][m] for r in records) / len(records) for m in sorted(metric_names)}
            if not missing and 'descriptive_legacy_means' in summary:
                require(set(summary['descriptive_legacy_means'].get(group, {})) == metric_names and
                        all(math.isclose(summary['descriptive_legacy_means'][group][m], aggregates[group][m], rel_tol=1e-9, abs_tol=1e-11) for m in metric_names),
                        'Reported legacy aggregate differs from unique committed records')
        recomputed_fields = {}
        for group in ('C7', 'D4_corner'):
            records = [r for r in fields if r['group'] == group]
            if not records:
                continue
            means = {}
            for method in ('generator_ridge', 'generator_input_only_field'):
                means[method] = math.fsum(r['generator']['mean_losses']['unseen_composite'][method] for r in records) / len(records)
            all_kernel = []
            for record in records:
                curves = record.get('input_comparisons', {}).get('storage_curves', [])
                matches = [c for c in curves if c.get('track') == 'all_element' and c.get('full_training_support') is True]
                require(len(matches) == 1, 'Missing/duplicate full-support all-element curve')
                all_kernel.append(matches[0]['results']['all_changed']['mean_loss'])
            recomputed_fields[group] = {'cells': len(records), 'generator_unseen': means,
                                         'all_element_full_kernel': math.fsum(all_kernel) / len(all_kernel)}
            if not missing:
                reported = summary.get('descriptive_field_means', {}).get(group, {})
                require(reported.get('cells') == len(records) and all(
                    math.isclose(reported.get('generator_unseen', {}).get(m, math.inf), v, rel_tol=1e-9, abs_tol=1e-11)
                    for m, v in means.items()) and math.isclose(reported.get('all_element_full_kernel', math.inf),
                        recomputed_fields[group]['all_element_full_kernel'], rel_tol=1e-9, abs_tol=1e-11),
                    'Reported field aggregate differs from unique committed conditions')
        uncertainty = 'analysis/descriptive_uncertainty.json'
        if uncertainty in a.names:
            u = a.js(uncertainty)
            require(u.get('seed') == cfg['descriptive_uncertainty']['seed'] and
                    u.get('replicates') == cfg['descriptive_uncertainty']['replicates'] and
                    u.get('multiple_comparison_adjustment') is False, 'Uncertainty protocol metadata changed')
            require(set(u.get('legacy_contrasts', {})) == set(aggregates), 'Uncertainty domain/split panel differs from committed records')
            for group in grouped.values():
                for endpoint in ('family_orbit_prediction', 'family_orbit_context'):
                    coverage = defaultdict(set)
                    for r in group:
                        identity = (r['model']['model_id'], str(r['layer']))
                        for pair in r[endpoint]:
                            coverage[pair].add(identity)
                    expected = {(s['id'], str(l)) for s in cfg['models'] for l in s['layers']}
                    require(all(v == expected for v in coverage.values()), 'Uncertainty lacks exact unique 24-condition coverage per family/orbit')
        elif summary.get('status') == COMPLETE_STATUS:
            missing.append(uncertainty)
        warnings.append('Transport, contract and coordinate validation do not verify raw model forwards, hidden-space PCA refits, estimator refits, novelty or scientific eligibility.')
        if not numerical:
            warnings.append('NumPy checks not requested: array contents have checksum/CRC/header validation only; finite-array/PCA geometry checks are not certified; compact S3 integer mappings were checked with the standard library.')
        return {'schema_version': 1, 'study_id': cfg['study_id'], 'archive_integrity': 'passed',
                'source_contract': 'matches_frozen_v1', 'source_zip_sha256': frozen.zip_sha256,
                'source_manifest_sha256': frozen.manifest_sha256,
                'source_verification_mode': frozen.verification_mode,
                'run_recorded_status': summary.get('status', 'no_summary'), 'run_recorded_error': summary.get('error'),
                'panel_validation': 'complete' if not missing else 'incomplete_or_failed',
                'counts': {'completion_markers': len(markers), 'legacy_split_records': len(legacy), 'field_cells': len(fields)},
                'missing_or_incomplete': sorted(set(missing)), 'warnings': warnings,
                'numerical_checks': {'enabled': numerical, 'scope': 'finite arrays; PCA orthogonality/train centering/energy; compact S3 inverse mapping' if numerical else 'NPY headers; exact compact S3 integer mappings',
                                     'independent_estimator_refits': False},
                'historical_s3_reserved_bank_validation': {
                    'revision': 'exact_historical_bank_v1',
                    'validated_models': validated_s3_banks,
                    'reserved_rows_per_model': 9216,
                    'scope': 'unchanged parent development cross-block metadata; no selection or scoring rule changed'},
                'projected_exports': projected, 'independently_recomputed_legacy_means': aggregates,
                'independently_recomputed_field_means': recomputed_fields,
                'scientific_eligibility': 'not_assessed_requires_separate_scientific_audit', 'paper_eligible': False,
                'automatic_manuscript_promotion': False, 'models_loaded': False,
                'raw_hidden_activations_verified': False, 'archive_inventory': a.inventory,
                'member_sha256': {n: a.digest(n) for n in sorted(a.names)}}


def import_validated(archive, checksum, source, *, allow_incomplete=False, numerical=False,
                     validate_only=False, receipt_name=None, project=PROJECT):
    archive, checksum, source = (boundary(p, project) for p in (archive, checksum, source))
    require(archive.is_file() and checksum.is_file() and (source.is_file() or source.is_dir()), 'An explicit result ZIP, checksum sidecar and pinned frozen source are required')
    archive_hash = check_sidecar(archive, checksum)
    frozen = load_frozen(source)
    try:
        report = validate_result(archive, frozen, numerical=numerical)
    except (zipfile.BadZipFile, KeyError, TypeError, IndexError, OverflowError) as exc:
        raise ValidationError('Malformed/inconsistent result structure: ' + str(exc)) from exc
    report['input_sha256'] = archive_hash
    report['input_checksum_sha256'] = sha_file(checksum)
    report['imported'] = False
    if validate_only:
        return report
    require(report['panel_validation'] == 'complete' or allow_incomplete,
            'Incomplete/failed result retained unchanged; use --allow-incomplete to import it with explicit failure labeling, or --validate-only to inspect')
    now = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    name = receipt_name or now + '-' + archive_hash[:12]
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,95}', name) and name not in ('.', '..'), 'Unsafe receipt name')
    base = boundary(STUDY_REL / 'imports', project)
    base.mkdir(parents=True, exist_ok=True)
    target = boundary(base / name, project)
    require(not target.exists(), 'Receipt namespace already exists; overwriting is forbidden')
    # Preserve the original sealed archive plus validated, non-executable payloads.
    with tempfile.TemporaryDirectory(prefix='.import-', dir=base) as temporary:
        temporary = Path(temporary)
        shutil.copyfile(archive, temporary / 'original.results.zip')
        require(sha_file(temporary / 'original.results.zip') == archive_hash, 'Input changed during import copy')
        shutil.copyfile(checksum, temporary / 'supplied.sha256')
        require(sha_file(temporary / 'supplied.sha256') == report['input_checksum_sha256'], 'Checksum sidecar changed during import copy')
        # Materialize only the already-validated JSON/NPZ result payload. Manual
        # exclusive creates avoid ZipFile.extractall and preserve all partial data.
        raw_root = temporary / 'raw'
        raw_root.mkdir()
        with zipfile.ZipFile(temporary / 'original.results.zip') as incoming:
            for info in incoming.infolist():
                if info.is_dir():
                    continue
                rel = info.filename[len(RESULT_PREFIX):]
                safe_name(rel)
                destination = raw_root / rel
                destination.parent.mkdir(parents=True, exist_ok=True)
                h = hashlib.sha256()
                with incoming.open(info) as src, destination.open('xb') as dst:
                    for block in iter(lambda: src.read(2**20), b''):
                        h.update(block)
                        dst.write(block)
                require(h.hexdigest() == report['member_sha256'][rel], 'Result member changed during receipt materialization')
        report.update(imported=True, imported_at_utc=now, receipt_name=name,
                      receipt_project_relative_directory=str(target.relative_to(Path(project).resolve())),
                      import_status='imported_complete_pending_scientific_audit' if report['panel_validation'] == 'complete'
                      else 'imported_incomplete_or_failed_not_scientifically_eligible')
        for filename, value in [('receipt.json', report), ('input_identity.json', {'archive_sha256': archive_hash,
                'original_filename': archive.name, 'original_checksum_filename': checksum.name,
                'source_zip_sha256': frozen.zip_sha256, 'supplied_checksum_sha256': report['input_checksum_sha256']})]:
            with (temporary / filename).open('x', encoding='utf-8') as f:
                json.dump(value, f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
                f.write('\n')
        (temporary / 'REPORT.txt').write_text('Characterization-v1 private results receipt\n\n'
            + 'Import status: ' + report['import_status'] + '\nRecorded run status: ' + report['run_recorded_status']
            + '\nPanel: ' + json.dumps(report['counts'], sort_keys=True) + '\n\n'
            + 'This receipt confirms bounded file/contract/artifact checks, not a scientific pass. '
              'No experiment was launched; no manuscript was changed. The original ZIP is preserved.\n\n'
            + 'Missing/incomplete: ' + json.dumps(report['missing_or_incomplete']) + '\n'
            + '\n'.join(report['warnings']) + '\n', encoding='utf-8')
        # mkdir reserves the final namespace exclusively; rename must never be
        # allowed to replace an existing empty receipt directory. Publish the
        # receipt last so interrupted publication cannot masquerade as committed.
        target.mkdir(mode=0o700, exist_ok=False)
        for item in sorted(temporary.iterdir(), key=lambda p: (p.name == 'receipt.json', p.name)):
            destination = target / item.name
            require(not destination.exists(), 'Concurrent receipt member exists; overwriting is forbidden')
            os.rename(item, destination)
    report['receipt_project_relative_directory'] = str(target.relative_to(Path(project).resolve()))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True, help='Result ZIP inside the research project')
    parser.add_argument('--checksum', '--sha256', dest='checksum', required=True, help='Explicit SHA256 sidecar inside the project')
    parser.add_argument('--source', default=str(PROJECT / SOURCE_REL), help='Verified frozen_run directory or exact pinned execution ZIP inside project')
    parser.add_argument('--allow-incomplete', action='store_true', help='Preserve a valid incomplete/failed archive with explicit failure labeling')
    parser.add_argument('--numpy-checks', action='store_true', help='Also inspect finite arrays, PCA geometry and compact S3 mappings; never imports Torch')
    parser.add_argument('--validate-only', action='store_true', help='Read-only validation; do not create a receipt')
    parser.add_argument('--receipt-name', help='Fresh namespace under study imports/; never overwritten')
    args = parser.parse_args(argv)
    try:
        inputs = [Path(p) if Path(p).is_absolute() else Path.cwd() / p for p in (args.archive, args.checksum, args.source)]
        report = import_validated(*inputs,
                                  allow_incomplete=args.allow_incomplete, numerical=args.numpy_checks,
                                  validate_only=args.validate_only, receipt_name=args.receipt_name)
        print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))
        return 0 if report['panel_validation'] == 'complete' else 2
    except (ValidationError, OSError, ImportError, MemoryError) as exc:
        print(json.dumps({'validation_status': 'rejected', 'imported': False, 'paper_eligible': False,
                          'error': type(exc).__name__, 'message': str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
