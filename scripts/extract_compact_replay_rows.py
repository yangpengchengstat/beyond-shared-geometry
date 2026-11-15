#!/usr/bin/env python3
"""Read-only extraction of reviewed compact per-action evidence; no model or fitting.

The all-action endpoint includes fitted generators. The unseen-only endpoint is
computed separately by excluding the recorded fitting_actions and identity.
The original result archive and previously curated aggregates remain unchanged.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import zipfile

ARCHIVE_SHA256 = '6a4c38237bf1368d4c77c97504f666d0d933bfebed9417d173fbf7441c0fecef'
PREFIX = 'parent-from-zero-v1/compact_context/checkpoints/'
IDENTITIES = {'C7': '0', 'D4_corner': '[0,0]'}
ACTIONS = {'C7': {str(i) for i in range(1, 7)},
           'D4_corner': {json.dumps([i, j], separators=(',', ':')) for i in range(4) for j in range(2)} - {'[0,0]'}}
PRIMITIVES = {'C7': {'1'}, 'D4_corner': {'[1,0]', '[0,1]'}}
METHODS = {'ridge', 'affine', 'decoded_centroid'} | {
    f'{method}_budget_{budget}' for method in ('limited_support', 'rff', 'nystrom')
    for budget in ('0p5', '1', '2', '4', '8')}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    # Original reproduction receipt uses default ensure_ascii=True.
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def close(actual, expected, label):
    if not math.isfinite(actual) or not math.isfinite(expected) or not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError('Archived aggregate disagrees with per-action values: ' + label)


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def extract(archive, output):
    archive, output = Path(archive), Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError('Use a fresh, nonexistent output directory')
    digest = sha(archive)
    if digest != ARCHIVE_SHA256:
        raise ValueError('Archive SHA256 differs from reviewed immutable evidence')
    print('ARCHIVE_SHA256_VERIFIED', digest, flush=True)
    action_rows, endpoint_rows, inputs = [], [], []
    jobs = []
    with zipfile.ZipFile(archive) as zf:
        members = [info for info in zf.infolist() if info.filename.startswith(PREFIX) and info.filename.endswith('.json')]
        if len(members) != 240 or len({m.filename for m in members}) != 240:
            raise ValueError('Expected exactly 240 unique compact/context checkpoints')
        for info in sorted(members, key=lambda x: x.filename):
            member = info.filename
            if PurePosixPath(member).is_absolute() or '..' in PurePosixPath(member).parts or '\\' in member or info.flag_bits & 1:
                raise ValueError('Unsafe checkpoint member')
            raw = zf.read(info)  # zipfile also validates the member CRC.
            payload = json.loads(raw)
            expected = payload.pop('payload_sha256')
            if hashlib.sha256(canonical(payload)).hexdigest() != expected:
                raise ValueError('Checkpoint payload identity mismatch: ' + member)
            job = payload['job']; jobs.append(job)
            origin = {'archive_member': member, 'archive_member_sha256': hashlib.sha256(raw).hexdigest(),
                      'payload_sha256': expected}
            inputs.append(origin)
            if job['component'] != 'compact':
                continue
            domain = job['domain']
            if domain not in IDENTITIES or set(payload['result']['tracks']) != {'all_element', 'generator_only'}:
                raise ValueError('Unexpected compact domain/tracks')
            result = payload['result']
            if result['paper_eligible'] is not False or result['confirmatory'] is not False:
                raise ValueError('Original descriptive boundary differs')
            for track, track_record in result['tracks'].items():
                fitting = set(track_record['fitting_actions'])
                required = ACTIONS[domain] if track == 'all_element' else PRIMITIVES[domain]
                if fitting != required or set(track_record['selection_actions']) != required:
                    raise ValueError('Fitting/selection action contract mismatch')
                models = track_record['models']
                if len(models) != len(METHODS) or {m['name'] for m in models} != METHODS:
                    raise ValueError('Missing/duplicate compact predictor')
                for model in models:
                    if set(model['evaluation']) != {'test'}:
                        raise ValueError('Unexpected evaluation split')
                    e = model['evaluation']['test']; actions = e['actions']
                    if set(actions) != ACTIONS[domain] or IDENTITIES[domain] in actions:
                        raise ValueError('Incomplete nonidentity action panel')
                    base = {**job, 'track': track, 'method': model['name'], 'split': 'test',
                            'fitting_actions': sorted(fitting), 'selection_actions': sorted(required),
                            'identity_action': IDENTITIES[domain], 'normalization': result['normalization'],
                            'training_energy': result['training_energy'], **origin}
                    means, unseen, pooled_sum, count = [], [], 0.0, 0
                    for action, summary in sorted(actions.items()):
                        value = summary['mean_nmse']; n = summary['n']
                        if not isinstance(value, (int, float)) or not math.isfinite(value) or n <= 0 or summary.get('invalid', 0):
                            raise ValueError('Non-finite/invalid/missing action result')
                        is_fitted = action in fitting
                        if bool(summary['is_primitive']) != (action in PRIMITIVES[domain]):
                            raise ValueError('Primitive marker mismatch')
                        means.append(value); pooled_sum += n * value; count += n
                        if not is_fitted and action != IDENTITIES[domain]:
                            unseen.append(value)
                        action_rows.append({**base, 'action': action, 'fitted_action': is_fitted,
                                            'unseen_generator_composite': track == 'generator_only' and not is_fitted,
                                            'action_summary': {k: v for k, v in summary.items() if k != 'family_orbit'},
                                            'family_orbit_records': len(summary.get('family_orbit', []))})
                    all_mean = statistics.fmean(means)
                    close(all_mean, e['equal_action_mean_nmse'], member + '/all_action')
                    close(pooled_sum / count, e['pooled_changed_pair_mean_nmse'], member + '/pooled')
                    row = {**base, 'nonidentity_action_count': len(means),
                           'all_nonidentity_equal_action_nmse': all_mean,
                           'pooled_changed_pair_mean_nmse': pooled_sum / count,
                           'changed_pair_count': count,
                           'generator_unseen_action_count': len(unseen) if track == 'generator_only' else None,
                           'generator_unseen_equal_action_nmse': None,
                           'original_equal_action_mean_nmse': e['equal_action_mean_nmse']}
                    if track == 'generator_only':
                        if len(unseen) != 5:
                            raise ValueError('Expected five excluded operator elements per domain')
                        unseen_mean = statistics.fmean(unseen)
                        close(unseen_mean, e['unseen_composite_equal_action_mean_nmse'], member + '/unseen')
                        row['generator_unseen_equal_action_nmse'] = unseen_mean
                        row['original_unseen_composite_equal_action_mean_nmse'] = e['unseen_composite_equal_action_mean_nmse']
                    else:
                        row['original_nonprimitive_direct_equal_action_mean_nmse'] = e['nonprimitive_direct_equal_action_mean_nmse']
                    endpoint_rows.append(row)
    job_counts = Counter((j['component'], j['domain'], j['registration']) for j in jobs)
    expected_jobs = {(c, d, r): 24 for d in ('C7', 'D4_corner') for c in ('compact', 'context')
                     for r in ('semantic', 'consistent_swap01')}
    expected_jobs.update({('context', 'S3_role', r): 24 for r in ('semantic', 'consistent_swap01')})
    if dict(job_counts) != expected_jobs or len(endpoint_rows) != 3456 or len(action_rows) != 22464:
        raise ValueError('Incomplete result panel')
    if len({(r['cell'], r['registration'], r['track'], r['method'], r['split'], r['action']) for r in action_rows}) != len(action_rows):
        raise ValueError('Duplicate action record')
    bins = defaultdict(list)
    for row in endpoint_rows:
        if row['track'] == 'generator_only':
            bins[(row['domain'], row['registration'], row['method'])].append(row)
    summary = []
    for key, rows in sorted(bins.items()):
        if len(rows) != 24 or len({r['cell'] for r in rows}) != 24:
            raise ValueError('Incomplete fixed-panel mean')
        summary.append({'domain': key[0], 'registration': key[1], 'method': key[2], 'cells': 24,
                        'unseen_actions_per_cell': 5,
                        'unseen_equal_action_mean_nmse': statistics.fmean(r['generator_unseen_equal_action_nmse'] for r in rows),
                        'all_nonidentity_equal_action_mean_nmse': statistics.fmean(r['all_nonidentity_equal_action_nmse'] for r in rows),
                        'scope': 'Descriptive equal fixed-cell means; original test inspected; no new experiment or significance decision'})
    output.mkdir(parents=True, exist_ok=False)
    files = [('compact_action_rows.json', action_rows), ('compact_endpoint_rows.json', endpoint_rows),
             ('generator_unseen_summary.json', summary)]
    for name, value in files:
        write(output / name, value)
    manifest = {'schema_version': 1, 'archive_sha256': digest, 'archive_members_verified': len(members),
                'compact_jobs': sum(j['component'] == 'compact' for j in jobs),
                'action_rows': len(action_rows), 'endpoint_rows': len(endpoint_rows),
                'script': 'scripts/extract_compact_replay_rows.py', 'script_sha256': sha(__file__),
                'extraction_only': True, 'model_execution': False, 'new_statistical_significance_decisions': False,
                'original_flags_changed': False, 'independent_confirmation': False,
                'endpoint_note': 'All nonidentity actions includes fitted generators. Unseen-only excludes fitting_actions and identity under generator_only.',
                'normalization': 'Shared construction coordinate energy within each cell; equal action mean before equal fixed-cell mean.',
                'family_orbit_note': 'Per-action scalar aggregates retained. Detailed family/orbit records remain in the source ZIP at the recorded member.',
                'inputs': inputs, 'outputs': {name: sha(output / name) for name, _ in files}}
    write(output / 'extraction_manifest.json', manifest)
    return {k: manifest[k] for k in ('archive_members_verified', 'compact_jobs', 'action_rows', 'endpoint_rows', 'extraction_only')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Fresh output directory; existing directories are refused.')
    args = parser.parse_args()
    print(json.dumps(extract(args.archive, args.output), indent=2))

if __name__ == '__main__':
    main()
