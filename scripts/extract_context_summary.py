#!/usr/bin/env python3
"""Audit stored context results. Never imports study code, fits, scores, or bootstraps.

Usage: python3 extract_context_summary.py RESULTS_ZIP OUTPUT_DIRECTORY
Only recorded losses are arithmetically averaged. Recorded bootstrap percentiles
are copied verbatim. SHA-256 verification and NPZ header inspection do not run
model code or recompute an experimental endpoint.
"""
import argparse
import ast
import collections
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import struct
import zipfile

PREFIX = 'context-b73c5b86cc8f4257961c9d2b1d91b121/'
SOURCE = 'source_snapshot/studies/context_generalization_v1/'
ARMS = ('identity', 'coordinate', 'difference', 'coordinate_nativeval')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode()


def npz_shapes(raw):
    result = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        for name in z.namelist():
            with z.open(name) as f:
                assert f.read(6) == b'\x93NUMPY'
                version = tuple(f.read(2))
                n = struct.unpack('<H' if version[0] == 1 else '<I',
                                  f.read(2 if version[0] == 1 else 4))[0]
                header = ast.literal_eval(f.read(n).decode('latin1'))
                result[name.removesuffix('.npy')] = list(header['shape'])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive', type=Path)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with args.archive.open('rb') as f:
        digest = hashlib.sha256()
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
        outer_digest = digest.hexdigest()
    evidence, checks = {}, {}
    with zipfile.ZipFile(args.archive) as z:
        names = set(z.namelist())
        assert len(names) == len(z.namelist()), 'duplicate ZIP names'

        def read(name):
            raw = z.read(name)
            evidence[name] = {'sha256': sha(raw), 'bytes': len(raw)}
            return raw

        def js(name):
            return json.loads(read(name))

        def local(name):
            return js(PREFIX + name)

        def verify_map(mapping, prefix=''):
            for path, digest in mapping.items():
                assert sha(read(prefix + path)) == digest, prefix + path

        exported = js('batch_export_manifest.json')
        identity = js('identity.json')
        state = js('state.json')
        # Verify every exported context member, including frozen development,
        # predictor, PCA, projected coordinates and metadata; no arrays are used.
        subset = {k: v for k, v in exported['files'].items()
                  if k.startswith(PREFIX) or k.startswith(SOURCE)}
        verify_map(subset)
        checks['export_manifest_context_and_study_members_verified'] = len(subset)
        for name in ['launcher.py', 'environment.json']:
            read(name)
        for n in sorted(names):
            if n.startswith('logs/context') or n.startswith('logs/review') or n.startswith('failures/context'):
                read(n)
        manifest = js(SOURCE + 'source_manifest.json')
        verify_map(manifest['files'], SOURCE)
        verify_map(manifest['dependencies'], 'source_snapshot/')
        checks['source_manifest_files_and_dependencies_verified'] = len(manifest['files']) + len(manifest['dependencies'])
        protocol = local('protocol.json')
        inputs = local('input_manifest.json')
        assert read(PREFIX + 'input_manifest.sha256').decode().strip() == sha(read(PREFIX + 'input_manifest.json'))
        assert protocol == inputs['protocol'] == js(SOURCE + 'protocol.json')
        assert inputs['source_manifest_sha256'] == sha(read(SOURCE + 'source_manifest.json'))
        verify_map(inputs['prepared_files'], PREFIX)
        checks['prepared_input_files_verified'] = len(inputs['prepared_files'])
        assert inputs['parent_archive_sha256'] == identity['archive_sha256'] == exported['source_archive_sha256']
        bank = local('bank.json')['rows']
        assert sha(canonical(bank)) == inputs['bank_sha256']
        assert len(bank) == len({r['row_id'] for r in bank}) == len({r['text'] for r in bank}) == 1408
        states = {'C7': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                  'D4': ['NW', 'NE', 'SE', 'SW']}
        families = [f'new_context_{i:03d}' for i in range(64)]
        for d, tokens in states.items():
            rows = [r for r in bank if r['domain'] == d]
            actual = {(r['family_id'], r['paraphrase_id'], r['state_id']) for r in rows}
            expected = {(f, para, s) for f in families for para in (0, 1) for s in range(len(tokens))}
            assert actual == expected and len(rows) == len(expected)
            for r in rows:
                a, b = r['source_span']
                assert r['target'] == tokens[r['state_id']] == r['text'][a:b]
                assert r['split'] == 'test' and r['text'].count(r['target']) == 1
                assert sum(r['text'].count(t) for t in tokens) == 1
        checks['new_bank_unique_prompts_targets_and_complete_blocks'] = True
        review = local('human_review.json')
        assert review['bank_sha256'] == inputs['bank_sha256']
        assert review['source_manifest_sha256'] == inputs['source_manifest_sha256']
        assert review['approval_before_model_execution'] is True
        fits = local('fits_manifest.json')
        fit_binding = sha(canonical({'input': inputs, 'protocol': protocol}))
        assert fits['binding'] == fit_binding
        assert fits['review_sha256'] == sha(read(PREFIX + 'human_review.json'))
        assert fits['new_test_outputs_seen'] is False
        downstream_binding = sha(canonical({'inputs': fit_binding, 'fits': fits}))
        extract = local('extraction_contract.json')
        assert extract['binding'] == downstream_binding
        assert extract['fit_manifest_sha256'] == sha(read(PREFIX + 'fits_manifest.json'))
        assert local('extraction_complete.json')['binding'] == downstream_binding
        models = js('source_snapshot/studies/spectral_finite_bank_v1/protocol.json')['models']
        expected_cells = {m['id'].replace('/', '__') + '__' + d + '__' + str(layer)
                          for m in models for d in states for layer in m['layers']}
        assert len(expected_cells) == 48
        assert {c['cell'] for c in inputs['cells']} == {c['cell'] for c in fits['cells']} == expected_cells
        cells, selection_counts = [], collections.defaultdict(collections.Counter)
        for c in inputs['cells']:
            cell, d = c['cell'], c['domain']
            assert c['status'] == 'prepared'
            model_spec = next(m for m in models if m['id'] == c['model_id'])
            assert c['revision'] == model_spec['revision']
            assert c['tokenizer_revision'] == model_spec['tokenizer_revision']
            k = len(states[d])
            assert npz_shapes(read(PREFIX + c['paths']['development.npz'])) == {
                'train': [48, 2, k, 16], 'validation': [16, 2, k, 16]}
            fit = local('fits/' + cell + '/complete.json')
            assert fit['binding'] == fit_binding and fit in fits['cells']
            verify_map(fit['artifacts'], PREFIX)
            metadata = local('fits/' + cell + '/fit.json')
            assert metadata['test_used_for_fit_or_selection'] is False
            assert metadata['training_difference_energy'] > 0
            assert len(metadata['selection_ledger']) == (6 if d == 'C7' else 7) * 3
            for entry in metadata['selection_ledger']:
                arm = entry.get('arm', entry['training_objective'])
                key = 'mse' if arm == 'secondary_coordinate_nativeval' else 'context_mse'
                assert entry['selection_objective'] == ('coordinate_validation_mse' if key == 'mse' else 'context_difference_validation_mse')
                candidates = entry['validation_candidates']
                assert [v['lambda'] for v in candidates] == protocol['lambda_grid']
                best = min(range(len(candidates)), key=lambda i: candidates[i][key])
                assert best == entry['selected_grid_index']
                assert candidates[best]['lambda'] == entry['selected_lambda']
                selection_counts[arm][str(entry['selected_lambda'])] += 1
            coords = local('new_coordinates/' + cell + '/complete.json')
            assert coords['binding'] == downstream_binding
            verify_map(coords['artifacts'], PREFIX)
            assert coords['saved_pca_sha256'] == sha(read(PREFIX + c['paths']['pca.npz']))
            assert npz_shapes(read(PREFIX + c['paths']['pca.npz'])) == {
                'center': [model_spec['hidden_dimension']], 'basis': [model_spec['hidden_dimension'], 16]}
            assert local('new_coordinates/' + cell + '/rows.json')['rows'] == [r for r in bank if r['domain'] == d]
            assert npz_shapes(read(PREFIX + 'new_coordinates/' + cell + '/coordinates.npz')) == {'z': [64 * 2 * k, 16]}
            model_key = c['model_id'].replace('/', '__')
            extraction_manifest_path = PREFIX + 'extraction/' + model_key + '/manifest.json'
            ex = js(extraction_manifest_path)
            assert coords['cache_manifest_sha256'] == sha(read(extraction_manifest_path))
            assert ex['model'] == local(c['paths']['old_runtime.json'])
            assert ex['model']['model_id'] == model_spec['id']
            assert ex['model']['model_revision'] == model_spec['revision']
            assert ex['model']['tokenizer_revision'] == model_spec['tokenizer_revision']
            assert ex['layers'] == model_spec['layers']
            assert ex['n_rows'] == 1408 and ex['dataset_sha256'] == inputs['bank_sha256']
            assert ex['status'] == 'complete' and ex['readout'] == 'span_mean'
            result = local('evaluations/' + cell + '/result.json')
            assert result['status'] == 'complete'
            assert [r['family_id'] for r in result['per_family']] == families
            for arm in ARMS:
                assert math.isclose(statistics.mean(r[arm] for r in result['per_family']), result['means'][arm], abs_tol=1e-14)
            for r in result['per_family']:
                assert math.isclose(r['difference'] - r['coordinate'], r['difference_minus_coordinate'], abs_tol=1e-14)
            cells.append(result)
        checks['fit_coordinate_receipts_and_runtime_identities_verified_cells'] = len(cells)
        checks['old_train_validation_array_shapes_only_verified_cells'] = len(cells)
        checks['validation_ledger_candidate_minima_verified'] = sum(sum(v.values()) for v in selection_counts.values())
        analysis = local('analysis_complete.json')
        assert analysis['binding'] == downstream_binding and analysis['status'] == 'complete'
        verify_map(analysis['artifacts'], PREFIX)
        summary = local('summary.json')
        assert summary['status'] == 'complete' and summary['completed_cells'] == 48
        assert summary['human_review_sha256'] == sha(read(PREFIX + 'human_review.json'))
        csv_rows = list(csv.DictReader(io.StringIO(read(PREFIX + 'cell_summary.csv').decode())))
        assert len(csv_rows) == 48
        for row in csv_rows:
            c = next(c for c in cells if c['cell'] == row['cell'])
            for arm in ARMS:
                assert float(row[arm]) == c['means'][arm]
        checks['recorded_per_family_cell_means_csv_and_summary_consistent'] = True
        # Historical stage receipts can include mutable state.json; validate their
        # immutable core evidence and report stale snapshots separately.
        receipt_comparison = {}
        for stage in ('context_prepare', 'context_fit', 'context_extract', 'context_analyze'):
            path = state['stages'][stage]['receipt']
            receipt = js(path)
            assert sha(read(path)) == state['stages'][stage]['receipt_sha256']
            present = {n: h for n, h in receipt['files'].items() if PREFIX + n in names}
            changed = [n for n, h in present.items() if sha(read(PREFIX + n)) != h]
            receipt_comparison[stage] = {'finished': receipt['finished'], 'recorded_files': len(receipt['files']),
                                         'present_files': len(present), 'later_changed_snapshot_files': changed}
        aggregate = {}
        family_vectors = {}
        for d in ('C7', 'D4', 'primary_equal_domain_panel'):
            selected = [c for c in cells if d == 'primary_equal_domain_panel' or c['domain'] == d]
            means = {a: statistics.mean(c['means'][a] for c in selected) for a in ARMS}
            fv = [statistics.mean(c['per_family'][i]['difference_minus_coordinate'] for c in selected) for i in range(64)]
            family_vectors[d] = fv
            assert math.isclose(statistics.mean(fv), summary['contrasts'][d]['mean_difference_minus_coordinate'], abs_tol=1e-14)
            aggregate[d] = {'n_cells': len(selected), 'means': means, 'recorded_contrast': summary['contrasts'][d],
                'difference_lower_than_coordinate_cells': sum(c['means']['difference'] < c['means']['coordinate'] for c in selected),
                'difference_lower_than_identity_cells': sum(c['means']['difference'] < c['means']['identity'] for c in selected),
                'coordinate_lower_than_identity_cells': sum(c['means']['coordinate'] < c['means']['identity'] for c in selected),
                'negative_family_contrasts': sum(v < 0 for v in fv), 'family_contrast_range': [min(fv), max(fv)],
                'difference_minus_identity': means['difference'] - means['identity'],
                'coordinate_minus_identity': means['coordinate'] - means['identity'],
                'difference_minus_coordinate_nativeval': means['difference'] - means['coordinate_nativeval']}
        by_model = []
        for model in models:
            item = {'model_id': model['id'], 'layers': model['layers']}
            for d in ('C7', 'D4'):
                selected = [c for c in cells if c['model_id'] == model['id'] and c['domain'] == d]
                item[d] = statistics.mean(c['means']['difference'] - c['means']['coordinate'] for c in selected)
            item['equal_domain_mean'] = (item['C7'] + item['D4']) / 2
            by_model.append(item)
        metrics = {'schema_version': 1, 'source_archive_basename': args.archive.name, 'source_archive_sha256': outer_digest,
            'context_archive_prefix': PREFIX.removesuffix('/'), 'scope': 'descriptive audit of recorded results; no experiment, refit, evaluation, or bootstrap rerun',
            'statistical_unit': '64 authored paired families shared across fixed domains/models/layers',
            'primary_effect_direction': 'negative favors difference fitting', 'bootstrap': summary['bootstrap_draws'],
            'bootstrap_seed': summary['bootstrap_seed'], 'bootstrap_scope': summary['scope'],
            'aggregate': aggregate, 'descriptive_model_contrasts': by_model,
            'selected_penalty_counts': {k: dict(v) for k, v in selection_counts.items()},
            'verification': checks, 'historical_stage_receipt_comparison': receipt_comparison,
            'provenance_limits': ['Operator review is an unauthenticated attestation, not independent authorship.',
                'Old prompt exact disjointness is recorded and source-audited; original old prompt corpus is outside this results archive.',
                'No population inference, equivalence test, or general superiority conclusion.',
                'New factual wording changes as well as outer context.']}
        (args.output / 'context_metrics.json').write_text(json.dumps(metrics, indent=2, sort_keys=True) + '\n')
        with (args.output / 'context_family_contrasts.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['family_id', *family_vectors]); w.writeheader()
            for i, family in enumerate(families):
                w.writerow({'family_id': family, **{k: v[i] for k, v in family_vectors.items()}})
        with (args.output / 'context_cell_metrics.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['cell', 'model_id', 'domain', 'layer', *ARMS, 'difference_minus_coordinate'])
            w.writeheader()
            for c in sorted(cells, key=lambda c: c['cell']):
                w.writerow({k: c[k] for k in ('cell', 'model_id', 'domain', 'layer')} | c['means'] |
                           {'difference_minus_coordinate': c['means']['difference'] - c['means']['coordinate']})
    evidence_record = {'source_archive_basename': args.archive.name, 'source_archive_sha256': outer_digest,
                       'extraction_script_sha256': sha(Path(__file__).read_bytes()), 'archive_members': evidence}
    (args.output / 'context_evidence.json').write_text(json.dumps(evidence_record, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': 'verified', 'archive_sha256': outer_digest, 'evidence_members': len(evidence),
                      'checks': checks, 'output_files': ['context_metrics.json', 'context_evidence.json',
                      'context_family_contrasts.csv', 'context_cell_metrics.csv']}, indent=2))


if __name__ == '__main__':
    main()
