#!/usr/bin/env python3
"""Prepared independent-orthogonality analysis. Default/status/help do not run it."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import uuid

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def read_json(path):
    def no_constant(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out: raise ValueError('Duplicate JSON key: ' + key)
            out[key] = value
        return out
    return json.loads(Path(path).read_text(), parse_constant=no_constant, object_pairs_hook=pairs)


def contained(base, name):
    rel = Path(name)
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError('Nonrelative source path')
    path = base/rel
    if path.is_symlink() or not path.resolve().is_relative_to(base.resolve()):
        raise ValueError('Escaping or symlink source path')
    return path


def publish_bytes(path, raw):
    """Atomic create-only publication: output files are never overwritten."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent/('.pending-'+uuid.uuid4().hex)
    try:
        with temporary.open('xb') as f:
            f.write(raw); f.flush(); os.fsync(f.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value):
    publish_bytes(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode())


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def source_contract():
    manifest = read_json(HERE/'source_manifest.json')
    required = {'run.py', 'core.py', 'protocol.json', 'historical_protocol.json',
                'requirements.txt', 'README.md', 'tests/test_core.py'}
    if not required <= set(manifest['files']):
        raise ValueError('Source manifest omits a required file')
    for name, expected in manifest['files'].items():
        if file_sha(contained(HERE, name)) != expected:
            raise ValueError('Prepared source differs: '+name)
    for name, expected in manifest['project_dependencies'].items():
        if file_sha(contained(PROJECT, name)) != expected:
            raise ValueError('Frozen dependency or saved endpoint differs: '+name)
    plan = read_json(HERE/'protocol.json')
    fixed = {'dimensions': [16, 8], 'planned_cells_per_dimension': 48,
             'planned_units': 96, 'context_blocks_per_cell': 128,
             'source_dimension': 16, 'include_identity': True,
             'include_fixed_points': True, 'extra_centering': False,
             'rank_relative_threshold': 1e-10,
             'matrix_normalized_frobenius_tolerance': 1e-8,
             'scalar_atol': 1e-8, 'scalar_rtol': 1e-7,
             'model_execution': False, 'tests_executed_at_preparation': False,
             'numerical_analysis_executed_at_preparation': False}
    if any(plan.get(k) != v for k, v in fixed.items()):
        raise ValueError('Prepared fixed protocol contract differs; use a new protocol')
    for name in ('state_weight', 'context_weight', 'action_weight'):
        if plan.get(name) != 'uniform': raise ValueError('Uniform weights required')
    return plan, manifest, read_json(HERE/'historical_protocol.json')


def units_from(historical, plan):
    cells = []
    for model in historical['models']:
        for layer in model['layers']:
            for domain in historical['domains']:
                cell = model['id'].replace('/', '__')+'__'+domain+'__'+str(layer)
                cells.append({'cell': cell, 'model_id': model['id'], 'layer': layer,
                              'domain': domain, 'model_revision': model['revision'],
                              'tokenizer_revision': model['tokenizer_revision']})
    if len(cells) != 48 or len({c['cell'] for c in cells}) != 48:
        raise ValueError('Declared 48-cell panel differs')
    return [{**c, 'dimension': dim, 'unit': c['cell']+'__r'+str(dim)}
            for dim in plan['dimensions'] for c in sorted(cells, key=lambda c:c['cell'])]


def read_reference_csv(path, expected_keys):
    with path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    index = {}
    for row in rows:
        key = (row['cell'], int(row['dimension']))
        if key in index: raise ValueError('Duplicate saved reference cell/dimension')
        index[key] = row
    if set(index) != expected_keys: raise ValueError('Saved reference is not the exact 96-unit panel')
    return index


def bind_to_saved(summary, reference, diagnostic, old16, core):
    """No new spectral calculation: attach authenticated same-bank saved bounds."""
    if summary.get('status') != 'complete': return
    checks = core.Checks(summary['checks'])
    try:
        if reference['canonical_status'] != 'complete':
            raise ValueError('Saved canonical reference unavailable')
        eta, orth = summary['metrics']['eta'], summary['metrics']['L_orth']
        old_eta = float(reference['eta'])
        gamma, lower, upper = (float(reference[k]) for k in ('gamma', 'L_lower', 'L_upper'))
        checks.scalar('eta_matches_saved_same_dimension', eta, old_eta)
        checks.scalar('identity_matches_saved_same_dimension', summary['metrics']['L_identity_direct'], float(reference['L_identity']))
        checks.scalar('sigma_ratio_matches_saved_same_dimension', summary['diagnostics']['sigma_min_max_ratio'], float(diagnostic['sigma_min_max_ratio']))
        checks.scalar('saved_lower_formula', lower, 2.0*(1.0-gamma))
        checks.scalar('saved_upper_formula', upper, 2.0*(1.0-gamma*gamma))
        if old16 is not None:
            checks.scalar('eta_matches_original16_archive', eta, float(old16['metrics']['eta_LS']))
            for key in ('sigma_min', 'sigma_max', 'sigma_min_max_ratio'):
                checks.scalar('moment_matches_original16_'+key, summary['diagnostics'][key], float(old16['diagnostics'][key]))
        # Orthogonal families form a larger class than exact orthogonal actions.
        # This compares class optima on ONE bank/metric, not fitted test losses.
        law_lower = max(0.0, lower-orth)  # Mathematical lower endpoint definition.
        law_upper = upper-orth
        checks.le('L_orth_le_exact_class_upper_bound', orth, upper)
        checks.le('extra_law_interval_ordered', law_lower, law_upper)
        checks.le('extra_law_upper_nonnegative', 0.0, law_upper)
        checks.scalar('saved_total_premium_lower', max(0.0, lower-old_eta), float(reference['p_minus']))
        checks.scalar('orth_plus_beta_eta_identity', eta+summary['metrics']['beta'], orth)
        summary['saved_reference'] = {'gamma': gamma, 'eta_LS': old_eta,
             'L_lower': lower, 'L_upper': upper, 'p_minus': float(reference['p_minus']),
             'p_plus': float(reference['p_plus']),
             'registration_rank_not_recomputed': True}
        summary['metrics'].update(extra_law_lower=law_lower, extra_law_upper=law_upper,
            extra_law_lower_above_tolerance=bool(law_lower > core.scalar_tolerance(law_lower)),
            extra_law_comparison_tolerance=core.scalar_tolerance(law_lower))
    except (core.ComparatorError, ValueError, TypeError, KeyError) as exc:
        summary.update(status='not_evaluable_saved_reference_or_bound', reason=f'{type(exc).__name__}: {exc}')


def save_unit(root, unit, summary, arrays, save_operators, np):
    folder = root/'units'/unit['unit']
    write_json(folder/'summary.json', summary)
    names = ['summary.json']
    if save_operators and arrays:
        buffer = io.BytesIO()
        # Preserve finite diagnostic arrays even when a later check fails. No pickle.
        keep = {k:v for k,v in arrays.items() if k in {'Sigma','sigma_eigenvalues','whitening','M','Q','cross_singular_values','actions'}}
        if keep:
            np.savez_compressed(buffer, **keep)
            publish_bytes(folder/'diagnostics.npz', buffer.getvalue()); names.append('diagnostics.npz')
    write_json(folder/'receipt.json', {'unit':unit['unit'], 'status':summary['status'],
        'source_binding_sha256':summary['source_binding_sha256'],
        'files':{name:file_sha(folder/name) for name in names}, 'terminal':True})


def export_summary(root, units, records, inputs_unchanged):
    byid = {row['unit']:row for row in records}
    columns = ['unit','cell','model_id','layer','domain','dimension','status','reason',
               'eta','eta_svd','L_orth','L_orth_svd','beta','beta_svd','beta_eta_floor',
               'extra_law_lower','extra_law_upper','extra_law_lower_above_tolerance',
               'sigma_min','sigma_max','sigma_min_max_ratio']
    rows = []
    for u in units:
        record = byid[u['unit']]
        row = {k:u.get(k) for k in columns}
        row.update(status=record['status'],reason=record.get('reason'))
        for k in columns:
            if k in record.get('metrics',{}): row[k]=record['metrics'][k]
            elif k in record.get('diagnostics',{}): row[k]=record['diagnostics'][k]
        rows.append(row)
    stream = io.StringIO(newline=''); writer=csv.DictWriter(stream,fieldnames=columns)
    writer.writeheader(); writer.writerows(rows); publish_bytes(root/'cell_summary.csv',stream.getvalue().encode())
    aggregates=[]
    for dim in (16,8):
        for domain in ('C7','D4'):
            subset=[r for r in rows if r['dimension']==dim and r['domain']==domain]
            good=[r for r in subset if r['status']=='complete']
            complete=len(subset)==24 and len(good)==24 and inputs_unchanged
            item={'dimension':dim,'domain':domain,'planned':24,'evaluable':len(good),
                  'status':'complete_descriptive' if complete else 'incomplete_no_mean',
                  'failures':[r['unit'] for r in subset if r['status']!='complete'],
                  'means':None,'positive_extra_law_lower_count':None}
            if complete:
                keys=('eta','L_orth','beta','beta_svd','extra_law_lower','extra_law_upper')
                item['means']={k:math.fsum(float(r[k]) for r in good)/24 for k in keys}
                item['positive_extra_law_lower_count']=sum(bool(r['extra_law_lower_above_tolerance']) for r in good)
            aggregates.append(item)
    coverage={'protocol_id':'orthogonal_comparator_v1','planned_units':96,
              'recorded_units':len(rows),'status_counts':dict(Counter(r['status'] for r in rows)),
              'inputs_unchanged_after_analysis':inputs_unchanged,'domain_summaries':aggregates,
              'status':'complete_descriptive' if all(r['status']=='complete' for r in rows) and inputs_unchanged else 'complete_with_failures',
              'review_before_manuscript_integration':True,'automatic_scientific_promotion':False,
              'same_metric_within_dimension_only':True,'exact_Lstar_measured':False,
              'original_primary16_results_replaced':False}
    write_json(root/'coverage.json',coverage)
    return coverage


def execute(args):
    if not args.execute_offline_analysis:
        raise ValueError('Run requires explicit --execute-offline-analysis; preparation does not authorize computation')
    if args.resume:
        raise ValueError('Resume is intentionally unsupported. Preserve existing outputs and use a fresh directory.')
    plan, manifest, historical = source_contract()
    root=Path(args.output).resolve(); repo=Path(args.repo).resolve(strict=True)
    if root.exists() or root.is_symlink(): raise ValueError('Output must not exist; never overwrite/reuse earlier outputs')
    if root==PROJECT or root.is_relative_to(PROJECT) or root==repo or root.is_relative_to(repo):
        raise ValueError('Use an output outside the source project and immutable data root')
    if sys.version_info[:2] != (3,11): raise ValueError('Recorded runtime requires Python 3.11')
    import numpy as np  # Only imported after the explicit execution gate.
    if np.__version__ != '2.2.6': raise ValueError('Recorded runtime requires NumPy 2.2.6')
    core=load_module('_bsg_orth_core_v1',HERE/'core.py')
    reader=load_module('_bsg_orth_reader_v1',PROJECT/'studies/spectral_dimension_sensitivity_v1/archive_inputs.py')
    kernel=load_module('_bsg_orth_group_spec_v1',PROJECT/'studies/spectral_dimension_sensitivity_v1/core/_vendor/gamma_r_frozen.py')
    if (core.RANK_RATIO,core.MATRIX_TOL,core.SCALAR_ATOL,core.SCALAR_RTOL)!=(kernel.RANK_RATIO,kernel.MATRIX_TOL,kernel.SCALAR_ATOL,kernel.SCALAR_RTOL):
        raise ValueError('Comparator/frozen-kernel tolerances differ')
    units=units_from(historical,plan)
    expected={(u['cell'],u['dimension']) for u in units}
    references=read_reference_csv(PROJECT/plan['reference_cells'],expected)
    diagnostics=read_reference_csv(PROJECT/plan['reference_diagnostics'],expected)
    root.mkdir(parents=True,exist_ok=False)
    write_json(root/'plan.json',{'protocol':plan,'units':units})
    write_json(root/'runtime.json',{'python':sys.version,'numpy':np.__version__,
        'platform':platform.platform(),'model_execution':False})
    for name in manifest['files']:
        publish_bytes(root/'source_snapshot'/name,(HERE/name).read_bytes())
    publish_bytes(root/'source_snapshot/source_manifest.json',(HERE/'source_manifest.json').read_bytes())
    for name in manifest['project_dependencies']:
        publish_bytes(root/'source_snapshot/project_dependencies'/name,(PROJECT/name).read_bytes())
    base_binding={'source_manifest_sha256':file_sha(HERE/'source_manifest.json'),
                  'protocol_sha256':file_sha(HERE/'protocol.json'),
                  'project_dependency_sha256':manifest['project_dependencies'],
                  'archive_sha256':{k:v[1] for k,v in reader.ARCHIVES.items()}}
    binding_sha=hashlib.sha256(canonical(base_binding)).hexdigest()
    write_json(root/'binding.json',base_binding)
    records=[]; recorded=set(); input_stable=False
    try:
        with reader.Study2Inputs(repo,historical) as inputs:
            preflight=inputs.preflight()
            write_json(root/'input_preflight.json',preflight)
            for u in units:
                meta={'cell_binding':inputs.by_cell[u['cell']], 'binding':base_binding,
                      'reference_row_sha256':hashlib.sha256(canonical(references[u['cell'],u['dimension']])).hexdigest(),
                      'diagnostic_row_sha256':hashlib.sha256(canonical(diagnostics[u['cell'],u['dimension']])).hexdigest()}
                arrays={}
                numerical_started=False
                try:
                    bank=inputs.load_test_bank(u['cell'])
                    expected_shape=(128,7 if u['domain']=='C7' else 4,16)
                    if bank.shape!=expected_shape: raise ValueError('Authenticated original bank shape differs')
                    Z=np.array(bank[...,:u['dimension']],dtype=np.float64,order='C',copy=True)
                    spec=kernel.group_spec(u['domain']); kernel.validate_group(spec)
                    numerical_started=True
                    summary,arrays=core.analyze_bank(Z,spec['actions'],expected_action_count=spec['m'],identity_index=spec['identity'])
                    bind_to_saved(summary,references[u['cell'],u['dimension']],diagnostics[u['cell'],u['dimension']],
                                  inputs.semantic_baselines[u['cell']] if u['dimension']==16 else None,core)
                except Exception as exc:
                    summary={'status':'not_evaluable_unit_exception','reason':f'{type(exc).__name__}: {exc}'}
                summary.update(u);summary.update(source_binding_sha256=binding_sha,input_identity=meta,
                    partition='test_same_bank_oracle_comparison',source_coordinate_dimension=16,
                    column_rule='first_r_columns_original_training_PCA',new_model_execution=False,
                    original_results_modified=False,
                    numerical_analysis_performed_for_unit=numerical_started)
                save_unit(root,u,summary,arrays,args.save_operators,np)
                records.append(summary);recorded.add(u['unit'])
                print('UNIT',len(records),'/96',u['unit'],summary['status'],flush=True)
            after={k:reader.file_sha(repo/v[0]) for k,v in reader.ARCHIVES.items()}
            source_contract()  # Recheck source/dependency identity after the run.
            input_stable=all(after[k]==v[1] for k,v in reader.ARCHIVES.items())
            write_json(root/'input_preservation.json',{'before':base_binding['archive_sha256'],'after':after,'unchanged':input_stable})
    except BaseException as exc:
        write_json(root/'failure.json',{'type':type(exc).__name__,'reason':str(exc),'completed_receipts':len(records)})
        for u in units:
            if u['unit'] in recorded:continue
            summary={**u,'status':'blocked_or_interrupted','reason':type(exc).__name__+': '+str(exc),
                     'source_binding_sha256':binding_sha,'numerical_analysis_performed_for_unit':False}
            # Do not overwrite a partially created unit; its raw files remain evidence.
            folder=root/'units'/u['unit']
            if folder.exists():
                write_json(root/'failure_units'/(u['unit']+'.json'),summary)
            else:
                save_unit(root,u,summary,{},False,np)
            records.append(summary)
    coverage=export_summary(root,units,records,input_stable)
    write_json(root/'output_manifest.json',{'files':{str(p.relative_to(root)):file_sha(p) for p in sorted(root.rglob('*')) if p.is_file()},
                                         'complete':coverage['status']=='complete_descriptive'})
    print(json.dumps(coverage,indent=2,sort_keys=True),flush=True)
    return 0 if coverage['status']=='complete_descriptive' else 2


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command')
    sub.add_parser('status',help='Read prepared source/protocol metadata only; no archives or numeric libraries')
    run=sub.add_parser('run',help='Explicit future same-bank O(r) analysis; requires separate authorization')
    run.add_argument('--repo',required=True,help='Immutable data root containing the two original incoming archive paths')
    run.add_argument('--output',required=True,help='Fresh directory outside source and data roots')
    run.add_argument('--execute-offline-analysis',action='store_true')
    run.add_argument('--save-operators',action='store_true',help='Retain Q/M/whitening diagnostic arrays as additional evidence')
    run.add_argument('--resume',action='store_true',help='Intentionally rejected; existing runs must be preserved')
    args=parser.parse_args(argv)
    if args.command in (None,'status'):
        plan,manifest,_=source_contract()
        print(json.dumps({'status':'PREPARED_NOT_RUN','protocol':plan['protocol_id'],
              'planned_units':plan['planned_units'],'source_verified':True,
              'tests_executed':False,'numerical_analysis_executed':False,
              'archives_opened':False,'numeric_libraries_imported':False},indent=2))
        return 0
    try:return execute(args)
    except (ValueError, OSError, KeyError) as exc:
        print('STOP:',type(exc).__name__+': '+str(exc),file=sys.stderr);return 2

if __name__=='__main__':raise SystemExit(main())
