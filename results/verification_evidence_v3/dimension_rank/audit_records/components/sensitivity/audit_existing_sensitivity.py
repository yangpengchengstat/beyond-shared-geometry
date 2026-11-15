#!/usr/bin/env python3
"""Read retained FULL results and summarize existing, exactly matched dimensions.

No model, PCA, fit, bootstrap, dimensional search, or input mutation is performed.
The output retains original and replay runs separately and includes validation and
test banks separately.  The q=16 comparator is restricted to the same two depths,
two splits, models, methods, control seeds and template IDs as q=8/32/64.
"""
from __future__ import annotations
import argparse, collections, csv, hashlib, io, json, math, pathlib, statistics
import sys, tarfile, zipfile

DIMS = (8, 16, 32, 64)
DEPTHS = (0.4, 0.8)
SPLITS = (101, 303)
EXPECTED_RANDOM = ('104729', '130363', '155921', '196613', '262147')
METRICS = (
    'E_eq', 'E_identity_baseline', 'E_identity_data', 'E_identity_op',
    'E_word', 'E_hom_data', 'E_hom_op', 'E_rel_data', 'E_rel_op',
    'E_comp_data', 'E_comp_op', 'E_inverse_data', 'E_inverse_op',
    'E_rel_r4_data', 'E_rel_s2_data', 'E_rel_srs_r_inverse_data',
    'E_rel_r4_op', 'E_rel_s2_op', 'E_rel_srs_r_inverse_op',
    'pca_variance',
)
KEYS = ('model', 'group', 'layer', 'depth_slot', 'split_seed',
        'data_split', 'method', 'random_baseline_seed', 'optimization_seed')

def digest(b):
    return hashlib.sha256(b).hexdigest()

def file_digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def norm_int(s):
    return '' if s in ('',None) else str(int(float(s)))

def write_csv(path, rows, columns=None):
    if columns is None:
        columns=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=columns);w.writeheader();w.writerows(rows)

def close(a,b):
    # Historical pandas CSV hydration changed decimal serialization (example:
    # 0.02740675264027544 -> 0.0274067526402754). This comparison tolerance is
    # for checkpoint-to-aggregate serialization only, never for numerical rank.
    return math.isclose(a,b,abs_tol=1e-12,rel_tol=1e-12) or (math.isnan(a) and math.isnan(b))

def main():
    a=argparse.ArgumentParser()
    a.add_argument('--repo', type=pathlib.Path, required=True)
    a.add_argument('--output', type=pathlib.Path, required=True)
    args=a.parse_args();root=args.repo.resolve();out=args.output.resolve()
    if root == out or root in out.parents:
        raise ValueError('Use a separate output directory; never mutate the source repository')
    out.mkdir(parents=True,exist_ok=True)
    archive_specs=[
      ('main_full_003',
       'runpod_output_by_date/result_9_6_2026/run_outputs_9_4_2026_full3.zip',
       'run_outputs/main_experiment/main_full_003/', 'zip', 'config_snapshot.yaml'),
      ('legacy_full_replay',
       'submission/archive/runpod-from-zero-v1/2026-10-22-full-completed/full-results-v1.tar.gz',
       'historical/runs/legacy_full/', 'tar', 'config_snapshot.json'),
    ]
    all_detail=[];coverage=[];inputs=[];member_records=[];notes=[];failure_records=[]
    for run,relative,prefix,kind,config_name in archive_specs:
        path=root/relative
        if not path.exists():
            coverage.append({'run':run,'status':'unavailable','reason':'archive absent','source':relative})
            continue
        before=file_digest(path)
        arc=zipfile.ZipFile(path) if kind=='zip' else tarfile.open(path)
        names=set(arc.namelist() if kind=='zip' else arc.getnames())
        tar_cache={};tar_hashes={}
        if kind=='tar':
            # Hash compressed archives in member order. Repeated backwards
            # gzip seeks are unnecessary; keep only small CSV/JSON inputs.
            for entry in arc.getmembers():
                member=entry.name
                if not entry.isfile() or not member.startswith(prefix):continue
                rel=member[len(prefix):]
                if not (rel in ('manifest.json','run_ledger.json',config_name,'structural/template_metrics.csv')
                        or rel.startswith(('checkpoints/structural_base/','checkpoints/structural_joint/'))):continue
                b=arc.extractfile(entry).read()
                tar_hashes[member]=(digest(b),len(b))
                if not member.endswith('.npz'):tar_cache[member]=b
        def read(member):
            return arc.read(member) if kind=='zip' else tar_cache[member]
        mb=read(prefix+'manifest.json');manifest=json.loads(mb)
        lb=read(prefix+'run_ledger.json');ledger=json.loads(lb)
        cb=read(prefix+config_name)
        data_member=prefix+'structural/template_metrics.csv';data=read(data_member)
        recorded_hash=manifest.get('artifacts',{}).get('structural/template_metrics.csv')
        if recorded_hash != digest(data):
            raise ValueError(f'{run}: aggregate input differs from retained manifest')
        for member,b in [(prefix+'manifest.json',mb),(prefix+'run_ledger.json',lb),
                         (prefix+config_name,cb),(data_member,data)]:
            member_records.append({'archive':relative,'member':member,'sha256':digest(b),'bytes':len(b)})
        rows=list(csv.DictReader(io.StringIO(data.decode())))
        grid=[]
        for r in rows:
            if r['depth_slot']=='terminal':continue
            if int(r['latent_dim']) not in DIMS:continue
            if float(r['depth_slot']) not in DEPTHS:continue
            if int(r['split_seed']) not in SPLITS:continue
            r=dict(r)
            for k in ['latent_dim','layer','split_seed','random_baseline_seed','optimization_seed','template_id']:
                r[k]=norm_int(r[k])
            r['depth_slot']=format(float(r['depth_slot']),'.1f')
            grid.append(r)
        grouped=collections.defaultdict(list)
        for r in grid:
            grouped[tuple(r[k] for k in KEYS)+(r['latent_dim'],)].append(r)
        expected_cells=6*2*2*2*2*8*4 # models,groups,depths,splits,banks,method/seed variants,dimensions
        if len(grouped)!=expected_cells:
            raise ValueError(f'{run}: expected {expected_cells} aggregate cells, got {len(grouped)}')
        for k, rs in grouped.items():
            if len(rs)!=6 or len({r['template_id'] for r in rs})!=6:
                raise ValueError(f'{run}: incomplete or duplicate template coverage {k}')
            q16key=k[:-1]+('16',)
            baseline=grouped.get(q16key)
            if baseline is None or {r['template_id'] for r in baseline}!={r['template_id'] for r in rs}:
                raise ValueError(f'{run}: missing exact q16 template match {k}')
        checkpoint_units=collections.Counter();failed=[];verified_ckpt=0
        verified_checkpoint_sha={}
        for key, unit in ledger['units'].items():
            module=key.split(':')[0];f=unit.get('fields',{})
            if module not in ('structural_base','structural_joint'):continue
            if f.get('depth_slot')=='terminal':continue
            if f.get('latent_dim') not in DIMS or f.get('depth_slot') not in DEPTHS or f.get('split_seed') not in SPLITS:continue
            checkpoint_units[(module,f['latent_dim'],unit['status'])]+=1
            if unit['status']!='COMPLETE':
                failed.append({'run':run,'unit':key,'status':unit['status'],'fields':f,'message':unit.get('message','')})
                continue
            for obj in unit.get('outputs',[]):
                op=obj['path']; marker='/checkpoints/'
                if marker not in op:continue
                member=prefix+'checkpoints/'+op.split(marker,1)[1]
                if member not in names:
                    raise ValueError(f'{run}: missing checkpoint artifact {member}')
                if kind=='tar':sha,byte_count=tar_hashes[member]
                else:
                    b=read(member);sha=digest(b);byte_count=len(b)
                if sha != obj['sha256']:
                    raise ValueError(f'{run}: checkpoint checksum mismatch {member}')
                verified_checkpoint_sha[member]=sha
                member_records.append({'archive':relative,'member':member,'sha256':sha,'bytes':byte_count})
            verified_ckpt+=1
        failure_records.extend(failed)
        if verified_ckpt != 384:
            raise ValueError(f'{run}: expected 384 completed base/joint units; got {verified_ckpt}')
        # Cross-check every selected aggregate input row against its original
        # per-checkpoint CSV, so completion flags alone are not the evidence.
        raw_index={tuple(r[k] for k in KEYS)+(r['latent_dim'],r['template_id']):r for r in grid}
        checkpoint_rows_checked=0;max_absolute_roundtrip_error=0.;max_relative_roundtrip_error=0.
        for member in verified_checkpoint_sha:
            if not member.endswith('.csv') or member.endswith('.irrep.csv'):continue
            for r in csv.DictReader(io.StringIO(read(member).decode())):
                for k in ['latent_dim','layer','split_seed','random_baseline_seed','optimization_seed','template_id']:
                    r[k]=norm_int(r[k])
                r['depth_slot']=format(float(r['depth_slot']),'.1f')
                key=tuple(r[k] for k in KEYS)+(r['latent_dim'],r['template_id'])
                if key not in raw_index:raise ValueError(f'{run}: checkpoint row absent from aggregate {key}')
                for metric in METRICS:
                    av=raw_index[key].get(metric,'');bv=r.get(metric,'')
                    if av=='' and bv=='':continue
                    if not close(float(av),float(bv)):
                        raise ValueError(f'{run}: aggregate vs checkpoint disagreement {key} {metric}')
                    if math.isfinite(float(av)) and math.isfinite(float(bv)):
                        err=abs(float(av)-float(bv));max_absolute_roundtrip_error=max(max_absolute_roundtrip_error,err)
                        max_relative_roundtrip_error=max(max_relative_roundtrip_error,err/max(1.,abs(float(av)),abs(float(bv))))
                checkpoint_rows_checked+=1
        if checkpoint_rows_checked!=len(grid):
            raise ValueError(f'{run}: checkpoint/aggregate row coverage mismatch')
        for k,rs in sorted(grouped.items()):
            d=dict(zip(KEYS,k[:-1]));dim=int(k[-1]);b=grouped[k[:-1]+('16',)]
            detail={
                'run':run,'study':'Study 1 structural sensitivity','model':d['model'],
                'domain':d['group'],'readout':'block_output','layer':d['layer'],
                'depth_slot':d['depth_slot'],'split':d['split_seed'],'bank':d['data_split'],
                'q':dim,'method':d['method'],'optimization_seed':d['optimization_seed'],
                'random_baseline_seed':d['random_baseline_seed'],
                'template_count':len(rs),'template_ids':';'.join(sorted({r['template_id'] for r in rs},key=int)),
                'matched_q16_template_ids':True,'aggregation':'arithmetic mean of original six per-template values',
                'normalization':'unchanged historical endpoint normalization; see metric_definitions.json',
                'status':'executed_complete_traceable','source_archive':relative,
                'source_archive_sha256':before,'source_member':data_member,'source_member_sha256':digest(data),
            }
            for metric in METRICS:
                vals=[float(r[metric]) for r in rs if r.get(metric,'')!='']
                bs=[float(r[metric]) for r in b if r.get(metric,'')!='']
                detail[metric]=statistics.fmean(vals) if vals else ''
                detail[metric+'_matched_q16']=statistics.fmean(bs) if bs else ''
                detail[metric+'_minus_matched_q16']=statistics.fmean(vals)-statistics.fmean(bs) if vals and bs else ''
            all_detail.append(detail)
        for dim in DIMS:
            subset=[r for r in grid if int(r['latent_dim'])==dim]
            coverage.append({
                'run':run,'q':dim,'status':'executed_complete_traceable',
                'model_count':6,'domains':'C7;D4','depth_slots':'0.4;0.8','split_seeds':'101;303',
                'banks':'validation;test','templates_per_bank':6,
                'optimization_seed':42,'random_baseline_seeds':';'.join(EXPECTED_RANDOM),
                'base_units_complete':checkpoint_units[('structural_base',dim,'COMPLETE')],
                'joint_units_complete':checkpoint_units[('structural_joint',dim,'COMPLETE')],
                'template_metric_rows':len(subset),'unavailable_or_failed_cells':0,
                'source':relative+'::'+data_member,
                'reason':'q16 matched subset only; excluded its other depths/splits/terminal readout',
            })
        after=file_digest(path)
        if after!=before:raise ValueError('Input archive changed during audit')
        inputs.append({'path':relative,'sha256_before':before,'sha256_after':after,'unchanged':True})
        notes.append({'run':run,'manifest_status':manifest['status'],
                      'config_hash':ledger['config_hash'],'verified_completed_units':verified_ckpt,
                      'verified_checkpoint_rows':checkpoint_rows_checked,'aggregate_rows':len(rows),
                      'matched_input_rows':len(grid),'matched_aggregate_cells':len(grouped),
                      'checkpoint_csv_roundtrip_atol':1e-12,'checkpoint_csv_roundtrip_rtol':1e-12,
                      'max_absolute_roundtrip_error':max_absolute_roundtrip_error,
                      'max_roundtrip_error_over_max_one_absolute_value':max_relative_roundtrip_error})
        arc.close()
        print(json.dumps(notes[-1],sort_keys=True),flush=True)
    write_csv(out/'dimension_sensitivity_existing.csv',all_detail)
    write_csv(out/'dimension_sensitivity_coverage.csv',coverage)
    sums=collections.defaultdict(list)
    for r in all_detail:
        sums[(r['run'],r['domain'],r['bank'],r['method'],r['q'])].append(r)
    summaries=[]
    for key,rs in sorted(sums.items()):
        run,domain,bank,method,q=key
        d={'run':run,'domain':domain,'bank':bank,'method':method,'q':q,
           'models':6,'depths':2,'splits':2,'condition_count':24,
           'control_seeds_per_condition':5 if method=='random_orthogonal' else 1,
           'aggregation':'equal conditions; equal six templates; equal control seeds where present',
           'status':'existing matched descriptive summary'}
        for metric in METRICS:
            vals=[float(r[metric]) for r in rs if r[metric]!='']
            bs=[float(r[metric+'_matched_q16']) for r in rs if r[metric+'_matched_q16']!='']
            d[metric]=statistics.fmean(vals) if vals else ''
            d[metric+'_matched_q16']=statistics.fmean(bs) if bs else ''
            d[metric+'_minus_matched_q16']=statistics.fmean(vals)-statistics.fmean(bs) if vals and bs else ''
        summaries.append(d)
    write_csv(out/'dimension_sensitivity_summary.csv',summaries)
    (out/'input_preservation.json').write_text(json.dumps(inputs,indent=2)+'\n')
    (out/'member_hashes.json').write_text(json.dumps(member_records,indent=2)+'\n')
    (out/'verification_summary.json').write_text(json.dumps(notes,indent=2)+'\n')
    (out/'missing_and_failure_records.json').write_text(json.dumps(failure_records,indent=2)+'\n')
    (out/'metric_definitions.json').write_text(json.dumps({
        'E_eq':'Original nonidentity-action average of mean rowwise squared prediction error / (target squared norm + 1e-12); per-template value unchanged.',
        'E_hom_data':'Original full group-pair data consistency normalized_state_defect; unchanged targets and identity handling.',
        'E_comp_data':'Original composition data defect excluding identity-containing pairs; unchanged.',
        'E_rel_data':'Original domain-specific relation data defects as recorded; not pooled into a new common metric.',
        '*_op':'Original unsquared relative Frobenius diagnostics; not compared in magnitude with squared data errors.',
        'pca_variance':'Original stored explained-variance-ratio sum computed from the complete training SVD denominator; not recomputed from retained-coordinate variance.',
        'inference':'No new CI, bootstrap, significance test or dimension selection. Original and replay are distinct records, not independent datasets.',
        'scope':'Does not measure dimension stability of gamma_r, exact-action risk premiums, or registration rankings.',
    },indent=2)+'\n')
    print(f'WROTE {len(all_detail)} condition rows; {len(summaries)} matched summary rows',flush=True)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
