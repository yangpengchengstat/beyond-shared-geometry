"""Run historical A/B/C, frozen rank-aware D and remaining controls after q16 failure."""
from copy import deepcopy
from pathlib import Path
import argparse
import os
import shutil
import sys
import time
from common import ROOT, HERE, historical as h, read, write, sha, identity, lock, write_once, recovery_identity, validate_parent

ABC_MODULES = {'generator_composition', 'algebra_permutation_null', 'correspondence_scramble'}


def adopt_abc(source, destination, ledger, expected_config):
    """Copy only verified completed A/B/C units under an explicit provenance receipt."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    original_config = read(source / 'derived_fresh_binding_config.json')
    strip = lambda cfg: {k: v for k, v in cfg.items() if k != 'replication_adapter'}
    if h.identity(strip(original_config)) != h.identity(strip(expected_config)):
        raise ValueError('Cannot reuse ABC: scientific configuration or FULL binding differs')
    old = read(source / 'run_ledger.json')
    proof = {'source_ledger_sha256': sha(source / 'run_ledger.json'),
             'source_config_sha256': sha(source / 'derived_fresh_binding_config.json'),
             'target_config_hash': ledger.data['config_hash'], 'units': {}}
    for key, value in old['units'].items():
        if key.split(':')[0] not in ABC_MODULES:
            continue
        if value.get('status') != 'COMPLETE':
            raise ValueError('Unexpected incomplete ABC unit')
        copied = deepcopy(value)
        for artifact in copied.get('outputs', []):
            old_path = Path(artifact['path'])
            if not old_path.is_absolute(): old_path = source / old_path
            if old_path.is_symlink() or not old_path.resolve().is_relative_to(source):
                raise ValueError('Reused checkpoint escapes source run')
            if old_path.stat().st_size != artifact['bytes'] or sha(old_path) != artifact['sha256']:
                raise ValueError('Reused checkpoint checksum mismatch')
            relative = old_path.relative_to(source)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_symlink(): raise ValueError('Symlink checkpoint destination')
            if target.exists():
                if sha(target) != artifact['sha256']: raise ValueError('Existing reused output differs')
            else: shutil.copyfile(old_path, target)
            artifact['path'] = str(target)
        copied['reused_from'] = {'source_run': str(source), 'source_unit': key,
                                 'source_ledger_sha256': proof['source_ledger_sha256']}
        if key in ledger.data['units'] and ledger.data['units'][key] != copied:
            raise ValueError('Existing migrated unit differs')
        ledger.data['units'][key] = copied
        proof['units'][key] = value
    ledger._save()
    write_once(destination / 'abc_checkpoint_reuse.json', proof)
    return proof


def validate_abc_pilot(root, original_config, source_receipt_sha256):
    import pandas as pd
    from bsg.experiments.algebra_falsification_config import validate_primary_coverage
    root = Path(root)
    manifest = h.validate_successful_manifest(root, tier='pre_full')
    if manifest['replication']['source_receipt_sha256'] != source_receipt_sha256 or manifest['execution_modules'] != ['A','B','C']:
        raise ValueError('Unexpected ABC pilot identity/modules')
    h.verify_ledger(root)
    for filename in ('generator_composition_metrics.csv', 'algebra_permutation_null.csv', 'correspondence_scramble_metrics.csv'):
        validate_primary_coverage(pd.read_csv(root / 'afs' / filename), original_config)
    gate = read(root / 'go_no_go.json')
    if gate.get('overall') != 'NOT_EVALUABLE':
        raise ValueError('Unexpected PRE-FULL scientific gate semantics')
    return {'authorized_for_afs_full': True, 'pre_full_run_id': manifest['run_id'],
        'authorization_basis': 'user_requested_full_historical_replay_and_verified_ABC_pilot_coverage',
        'scientific_pass': False, 'scientific_gate': gate,
        'manifest_sha256': sha(root/'manifest.json'), 'ledger_sha256': sha(root/'run_ledger.json'),
        'source_receipt_sha256': source_receipt_sha256,
        'legacy_D': 'rank failure retained separately; no q16 result or pass claimed'}


class Recovery(h.HistoricalReplay):
    def __init__(self, parent, output):
        self.project, self.parent, self.output = ROOT, Path(parent).resolve(), Path(output).resolve()
        if self.output == self.parent or self.output.is_relative_to(self.parent) or self.output.is_relative_to(ROOT):
            raise ValueError('Recovery output must be separate from parent and source')
        sys.path.insert(0, str(ROOT/'src'))
        self.imports = validate_parent(self.parent)
        self.source = h.source_snapshot(ROOT)
        self.recovery_hash = recovery_identity()
        self.contract = {'study_id':'bsg-rank-aware-historical-continuation-v1',
            'source':self.source, 'recovery_source_sha256':self.recovery_hash,
            'environment':h.environment(), 'parent':self.imports, 'reproduction_not_new_confirmation':True}
        self.output.mkdir(parents=True, exist_ok=True)
        write_once(self.output/'source_receipt.json',self.contract)
        self.receipt_sha = sha(self.output/'source_receipt.json')
        self.cache = self.parent/'historical/activation_cache'
        write_once(self.output/'original_q16_D_failure.json',self.imports)
        for stage, entry in self.imports['imported_stages'].items():
            write_once(self.output/'stage_state'/(stage+'.json'), {
                'stage':stage,'status':'complete','source_receipt_sha256':self.receipt_sha,
                'imported_unchanged':True,'parent_stage_state_sha256':entry['state_sha256'],
                'outputs':entry['outputs']})

    def run_dir(self, stage):
        if stage in ('legacy_pre_full','legacy_full'):
            return self.parent/'historical/runs'/stage
        return self.output/'runs'/stage

    def assert_source(self):
        if h.source_snapshot(self.project) != self.source or recovery_identity() != self.recovery_hash:
            raise ValueError('Numerical/recovery source changed during continuation')

    def replication(self):
        return {**super().replication(), 'rank_recovery_source_sha256':self.recovery_hash,
                'parent_historical_source_receipt_sha256':self.imports['parent_historical_source_receipt_sha256'],
                'parent_results_modified':False, 'original_q16_D_failed_not_replaced':True}

    def legacy(self,tier):
        raise ValueError('Legacy FULL/PRE-FULL are read-only verified imports; never rerun them here')

    def afs(self,tier):
        import pandas as pd
        from bsg.experiments import algebra_falsification as m
        from bsg.experiments.algebra_falsification_config import resolve_afs_config, afs_config_hash, validate_primary_coverage
        from bsg.experiments.algebra_falsification_outputs import generator_inference, scramble_inference, evaluate_gates, write_stable_outputs
        from bsg.experiments.runtime_state import RunLedger, RuntimeTelemetry
        original = resolve_afs_config(self.project/'configs/algebra_falsification.yaml',tier)
        cfg = deepcopy(original)
        cfg['historical_source'] = h.source_binding(self.run_dir('legacy_full'))
        cfg['replication_adapter'] = self.replication()
        stage = 'afs_'+tier
        authorization = validate_abc_pilot(self.run_dir('afs_pre_full'),
            resolve_afs_config(self.project/'configs/algebra_falsification.yaml','pre_full'),self.receipt_sha) if tier=='full' else None
        directory = m.create_output_tree(self.output/'runs', stage)
        ledger = RunLedger(directory/'run_ledger.json',stage,afs_config_hash(cfg))
        telemetry = RuntimeTelemetry(directory/'runtime_profile.csv')
        proposed = m._build_manifest(cfg,stage,self.project,include_embedding=False)
        proposed.update(replication=self.replication(), legacy_q16_D_status='INFEASIBLE_RANK_DEFICIENT',
                        repaired_D_separate_stage='afs_d')
        manifest = self.base_manifest(directory,proposed)
        write_once(directory/'original_numerical_config.json',original)
        write_once(directory/'derived_fresh_binding_config.json',cfg)
        if authorization: write_once(directory/'pre_full_authorization.json',authorization)
        if tier=='pre_full': adopt_abc(self.parent/'historical/runs/afs_pre_full',directory,ledger,cfg)
        original_complete=ledger.complete
        expected_units=3*len(cfg['selected_models'])*len(cfg['groups'])*len(cfg['depth_slots'])*len(cfg['split_seeds'])
        def complete_with_progress(key,fields,paths,metrics=None):
            original_complete(key,fields,paths,metrics)
            done=sum(v.get('status')=='COMPLETE' and k.split(':')[0] in ABC_MODULES for k,v in ledger.data['units'].items())
            print(f"AFS_ABC_CHECKPOINT {tier} {key.split(':')[0]} completed={done}/{expected_units}",flush=True)
        ledger.complete=complete_with_progress
        source = h.FreshCheckpointSource(self.run_dir('legacy_full'),cfg['historical_source'])
        try:
            m._run_reused_modules(cfg,source,m.group_specs(),directory,ledger,telemetry)
            generator=m._collect(ledger,'generator_composition')
            permutation_raw=m._collect(ledger,'algebra_permutation_null')
            scramble=m._collect(ledger,'correspondence_scramble')
            for frame in (generator,permutation_raw,scramble):validate_primary_coverage(frame,cfg)
            permutation=m._permutation_summaries(permutation_raw)
            print('AFS_ABC_POSTPROCESS '+tier,flush=True)
            gci=generator_inference(generator,n_boot=int(cfg['bootstrap_samples']),seed=int(cfg['bootstrap_seed']))
            sci=scramble_inference(scramble,n_boot=int(cfg['bootstrap_samples']),seed=int(cfg['bootstrap_seed']))
            gate=evaluate_gates(generator,gci,permutation,sci,cfg)
            gate.update(execution_modules=['A','B','C'],D='NOT_EVALUABLE_RANK_DEFICIENT',
                        rank_aware_D_separate_stage='afs_d')
            write_stable_outputs(directory,generator,permutation_raw,permutation,scramble,None,gci,sci)
            write(directory/'go_no_go.json',gate)
            source.assert_unchanged()
            self.finish(directory,manifest)
        except BaseException as exc:
            self.finish(directory,manifest,error=exc)
            raise
        return {'run':str(directory),'scientific_gate':gate,'falsification_failure_is_retained':True,
                'original_q16_D':'infeasible; not counted as successful or automatically reduced'}


def execute(parent,output):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.manual_seed(42)
    worker=Recovery(parent,output)
    blocked=[]
    with lock(worker.output/'historical.lock'):
        for stage in h.STAGES:
            try: worker.run_stage(stage)
            except h.ScientificStop as exc:
                blocked.append(stage)
                write(worker.output/'stage_state'/(stage+'.json'),{
                    'stage':stage,'status':'gate_stopped','source_receipt_sha256':worker.receipt_sha,
                    'error':{'type':type(exc).__name__,'message':str(exc)}})
                print('BLOCKED_HISTORICAL '+stage+': '+str(exc),flush=True)
        write(worker.output/'historical_status.json',{
            'status':'completed_with_scientific_blocks' if blocked else 'complete',
            'blocked_stages':blocked,'original_q16_D':'INFEASIBLE_RANK_DEFICIENT_PRESERVED',
            'rank_aware_D':read(worker.output/'stage_state/afs_d.json')['status'],
            'parent_results_modified':False,'scientific_pass_claimed':False})
    return 3 if blocked else 0


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--execute-models',action='store_true');a=p.parse_args()
    if not a.execute_models:p.error('--execute-models required')
    try:raise SystemExit(execute(a.parent,a.output))
    except (ValueError,RuntimeError,FileNotFoundError,KeyError) as exc:
        print(type(exc).__name__+': '+str(exc),file=sys.stderr);raise SystemExit(2)
