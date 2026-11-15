"""Continue the exact queue in a separate output, importing verified completed work."""
import argparse
import os
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from common import ROOT, HERE, runner, read, write, sha, identity, lock, safe_output, write_once, recovery_identity, validate_parent, verify_inventory


def compatible_output(parent,output):
    parent=Path(parent).resolve();output=safe_output(output).resolve()
    if output==parent or output.is_relative_to(parent) or parent.is_relative_to(output):
        raise ValueError('Continuation output and parent must be separate, nonoverlapping directories')
    return parent,output


def command(stage,out,parent):
    if stage=='historical':
        return [sys.executable,str(HERE/'historical_worker.py'),'--parent',str(parent),
                '--output',str(out/'historical'),'--execute-models']
    cmd=runner.command(stage,out)
    if stage in ('structural_transport','interface'):
        i=cmd.index('--cache-root')
        cmd[i+1]=str(parent/'characterization/cache')
    return cmd


def parent_inventory(parent):
    return {str(p.relative_to(parent)):sha(p) for p in sorted(parent.rglob('*')) if runner.collectable(p,parent)}


def collect(parent,out):
    parent,out=compatible_output(parent,out)
    descriptor=read(out/'identity.json')
    recovery_identity()
    verify_inventory(parent,read(out/'parent_evidence.json')['files'])
    dest=out.with_name(out.name+'.results.zip');temp=dest.with_name(dest.name+'.partial')
    entries={}
    for source,prefix in ((parent,'parent-'+parent.name),(out,out.name)):
        for p in sorted(source.rglob('*')):
            if runner.collectable(p,source):
                entries[prefix+'/'+str(p.relative_to(source))]=(p,sha(p))
    manifest={'files':{name:digest for name,(_,digest) in entries.items()},
        'parent_results_modified':False,'raw_hidden_states_included':False,
        'original_q16_D':'INFEASIBLE_RANK_DEFICIENT_PRESERVED',
        'rank_aware_D':'separate frozen C7=6,D4=3 experiment',
        'identity':descriptor,'paper_eligible':False}
    write(out/'export_manifest.json',manifest)
    entries[out.name+'/export_manifest.json']=(out/'export_manifest.json',sha(out/'export_manifest.json'))
    with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for name,(p,_) in entries.items():archive.write(p,name)
    with zipfile.ZipFile(temp) as archive:
        if archive.testzip():raise ValueError('Archive CRC failure')
    temp.replace(dest)
    dest.with_name(dest.name+'.sha256').write_text(sha(dest)+'  '+dest.name+'\n')
    print('RESULT_ARCHIVE '+str(dest),flush=True)
    return dest


def execute(parent,out):
    parent,out=compatible_output(parent,out)
    # Hold both parent writer locks for the entire continuation; all parent files
    # remain read-only, including failed states and their original source identity.
    with lock(parent/'.orchestrator.lock'),lock(parent/'historical/historical.lock'),lock(out/'.orchestrator.lock'):
        proof=validate_parent(parent)
        preflight=read(out/'environment/hardware_preflight.json')
        if preflight.get('status')!='hardware_passed':raise ValueError('Run the recovery start.sh preflight first')
        model_inventory=read(out/'environment/model_inventory.json')
        if not model_inventory.get('models'):raise ValueError('Missing sealed models')
        for item in model_inventory['models']:
            if sha(out/'environment'/item['seal_file'])!=item['seal_sha256']:
                raise ValueError('Model seal mismatch')
        descriptor={'parent_identity_sha256':sha(parent/'identity.json'),
            'parent_root':str(parent),'recovery_source_sha256':recovery_identity(),
            'original_source_sha256':read(parent/'identity.json')['source_sha256'],
            'queue':runner.plan(),'original_q16_D':'INFEASIBLE_RANK_DEFICIENT_PRESERVED',
            'historical_full_imports':proof['imported_stages']}
        write_once(out/'identity.json',descriptor)
        evidence=out/'parent_evidence.json'
        if evidence.exists():verify_inventory(parent,read(evidence)['files'])
        else:write(evidence,{'files':parent_inventory(parent),'read_only_parent':True})
        write_once(out/'original_failure.json',proof)
        write_once(out/'plan.json',{'original_queue':runner.plan(),
            'verified_imports':['characterization','compact_context','legacy_pre_full','legacy_full'],
            'historical_change':'AFS-A/B/C execution; record infeasible legacy q16 D, run unchanged rank-aware D separately',
            'thresholds_and_model_revisions_changed':False,'three_hour_timeout':False})
        state_path=out/'state.json'
        state=read(state_path) if state_path.exists() else {'status':'running','stages':{},
            'paper_eligible':False,'independent_confirmation':False,
            'original_q16_D':'INFEASIBLE_RANK_DEFICIENT_PRESERVED'}
        for stage in ('characterization','compact_context'):
            state['stages'][stage]={'status':'complete','imported_verified_from_parent':True}
        (out/'logs').mkdir(parents=True,exist_ok=True)
        active=None;start=time.monotonic()
        def save():
            state['status']=runner.result_state(state['stages'])
            state['elapsed_this_session_seconds']=round(time.monotonic()-start,1)
            write(state_path,state)
        def stop(signum,frame):
            if active is not None and active.poll() is None:
                os.killpg(active.pid,signal.SIGTERM)
                try:active.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(active.pid,signal.SIGKILL);active.wait()
            state['status']='interrupted';write(state_path,state)
            raise SystemExit(128+signum)
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        for stage,deps in runner.STAGES:
            if stage in ('characterization','compact_context'):
                print('IMPORT_VERIFIED '+stage,flush=True);continue
            bad=[d for d in deps if state['stages'].get(d,{}).get('status')!='complete']
            if bad:
                state['stages'][stage]={'status':'blocked_by_scientific_dependency','dependencies':bad};save();continue
            tick=time.monotonic();state['stages'][stage]={'status':'running'};save()
            print('START_STAGE '+stage,flush=True)
            env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'HF_HUB_OFFLINE':'1',
                 'TRANSFORMERS_OFFLINE':'1','PYTHONHASHSEED':'0'}
            with (out/'logs'/(stage+'.log')).open('ab',buffering=0) as log:
                active=subprocess.Popen(command(stage,out,parent),cwd=ROOT,env=env,stdout=log,
                    stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
                while True:
                    try:code=active.wait(timeout=30);break
                    except subprocess.TimeoutExpired:
                        print(f'HEARTBEAT stage={stage} elapsed={time.monotonic()-tick:.0f}s log=logs/{stage}.log',flush=True);save()
            active=None
            status='complete' if code==0 else 'scientific_gate_stopped' if code==3 else 'technical_failure'
            state['stages'][stage]={'status':status,'exit_code':code,'elapsed_seconds':time.monotonic()-tick};save()
            print('DONE_STAGE '+stage+' '+status,flush=True)
            if code not in (0,3):return 2
        save();collect(parent,out)
        print('CONTINUATION_FINISHED '+state['status']+'; original q16 D remains infeasible, not a pass.',flush=True)
        return 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('verify','run','status','collect'))
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--execute-models',action='store_true');a=p.parse_args()
    parent,out=compatible_output(a.parent,a.output)
    if a.action=='status':print(__import__('json').dumps(read(out/'state.json'),indent=2));return 0
    if a.action=='verify':
        print('RECOVERY_SOURCE_VERIFIED '+recovery_identity(),flush=True)
        proof=validate_parent(parent)
        print('PARENT_FULL_VERIFIED; original rank failure preserved; no models loaded',flush=True)
        return 0
    if a.action=='collect':
        with lock(parent/'.orchestrator.lock'),lock(parent/'historical/historical.lock'),lock(out/'.orchestrator.lock'):
            collect(parent,out)
        return 0
    if not a.execute_models:p.error('--execute-models required')
    return execute(parent,out)


if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,RuntimeError,FileNotFoundError,KeyError) as exc:
        print(type(exc).__name__+': '+str(exc),file=sys.stderr);raise SystemExit(2)
