#!/usr/bin/env python3
"""Serial from-zero research replay. Scientific stops never become passes."""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ['TOKENIZERS_PARALLELISM']='false'
import argparse, signal, subprocess, sys, time, zipfile
from pathlib import Path
from receipt import ROOT,read,write,sha,identity,source_identity,safe_output,lock
HERE=Path(__file__).resolve().parent

# Independent branches are intentionally not hidden behind a global scientific gate.
STAGES=[
    ('characterization',()),('compact_context',('characterization',)),('historical',()),
    ('original_calibration',()),('revised_calibration',()),('pilot',('revised_calibration',)),
    ('confirmation',('pilot',)),('native_prepare',()),('native_confirmation',('native_prepare',)),
    ('peer_anchors',()),('peer_factorial',()),('word_extract',()),('word_analyze',('word_extract',)),
    ('structural_transport',('characterization',)),('product',()),('interface',('characterization',))]

def command(stage,out):
    py=sys.executable
    if stage=='characterization':return [py,str(HERE/'characterization.py'),'--output',str(out/stage),'--execute-models']
    if stage=='compact_context':return [py,str(HERE/'compact_replay.py'),'--parent',str(out/'characterization'),'--output',str(out/stage)]
    if stage=='historical':return [py,str(HERE/'historical.py'),'--project-root',str(ROOT),'--output',str(out/stage),'--stage','all','--execute-models']
    cmd=[py,str(HERE/'prospective.py'),'--task',stage,'--output',str(out/'prospective'),'--execute-models']
    if stage in ('structural_transport','interface'):cmd += ['--cache-root',str(out/'characterization/cache')]
    return cmd

def plan():
    return {'study_id':'bsg-from-zero-reproduction-v1','scope':'current_mainline_and_all_implemented_historical_experiments',
        'stages':[{'id':name,'depends_on':list(deps)} for name,deps in STAGES],
        'fixed_limits':{'gpu_workers':1,'models_parallel':1,'three_hour_timeout':False},
        'scientific_role':'previously inspected designs; reproduction, not a new independent confirmatory sample',
        'missing_external_prerequisites':{
            'sun_toneva_getimpl':'No behavior-preserving implementation family data or GetImpl discovery pipeline supplied; not runnable from this repository.',
            'external_multilingual_contexts':'No independent authored external language banks supplied.',
            'third_lineage':'Conditional future study; disabled in registered six-model study, not a previously run historical result.'},
        'calibration':'Both original and revised design recipes, current corrected numerical engine; no emulation of old numerical defects',
        'scope_note':'Historical FULL/AFS/additive recipes plus current characterization, compact/context, prospective gates, peer adaptations, word transport, product/interface extensions. Prior ad-hoc debugging sessions are preserved as records, not counted as independent formal experiments.'}

def result_state(stages):
    if any(v['status']=='technical_failure' for v in stages.values()):return 'technical_failure'
    if len(stages)!=len(STAGES) or any(v['status']=='running' for v in stages.values()):return 'running'
    if any(v['status']!='complete' for v in stages.values()):return 'completed_with_scientific_stops'
    return 'completed_all_implemented_stages'

def execute(out,execute_models=False):
    if not execute_models:raise ValueError('Explicit --execute-models required to run the A40 replay')
    out=safe_output(out);out.mkdir(parents=True,exist_ok=True)
    source=source_identity();protocol=read(HERE/'protocol.json')
    preflight=out/'environment/hardware_preflight.json'
    if not preflight.exists() or read(preflight).get('status')!='hardware_passed':
        raise ValueError('Run start.sh first: a successful hardware/environment check is required')
    models=out/'environment/model_inventory.json'
    if not models.exists():raise ValueError('Pinned model inventory missing; run start.sh first')
    for item in read(models)['models']:
        if sha(out/'environment'/item['seal_file'])!=item['seal_sha256']:
            raise ValueError('Model seal was modified')
    descriptor={'source_sha256':source,'protocol_sha256':sha(HERE/'protocol.json'),'plan':plan()}
    with lock(out/'.orchestrator.lock'):
        if (out/'identity.json').exists():
            if read(out/'identity.json')!=descriptor:raise ValueError('This directory belongs to another source/protocol')
        elif any(p.name not in ('.orchestrator.lock','.launch.lock','environment','logs') for p in out.iterdir()):
            raise ValueError('Unrecognized output directory; choose a fresh path')
        else:write(out/'identity.json',descriptor)
        write(out/'plan.json',plan());write(out/'protocol.json',protocol)
        (out/'logs').mkdir(exist_ok=True)
        state={'status':'running','stages':{},'paper_eligible':False,'independent_confirmation':False}
        if (out/'state.json').exists():state=read(out/'state.json')
        start=time.monotonic();active=None
        def save():
            state['status']=result_state(state['stages']);state['elapsed_this_session_seconds']=round(time.monotonic()-start,1);write(out/'state.json',state)
        def stopped(signum,frame):
            if active is not None and active.poll() is None:
                os.killpg(active.pid,signal.SIGTERM)
                try:active.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(active.pid,signal.SIGKILL);active.wait()
            state['status']='interrupted';write(out/'state.json',state);raise SystemExit(128+signum)
        signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
        for stage,deps in STAGES:
            bad=[d for d in deps if state['stages'].get(d,{}).get('status')!='complete']
            if bad:
                state['stages'][stage]={'status':'blocked_by_scientific_dependency','dependencies':bad};save();continue
            print('START_STAGE '+stage,flush=True);tick=time.monotonic()
            state['stages'][stage]={'status':'running'};save()
            with (out/'logs'/(stage+'.log')).open('ab',buffering=0) as log:
                env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'PYTHONHASHSEED':'0','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
                active=subprocess.Popen(command(stage,out),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env,start_new_session=True)
                while True:
                    try:code=active.wait(timeout=30);break
                    except subprocess.TimeoutExpired:
                        print(f'HEARTBEAT stage={stage} elapsed={time.monotonic()-tick:.0f}s log=logs/{stage}.log',flush=True);save()
            active=None
            status='complete' if code==0 else 'scientific_gate_stopped' if code==3 else 'technical_failure'
            state['stages'][stage]={'status':status,'exit_code':code,'elapsed_seconds':time.monotonic()-tick,'log':'logs/'+stage+'.log'};save()
            print('DONE_STAGE '+stage+' '+status,flush=True)
            if status=='technical_failure':
                print('TECHNICAL_STOP: inspect the named log; fix environment or code before resuming. Existing evidence preserved.',flush=True)
                return 2
        save();collect(out)
        print('REPLAY_FINISHED '+state['status']+'; inspect scientific stops and scope ledger. No manuscript promotion.',flush=True)
        return 0

# Include numerical evidence and row banks, exclude only regenerable bulky caches.
CACHE_PARTS={'cache','caches','activation_cache','embedding_cache','hf_cache','extract_development','extract_confirmation','merged','field_cache','model_cache','native_cache'}
def collectable(path,root):
    rel=path.relative_to(root)
    if not path.is_file() or path.is_symlink() or any(p.startswith('.') for p in rel.parts) or path.name=='export_manifest.json' or path.name.endswith(('.partial','.lock','.zip','.sha256')):return False
    if any(p in CACHE_PARTS for p in rel.parts):return False
    if path.suffix in ('.safetensors','.bin','.pt','.pth'):return False
    # Preserve numerical pilot contrasts; exclude only actual raw extraction arrays.
    if (path.suffix=='.npy' and (path.name.startswith('layer_') or path.name=='potential_features.npy')) or path.name in ('token','stored_tokens','credentials') or path.name.endswith(('.pem','.key')):return False
    return True

def collect(out):
    out=Path(out);dest=out.with_name(out.name+'.results.zip');temp=dest.with_name(dest.name+'.partial')
    files=sorted(p for p in out.rglob('*') if collectable(p,out))
    filemap={str(p.relative_to(out)):sha(p) for p in files}
    write(out/'export_manifest.json',{'files':filemap,'raw_hidden_states_included':False,
        'scope':'Results, generated rows outside raw caches, q16 coordinates, fits, configuration, model/environment hashes and logs. Raw high-dimensional arrays are regenerable and excluded; local model weights and credentials are never collected.'})
    files.append(out/'export_manifest.json')
    with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for p in files:archive.write(p,out.name+'/'+str(p.relative_to(out)))
    with zipfile.ZipFile(temp) as archive:
        if archive.testzip():raise ValueError('Result archive CRC failure')
    temp.replace(dest);dest.with_name(dest.name+'.sha256').write_text(sha(dest)+'  '+dest.name+'\n')
    print('RESULT_ARCHIVE '+str(dest),flush=True)
    return dest

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['plan','verify','run','status','collect']);p.add_argument('--output',type=Path)
    p.add_argument('--execute-models',action='store_true');a=p.parse_args()
    if a.action=='plan':print(__import__('json').dumps(plan(),indent=2));return 0
    if a.action=='verify':print('SOURCE_VERIFIED '+source_identity());return 0
    if not a.output:p.error('--output required')
    if a.action=='status':print(__import__('json').dumps(read(a.output/'state.json'),indent=2));return 0
    if a.action=='collect':
        with lock(a.output/'.orchestrator.lock'):collect(a.output)
        return 0
    return execute(a.output,a.execute_models)
if __name__=='__main__':raise SystemExit(main())
