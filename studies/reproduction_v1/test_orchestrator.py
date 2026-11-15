import importlib.util, sys, zipfile
from pathlib import Path
import pytest
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
spec=importlib.util.spec_from_file_location('reproduction_orchestrator',HERE/'run.py');runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
from receipt import write,read,sha,verify_inventory,safe_output

def test_dependency_order_and_complete_historical_scope():
    seen=set()
    for name,deps in runner.STAGES:
        assert set(deps)<=seen;seen.add(name)
    assert {'historical','original_calibration','revised_calibration','compact_context','peer_anchors','word_extract','interface'}<=seen
    assert 'characterization' not in dict(runner.STAGES)['historical']

def test_running_status_not_confused_with_finished_gate_stop():
    states={name:{'status':'complete'} for name,_ in runner.STAGES}
    states['characterization']['status']='running'
    assert runner.result_state(states)=='running'
    states['characterization']['status']='scientific_gate_stopped'
    assert runner.result_state(states)=='completed_with_scientific_stops'

def test_fresh_output_required():
    with pytest.raises(ValueError):safe_output(runner.ROOT/'local_results/forbidden')
    with pytest.raises(ValueError):safe_output('/workspace/bsg-runs/icml2027-characterization-v1')

def test_inventory_mutation_rejected(tmp_path):
    p=tmp_path/'result.json';write(p,{'score':1});files={'result.json':sha(p)}
    assert verify_inventory(tmp_path,files)
    write(p,{'score':2})
    with pytest.raises(ValueError):verify_inventory(tmp_path,files)

def test_collection_repeatable_complete_small_evidence_and_excludes_weights(tmp_path):
    root=tmp_path/'run'
    write(root/'state.json',{'status':'fixture'})
    write(root/'environment/model_inventory.json',{'models':[]})
    write(root/'characterization/analysis/cell/result.json',{'metric':0.1})
    for name in ('characterization/cache/raw.npy','.cache/huggingface/token','model.safetensors','activation_cache/large.npz'):
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'not-exported')
    for _ in range(2):
        archive=runner.collect(root)
        with zipfile.ZipFile(archive) as z:
            names=z.namelist();assert len(names)==len(set(names));assert not any('not-exported' in z.read(n).decode(errors='ignore') for n in names)
            manifest=read(root/'export_manifest.json')
            for name,digest in manifest['files'].items():assert sha(root/name)==digest
        assert archive.with_name(archive.name+'.sha256').exists()

def test_run_requires_execution_and_preflight_before_experiment(tmp_path):
    with pytest.raises(ValueError,match='Explicit'):runner.execute(tmp_path/'run')

def test_real_subprocess_queue_continues_independent_stages_after_gate(tmp_path,monkeypatch):
    root=tmp_path/'run';write(root/'environment/hardware_preflight.json',{'status':'hardware_passed'});write(root/'environment/model_inventory.json',{'models':[]})
    script=tmp_path/'fake_worker.py'
    script.write_text('import sys\nprint("offline fixture stage "+sys.argv[1],flush=True)\nsys.exit(3 if sys.argv[1]=="pilot" else 0)\n')
    monkeypatch.setattr(runner,'source_identity',lambda:'offline-fixture')
    monkeypatch.setattr(runner,'command',lambda name,out:[sys.executable,str(script),name])
    assert runner.execute(root,execute_models=True)==0
    state=read(root/'state.json')
    assert state['stages']['pilot']['status']=='scientific_gate_stopped'
    assert state['stages']['confirmation']['status']=='blocked_by_scientific_dependency'
    assert state['stages']['peer_anchors']['status']=='complete'
    assert state['stages']['interface']['status']=='complete'
    assert state['status']=='completed_with_scientific_stops'
    assert not state['paper_eligible']
    assert root.with_name('run.results.zip').exists()
