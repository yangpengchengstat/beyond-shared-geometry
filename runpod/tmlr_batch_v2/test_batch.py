"""Controller fixtures only; no scientific analysis, model import or network."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch
import zipfile

spec=importlib.util.spec_from_file_location('batch',Path(__file__).with_name('run.py'))
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)/'batch';self.root.mkdir()
        self.state={'status':'ready','stages':{},'context':'context'}
        self.cp=self.root/'context';self.cp.mkdir()
        b.atomic(self.cp/'input_manifest.json',{'bank_sha256':'bank','source_manifest_sha256':'source',
            'cells':[{'status':'prepared','cell':str(i)} for i in range(48)]})
        (self.cp/'input_manifest.sha256').write_text(b.sha(self.cp/'input_manifest.json'))
        b.atomic(self.cp/'human_review.json',{'bank_sha256':'bank','source_manifest_sha256':'source','approval_before_model_execution':True})
        b.atomic(self.root/'environment.json',b.environment())
        b.checkpoint(self.root,self.state)

    def tearDown(self):self.temp.cleanup()

    def fake_child(self,cmd,log,name,root):
        log.parent.mkdir(exist_ok=True);log.write_text(name+'\n')
        out=Path(cmd[cmd.index('--output')+1]) if '--output' in cmd else self.cp
        out.mkdir(parents=True,exist_ok=True)
        if name=='spectral_analyze':
            b.atomic(out/'validation.json',{'reported':48,'planned':48})
            b.atomic(out/'output_manifest.json',{'files':{'validation.json':b.sha(out/'validation.json')}})
        elif name in ('witness','simulate','alternatives'):
            n={'witness':48,'simulate':36,'alternatives':2952}[name]
            b.atomic(out/'state.json',{'execution_status':'complete'})
            b.atomic(out/'plan.json',{'units':list(range(n))})
        else:b.atomic(out/(name+'.json'),{'fixture':True})
        if name=='context_analyze':b.atomic(out/'summary.json',{'status':'complete'})

    def load_fixture(self,*args):return Path('/fixture-source'),Path('/fixture.zip'),b.read(self.root/'state.json')

    def test_full_queue_and_completed_resume_never_call_old_launchers(self):
        with patch.object(b,'load',side_effect=self.load_fixture),patch.object(b,'child',side_effect=self.fake_child) as c:
            b.run(self.root,True,True,False)
            self.assertEqual([x.args[2] for x in c.call_args_list],list(b.STEPS))
            c.reset_mock();b.run(self.root,True,True,False);c.assert_not_called()
        self.assertEqual(b.read(self.root/'state.json')['status'],'completed_pending_scientific_review')
        self.assertEqual(len(list(self.root.parent.glob('*.results.zip'))),2)

    def test_explicit_execution_flags_required(self):
        with patch.object(b,'child') as c:
            with self.assertRaises(ValueError):b.run(self.root,False,True,False)
            c.assert_not_called()

    def test_incomplete_preparation_cannot_start_models(self):
        m=b.read(self.cp/'input_manifest.json');m['cells'][0].update(status='not_evaluable_input',reason='fixture input mismatch')
        b.atomic(self.cp/'input_manifest.json',m)
        with patch.object(b,'load',side_effect=self.load_fixture),patch.object(b,'child') as c:
            with self.assertRaisesRegex(ValueError,'fixture input mismatch'):b.run(self.root,True,True,False)
            c.assert_not_called()

    def test_unreviewed_bank_blocks_all_execution(self):
        (self.cp/'human_review.json').unlink()
        with patch.object(b,'load',side_effect=self.load_fixture),patch.object(b,'child') as c:
            with self.assertRaisesRegex(ValueError,'Human review pending'):b.run(self.root,True,True,False)
            c.assert_not_called()

    def test_changed_bank_review_is_refused(self):
        b.atomic(self.cp/'human_review.json',{'bank_sha256':'other'})
        with self.assertRaises(ValueError):b.require_review(self.cp)

    def test_corrupt_complete_stage_stops_before_rerun(self):
        cmd=lambda out:['fixture','--output',str(out)]
        with patch.object(b,'child',side_effect=self.fake_child):
            out=b.stage(self.root,self.state,'probe',self.root/'probe',cmd)
        (out/'probe.json').write_text('corrupt')
        with patch.object(b,'child') as c:
            with self.assertRaisesRegex(ValueError,'Changed or missing'):b.stage(self.root,self.state,'probe',self.root/'probe',cmd)
            c.assert_not_called()

    def test_interrupted_nonresumable_stage_keeps_old_attempt(self):
        cmd=lambda out:['fixture','--output',str(out)]
        def fail(command,log,name,root):
            p=Path(command[-1]);p.mkdir();(p/'partial.json').write_text('{}');raise RuntimeError('fixture interruption')
        with patch.object(b,'child',side_effect=fail):
            with self.assertRaises(RuntimeError):b.stage(self.root,self.state,'spectral_probe',self.root/'spectral',cmd)
        original=self.root/self.state['stages']['spectral_probe']['output']
        with patch.object(b,'child',side_effect=self.fake_child):
            new=b.stage(self.root,self.state,'spectral_probe',self.root/'spectral',cmd)
        self.assertNotEqual(new,original);self.assertTrue((original/'partial.json').exists())

    def test_resumable_stage_reuses_same_output(self):
        commands=[]
        def cmd(out):return ['fixture','--output',str(out)]+(['--resume'] if out.exists() else [])
        def fail(command,log,name,root):
            commands.append(command);p=Path(command[2]);p.mkdir();(p/'unit.json').write_text('{}');raise RuntimeError('fixture')
        with patch.object(b,'child',side_effect=fail):
            with self.assertRaises(RuntimeError):b.stage(self.root,self.state,'units',self.root/'units',cmd,resume=True)
        with patch.object(b,'child',side_effect=self.fake_child) as c:
            result=b.stage(self.root,self.state,'units',self.root/'units',cmd,resume=True)
            self.assertIn('--resume',c.call_args.args[0])
        self.assertTrue((result/'unit.json').exists())

    def test_collection_preserves_partial_and_hashes(self):
        (self.root/'interrupted.partial').write_text('retained')
        out=b.collect_unlocked(self.root)
        self.assertEqual(out.with_name(out.name+'.sha256').read_text().split()[0],b.sha(out))
        with zipfile.ZipFile(out) as z:
            self.assertEqual(z.read('interrupted.partial'),b'retained')
            self.assertTrue(json.loads(z.read('batch_export_manifest.json'))['partial'])

    def test_parallel_controller_refused(self):
        with b.lock(self.root):
            with self.assertRaises(ValueError):
                with b.lock(self.root):pass

    def test_runtime_change_blocks_execution(self):
        d=b.environment();d['python']='wrong';b.atomic(self.root/'environment.json',d)
        with patch.object(b,'load',side_effect=self.load_fixture),patch.object(b,'child') as c:
            with self.assertRaisesRegex(ValueError,'Runtime versions changed'):b.run(self.root,True,True,False)
            c.assert_not_called()

    def test_path_escape_and_symlink_rejected(self):
        with self.assertRaises(ValueError):b.verify(self.root,{'../outside':'none'})
        (self.root/'link').symlink_to(self.cp)
        with self.assertRaises(ValueError):b.tree(self.root)

    def test_real_child_exit_and_log_capture(self):
        log=self.root/'logs'/'actual-fixture.log'
        with self.assertRaisesRegex(RuntimeError,'exited 7'):
            b.child([sys.executable,'-c',"print('fixture-output'); raise SystemExit(7)"],log,'fixture-process',self.root)
        self.assertIn('fixture-output',log.read_text())


if __name__=='__main__':unittest.main()
