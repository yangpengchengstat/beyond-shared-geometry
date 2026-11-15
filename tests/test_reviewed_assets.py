"""Regression tests for provenance ownership and audited-input boundaries."""
from pathlib import Path
import hashlib
import importlib.util
import json
import shutil
import tempfile

ROOT=Path(__file__).resolve().parents[1]
def _module(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def test_historical_rebuild_preserves_separately_reviewed_provenance():
    mod=_module('build_paper_assets')
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        for directory in ['results','release','paper/generated','paper/figures']:(root/directory).mkdir(parents=True)
        (root/'results/registry.json').write_text('{"entries":[]}')
        (root/'results/README.md').write_text('Historical evidence')
        prior={'path':'paper/generated/characterization_main.tex','sha256':'reviewed','source_run':'independent','source_path':'specific-original-member','role':'reviewed-descriptive-paper-asset'}
        (root/'release/artifact_manifest.json').write_text(json.dumps({'artifacts':[prior]}))
        mod.ROOT=root;mod.R=root/'results';mod.GENERATED=root/'paper/generated';mod.FIGURES=root/'paper/figures'
        mod.update_manifest()
        rows=json.loads((root/'release/artifact_manifest.json').read_text())['artifacts']
        assert next(r for r in rows if r['path']==prior['path'])==prior

def test_reviewed_renderer_rejects_changed_inputs_and_promoted_flags():
    mod=_module('build_characterization_assets')
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);(root/'release').mkdir();shutil.copytree(ROOT/'results/characterization_v1',root/'results/characterization_v1')
        path=root/'release/characterization_review.json';shutil.copyfile(ROOT/'release/characterization_review.json',path)
        mod.load_reviewed(root)
        original=path.read_text();record=json.loads(original);record['formal_inference']=True;path.write_text(json.dumps(record))
        try:mod.load_reviewed(root)
        except ValueError as e:assert 'cannot upgrade' in str(e)
        else:raise AssertionError('renderer accepted promotion')
        path.write_text(original);csv=root/'results/characterization_v1/prediction_summary.csv';csv.write_text(csv.read_text()+'tampered')
        try:mod.load_reviewed(root)
        except ValueError as e:assert 'identity changed' in str(e)
        else:raise AssertionError('renderer accepted changed result')

def test_curated_input_rejects_path_escape_and_symlink():
    mod=_module('build_characterization_assets')
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);(root/'real.txt').write_text('fixture');(root/'link.txt').symlink_to('real.txt')
        for name in ('../outside','link.txt'):
            try:mod._curated_path(root,name)
            except ValueError:pass
            else:raise AssertionError('unsafe curated path accepted')

def test_generated_paper_sources_have_no_accidental_control_characters():
    for path in (ROOT/'paper/generated').glob('*.tex'):
        assert not [b for b in path.read_bytes() if b<32 and b not in (9,10)],path.name
    assert r'E_{\rm eq}' in (ROOT/'paper/generated/lexical_control.tex').read_text()
