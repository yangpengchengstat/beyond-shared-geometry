"""TMLR migration contracts; tiny software fixtures, never empirical execution."""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tests'))
from check_release import check_release, sha256, ReleaseError
from package_release import package_release
from test_release_contract import _fixture, _rejects
from verify_tmlr_studies import verify as verify_study, verify_all


def _tmlr_fixture(root):
    _fixture(root)
    (root / 'paper/tmlr.sty').write_text('\\ProvidesPackage{tmlr}\n')
    p = root / 'paper/main.tex'
    p.write_text(p.read_text().replace('icml2026', 'tmlr'))
    p = root / 'release/release_manifest.json'
    d = json.loads(p.read_text())
    d.update(target_venue='TMLR', template_status='official_tmlr_unmodified',
             submission_ready=False, pending_analyses=['fixed_bank_analysis'])
    p.write_text(json.dumps(d))
    (root / 'provenance/template_source.json').write_text(json.dumps({
        'target_venue': 'TMLR', 'modified': False, 'source_url': 'https://github.com/JmlrOrg/tmlr-style-file',
        'support_sha256': {'tmlr.sty': sha256(root / 'paper/tmlr.sty')}}))
    (root / 'paper/TMLR_TEMPLATE_LICENSE.txt').write_text('Template license notice preserved\n')


def test_tmlr_release_keeps_pending_status_and_anonymous_template():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp); _tmlr_fixture(root)
        result = check_release(root)
        assert result['target_venue'] == 'TMLR'
        assert result['submission_ready'] is False
        assert result['pending_analyses'] == ['fixed_bank_analysis']
        p = root / 'paper/main.tex'
        p.write_text(p.read_text().replace('\\begin{document}', '\\icmlfinalcopy\n\\begin{document}'))
        _rejects(lambda: check_release(root), 'anonymous TMLR mode')


def test_tmlr_release_rejects_readiness_promotion_and_unverified_style():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp); _tmlr_fixture(root)
        p = root / 'release/release_manifest.json'; original = json.loads(p.read_text())
        d = dict(original); d['submission_ready'] = True; p.write_text(json.dumps(d))
        _rejects(lambda: check_release(root), 'submission_ready=false')
        d = dict(original); d['template_status'] = 'fallback'; p.write_text(json.dumps(d))
        _rejects(lambda: check_release(root), 'official TMLR style')


def test_tmlr_overleaf_has_own_venue_and_retains_template_license():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / 'source'; root.mkdir(); _tmlr_fixture(root)
        package_release(root, Path(tmp) / 'dist')
        with zipfile.ZipFile(Path(tmp) / 'dist/overleaf.zip') as z:
            d = json.loads(z.read('submission_metadata.json'))
            assert d['target_venue'] == 'TMLR' and d['submission_ready'] is False
            assert 'TMLR_TEMPLATE_LICENSE.txt' in z.namelist()
            assert 'tmlr.sty' in z.namelist() and 'icml2026.sty' not in z.namelist()
            assert 'not anonymous review material' in z.read('OVERLEAF_README.txt').decode()


def test_tmlr_submission_ready_option_refuses_unrun_draft():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp); _tmlr_fixture(root)
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/package_release.py'),
                                 '--root', str(root), '--require-submission-ready'],
                                capture_output=True, text=True)
        assert result.returncode != 0
        assert 'preparation draft' in result.stderr
        assert not (root / 'dist').exists()


def test_tmlr_study_verifier_binds_sources_and_inherited_dependencies():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp); study = root / 'studies/context_generalization_v1'; study.mkdir(parents=True)
        (study / 'runner.py').write_text('pass\n')
        (root / 'base.py').write_text('fixed = True\n')
        d = {'protocol_id': 'test-only', 'files': {'runner.py': sha256(study / 'runner.py')},
             'dependencies': {'base.py': sha256(root / 'base.py')}}
        (study / 'source_manifest.json').write_text(json.dumps(d))
        assert verify_study(root)['dependencies'] == 1
        (root / 'base.py').write_text('fixed = False\n')
        try: verify_study(root)
        except ValueError as error: assert 'dependency mismatch' in str(error)
        else: raise AssertionError('changed inherited dependency accepted')


def test_tmlr_study_verifier_rejects_dependency_traversal():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp); study = root / 'studies/context_generalization_v1'; study.mkdir(parents=True)
        (study / 'runner.py').write_text('pass\n')
        d = {'files': {'runner.py': sha256(study / 'runner.py')}, 'dependencies': {'../outside.py': '0' * 64}}
        (study / 'source_manifest.json').write_text(json.dumps(d))
        try: verify_study(root)
        except ValueError as error: assert 'unsafe dependency path' in str(error)
        else: raise AssertionError('unsafe dependency accepted')


def test_tmlr_verifier_requires_every_declared_followup_source():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp); (root / 'release').mkdir()
        (root / 'release/release_manifest.json').write_text(json.dumps({
            'context_generalization_implementation_added': True,
            'spectral_extensions_implementation_added': True}))
        (root / 'base.py').write_text('fixed = True\n')
        for study_name in ('context_generalization_v1', 'spectral_extensions_v1'):
            folder = root / 'studies' / study_name; folder.mkdir(parents=True)
            (folder / 'run.py').write_text('pass\n')
            (folder / 'source_manifest.json').write_text(json.dumps({
                'files': {'run.py': sha256(folder / 'run.py')},
                'dependencies': {'base.py': sha256(root / 'base.py')}}))
        assert set(verify_all(root)) == {'context_generalization_v1', 'spectral_extensions_v1'}
        (root / 'studies/spectral_extensions_v1/run.py').write_text('changed = True\n')
        try: verify_all(root)
        except ValueError as error: assert 'source mismatch' in str(error)
        else: raise AssertionError('one declared study was silently skipped')
