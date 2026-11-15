"""Software checks for a descriptive renderer, not a repeated research experiment."""
import importlib.util
from pathlib import Path
import tempfile
import shutil
import json

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('replay_assets',ROOT/'scripts/build_reproduction_assets.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def test_replay_renderer_rejects_duplicate_and_missing_cells():
    rows=[{'cell':str(i),'domain':'C7','nmse':float(i),'invalid':0} for i in range(24)]
    assert module.aggregate(rows,['domain'],'nmse',24)[0]['mean']==11.5
    for invalid in (rows[:-1],rows[:-1]+[rows[0]],rows[:-1]+[{**rows[-1],'nmse':float('nan')}],rows[:-1]+[{**rows[-1],'invalid':1}]):
        try:module.aggregate(invalid,['domain'],'nmse',24)
        except ValueError:pass
        else:raise AssertionError('Invalid evidence accepted')

def test_replay_review_rejects_inference_upgrade_and_altered_input():
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);(root/'release').mkdir()
        shutil.copytree(ROOT/'results/reproduction_v1_review',root/'results/reproduction_v1_review')
        shutil.copytree(ROOT/'results/reproduction_action_review',root/'results/reproduction_action_review')
        review=json.loads((ROOT/'release/reproduction_review.json').read_text())
        (root/'release/reproduction_review.json').write_text(json.dumps(review))
        module.load_review(root)
        review['independent_confirmation']=True
        (root/'release/reproduction_review.json').write_text(json.dumps(review))
        try:module.load_review(root)
        except ValueError:pass
        else:raise AssertionError('False scientific promotion accepted')
        review['independent_confirmation']=False
        (root/'release/reproduction_review.json').write_text(json.dumps(review))
        next((root/'results/reproduction_v1_review').glob('compact_rows.json')).write_text('[]')
        try:module.load_review(root)
        except ValueError:pass
        else:raise AssertionError('Changed curated data accepted')
