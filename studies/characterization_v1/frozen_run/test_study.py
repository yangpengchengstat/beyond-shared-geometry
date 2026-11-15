"""CPU-only engineering fixtures; no research holdout or model execution."""
from pathlib import Path
import json
import sys
import tempfile
from unittest.mock import patch
import numpy as np


def fixture(domain):
    from bsg_next.groups import get_group
    group=get_group(domain);rng=np.random.default_rng(771)
    rows=[];vectors=[]
    for split,count in [('train',8),('validation',4),('pilot',4),('test',4)]:
        for f in range(count):
            for p in range(2):
                context=rng.normal(size=4)
                for s in range(group.state_count):
                    rid=f'{domain}/{split}/{f}/{p}/{s}'
                    rows.append({'row_id':rid,'record_id':rid,'family_id':f'fixture_{split}_{f}',
                        'paraphrase_id':p,'orbit_id':'fixture_one','state_id':s,'split':split,
                        'domain':group.name,'text':'synthetic fixture '+rid})
                    vectors.append(np.r_[np.eye(group.state_count)[s],context]+rng.normal(0,.01,group.state_count+4))
    return group,rows,np.array(vectors)


def run_tests():
    from fields import self_test,compare_cell,input_comparisons,farthest_first
    from bsg_next.structural import analyze_cell
    from run_study import merge_resumable,digest,selected_test_rows,atomic,read
    self_test()
    interval_records=[];field_records=[]
    assert farthest_first(np.array([[0.],[2.],[-2.],[1.]]),3).tolist()==[0,1,2]
    assert farthest_first(np.ones((6,2)),4).tolist()==[0,1,2,3]
    for domain in ('C7','D4_corner'):
        group,rows,z=fixture(domain)
        cfg={'lambda_grid':[0.,.001,.01],'secondary':['generator','local'],'legal_action_reference':False}
        legacy=analyze_cell(z,rows,group,cfg,test_split='test')
        gen=compare_cell(z,rows,group,cfg,evaluation_split='test')
        assert np.isclose(legacy['mean_losses']['generator_unseen_actual_target'],gen['mean_losses']['unseen_composite']['generator_ridge'])
        result=input_comparisons(z,rows,group,cfg)
        interval_records.append(legacy)
        field_records.append({'group':group.name,'generator':gen,'input_comparisons':result})
        for control,old in [('primary_operator','primary_operator'),('oracle_local','local_displacement')]:
            assert np.isclose(result['controls'][control]['mean_loss'],legacy['mean_losses'][old])
        full=next(c for c in result['storage_curves'] if c['track']=='generator_only' and c['full_training_support'])
        for kind in ('primitive','unseen_composite'):
            assert np.isclose(full['results'][kind]['mean_loss'],gen['mean_losses'][kind]['generator_input_only_field'])
        # Changing every test coordinate must not change training/validation choices.
        altered=z.copy();altered[[r['split']=='test' for r in rows]]+=15
        changed=input_comparisons(altered,rows,group,cfg)
        assert changed['primary_selection']==result['primary_selection']
        for a,b in zip(result['storage_curves'],changed['storage_curves']):
            assert a['selection']==b['selection'] and a['selected_training_indices']==b['selected_training_indices']
            actions=len(group.elements) if a['track']=='all_element' else len(group.generators)
            assert a['counted_float64_coordinates']==a['K']*z.shape[1]*(actions+1)
            if a['K']==a['coordinate_storage_matched_budget_K']:
                assert a['counted_predictor_array_bytes']<=a['linear_map_array_bytes']
        bad=[dict(r) for r in rows];bad[-1]['family_id']=bad[0]['family_id']
        try:compare_cell(z,bad,group,cfg,evaluation_split='test')
        except ValueError:pass
        else:raise AssertionError('Partition leak accepted')
    from uncertainty import summarize
    s3={'group':'S3_role','test_split':'test_joint','eligible':True,
        'family_orbit_prediction':{f'f{f}|o{o}':.25 for f in range(3) for o in range(2)},
        'family_orbit_context':{f'f{f}|o{o}':.5 for f in range(3) for o in range(2)}}
    intervals=summarize(interval_records+[s3],field_records,expected_conditions=1,replicates=100,seed=9)
    assert intervals['legacy_contrasts']['S3_role|test_joint']['prediction']['sensitivity_interval']['percentile_ci95']==[.25,.25]
    assert intervals['storage_curve_contrasts'] and not intervals['multiple_comparison_adjustment']
    # Separate toy seed: verifies selection/block counts without opening actual sealed prompts.
    from bsg_next.data import build_study_records,build_support_records
    config=json.loads((Path(__file__).parent/'study_config.json').read_text())
    for domain in ('C7','D4','S3'):
        records=build_study_records(domain,seed=91,max_test_families=64,max_test_orbits=24)
        if domain=='S3':records+=build_support_records(domain,seed=91,test_orbits=24)
        assert len(selected_test_rows(records,domain,config))==config['all_test_rows_per_model'][domain]
    from bsg_next.extraction import extract_features,read_cache
    group,rows,z=fixture('C7');values={r['row_id']:z[i].astype(np.float32) for i,r in enumerate(rows)}
    class FakeRuntime:
        telemetry=[]
        fail=False
        def identity(self):return {'model_id':'SYNTHETIC_FIXTURE_ONLY'}
        def features(self,records,layers,readout):
            for begin in range(0,len(records),16):
                batch=records[begin:begin+16]
                yield list(range(begin,begin+len(batch))),{l:np.stack([values[r['row_id']] for r in batch]) for l in layers}
                if self.fail:raise RuntimeError('Injected interruption after one durable batch')
    with tempfile.TemporaryDirectory() as temp:
        p=Path(temp);runtime=FakeRuntime();runtime.fail=True
        atomic(p/'legacy_fixture.json',legacy)
        assert read(p/'legacy_fixture.json')['group']=='D4_corner'
        trainrows=[r for r in rows if r['split']!='test'];testrows=[r for r in rows if r['split']=='test']
        try:extract_features(runtime,trainrows,[1],'span_mean',p/'old')
        except RuntimeError:pass
        else:raise AssertionError('Expected injected interruption')
        assert (p/'old/progress.json').exists() and not (p/'old/manifest.json').exists()
        runtime.fail=False
        extract_features(runtime,trainrows,[1],'span_mean',p/'old')
        extract_features(runtime,testrows,[1],'span_mean',p/'new')
        before={str(f.relative_to(p)):digest(f) for f in p.rglob('*') if f.is_file()}
        merge_resumable(p/'old',p/'new',p/'merged')
        merged,arrays,m=read_cache(p/'merged')
        assert [r['row_id'] for r in merged]==[r['row_id'] for r in trainrows+testrows]
        assert np.array_equal(arrays[1],np.stack([values[r['row_id']] for r in merged]))
        merge_resumable(p/'old',p/'new',p/'merged')
        for name,sha in before.items():assert digest(p/name)==sha
        original=extract_features
        def interrupt_copy(runtime,records,layers,readout,directory,**kw):
            actual=runtime.features
            def broken(*a,**k):
                for value in actual(*a,**k):
                    yield value
                    raise RuntimeError('Injected interrupted merge')
            runtime.features=broken
            return original(runtime,records,layers,readout,directory,**kw)
        with patch('bsg_next.extraction.extract_features',side_effect=interrupt_copy):
            try:merge_resumable(p/'old',p/'new',p/'partial_merge')
            except RuntimeError:pass
            else:raise AssertionError('Expected partial merge')
        merge_resumable(p/'old',p/'new',p/'partial_merge')
        rr,aa,_=read_cache(p/'partial_merge')
        assert np.array_equal(aa[1],arrays[1]) and rr==merged
        # A completed extraction hit must not call the runtime's feature generator.
        with patch.object(runtime,'features',side_effect=AssertionError('Cache hit ran extraction')):
            extract_features(runtime,trainrows,[1],'span_mean',p/'old')
    assert 'torch' not in sys.modules and 'transformers' not in sys.modules
    print('STUDY_CPU_TESTS_OK: test-outcome-blind selection, legacy replay, storage accounting, toy block counts, interrupted extraction and merge resume, exact merge, unchanged inputs, model-free cache hits.',flush=True)


if __name__=='__main__':run_tests()
