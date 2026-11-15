"""Conditional product/interface implementation tests using disclosed artificial features."""
import hashlib
import numpy as np
import pytest
from bsg_next.extensions import ProductGroup,build_product_dataset,run_product,_train_interface


def test_product_generators_are_independent_commuting_bijections():
    group=ProductGroup();r,s=group.generators
    assert len(set(group.elements))==16
    assert {group.apply(r,x) for x in range(16)}==set(range(16))
    assert group.compose(r,s)==group.compose(s,r)
    assert group.evaluate_word([r]*4)==group.identity
    assert group.evaluate_word([s]*4)==group.identity
    assert group.evaluate_word([r,s])!=group.evaluate_word([r,r])
    for a in group.elements:
        for b in group.elements:
            for x in range(16):assert group.apply(a,group.apply(b,x))==group.apply(group.compose(a,b),x)


def test_product_complete_orbit_augmentation_is_duplication():
    records=build_product_dataset(splits={'train':2,'validation':2,'test':2})
    assert len(records)==6*2*16
    group=ProductGroup()
    for g in group.elements:assert {group.apply(g,x) for x in range(16)}==set(range(16))
    assert len({r['family_id'] for r in records})==6
    assert len({r['text'] for r in records})==len(records)


class ArtificialFeatureRuntime:
    """Test-only feature source, never presented as a pretrained model."""
    def identity(self):return {'id':'artificial_test_runtime','revision':'fixture'}
    def features(self,records,layers,readout):
        values=[]
        for row in records:
            x=np.zeros(20);x[row['state_id']]=1
            seed=int(hashlib.sha256(row['family_id'].encode()).hexdigest()[:8],16)
            x[16:]=np.random.default_rng(seed+row['paraphrase_id']).normal(size=4)
            values.append(x)
        yield list(range(len(records))),{layer:np.asarray(values) for layer in layers}


def test_conditional_product_cached_pipeline_and_primary_scope(tmp_path):
    config={'models':[{'id':'artificial_test_runtime','layers':[0]}],
            'product':{'splits':{'train':2,'validation':2,'test':2},'q':4},
            'structural':{'secondary':[],'lambda_grid':[0,.01]}}
    result=run_product(config,tmp_path/'run',runtime_factory=lambda spec:ArtificialFeatureRuntime())
    assert result['cells']==1 and result['rows']==192
    assert result['results'][0]['not_primary_C7_D4_S3_family']
    assert np.isfinite(result['results'][0]['mean_losses']['primary_operator'])


def test_constructive_four_arms_train_equal_steps_on_actual_targets():
    torch=pytest.importorskip('torch');previous=torch.get_num_threads();torch.set_num_threads(1)
    try:
        rows=build_product_dataset(splits={'train':2,'validation':2,'test':2})
        runtime=ArtificialFeatureRuntime();h=next(runtime.features(rows,[0],'final_token'))[1][0]
        arms=['lowrank_reconstruction','complete_orbit_augmentation','equivariance_consistency','group_regularized']
        outputs=[_train_interface(h,rows,ProductGroup(),4,arm,.1,101,steps=4,batch_size=16) for arm in arms]
        assert {r['steps'] for r in outputs}=={4}
        assert all(np.isfinite(r['test']['test']['target_prediction_loss']) for r in outputs)
        assert all(r['test']['test']['target_rows']>0 for r in outputs)
        assert outputs[1]['augmentation_disclosure']
        assert all('not native' in r['claim'] for r in outputs)
    finally:torch.set_num_threads(previous)
