"""Numerical truth, leakage, dependence and resumability checks; no model loads."""
import io
import json
from pathlib import Path
import zipfile
import numpy as np
import pytest

from bsg_next.estimators import (fit_ridge, select_action_ridge, fit_procrustes,
    consistent_action_images, inclusive_reference_rank)
from bsg_next.statistics import (holm, family_decisions, crossed_summary,
    bootstrap_sensitivity, run_power,run_s3_power,crossed_variance_components,
    _crossed_vector_rejections,crossed_decisions)
from bsg_next.synthetic import (generate_dataset, known_estimands, run_synthetic,
    expected_linear_loss)
from bsg_next.structural import analyze_cell, fit_train_pca, action_reference
from bsg_next.offline import load_archive_cell, historical_replay


def test_zero_ridge_uses_x_pseudoinverse_and_output_df():
    rng=np.random.default_rng(1);x=rng.normal(size=(11,5));x[:,-1]=x[:,0]+x[:,1]
    y=rng.normal(size=(11,3));m=fit_ridge(x,y,0)
    assert np.allclose(m.coef,np.linalg.pinv(x,rcond=1e-12)@y)
    assert m.effective_dof==12


def test_regularization_scale_and_validation_only_generator_keys():
    x=np.eye(3);y=2*x
    m=fit_ridge(x,y,.1)
    assert m.alpha==pytest.approx(.1)
    maps,meta=select_action_ridge({'r':(x,y)},{'r':(x,y)},grid=[0,.1])
    assert set(maps)=={'r'} and meta['normalized_lambda']==0
    with pytest.raises(ValueError):select_action_ridge({'r':(x,y)},{'r':(x,y),'r2':(x,y)})


def test_rectangular_procrustes_and_train_pca_no_test_fit():
    rng=np.random.default_rng(2);x=rng.normal(size=(20,3));y=rng.normal(size=(20,5))
    assert fit_procrustes(x,y).coef.shape==(3,5)
    train=np.arange(20)<10;c,b,_=fit_train_pca(x,train,2)
    changed=x.copy();changed[~train]*=1000
    c2,b2,_=fit_train_pca(changed,train,2)
    assert np.array_equal(c,c2) and np.array_equal(b,b2)


def test_faithful_action_counts_and_inclusive_tie_mass():
    assert len(consistent_action_images('C7'))==120
    assert len(consistent_action_images('D4'))==3
    assert inclusive_reference_rank(1.,[1.,1.,2.])==pytest.approx(2/3)


def test_known_equivariant_construction_transports_actual_targets():
    ds=generate_dataset('D4','Y4',noise=0,condition=30,seed=4,train_families=3,validation_families=2,test_families=3)
    z=ds['z'];group=ds['group'];k=group.state_count
    for g,r in ds['true_operators'].items():
        for start in range(0,len(z),k):
            assert np.allclose(z[start:start+k]@r,z[start+np.asarray([group.apply(g,x) for x in range(k)])],atol=1e-10)


def test_category_code_passes_algebra_but_not_beyond_baseline():
    ds=generate_dataset('D4','Y1',noise=0,seed=3,train_families=3,validation_families=2,test_families=3)
    r=analyze_cell(ds['z'],ds['rows'],'D4',{'secondary':[],'lambda_grid':[0]})
    assert r['mean_losses']['primary_operator']<1e-20
    assert not r['eligible']
    assert max(abs(v) for v in r['family_orbit_prediction'].values())<1e-15
    ref=action_reference(ds['z'],ds['rows'],ds['group'],{'lambda_grid':[0]})
    assert ref['image_count']==3 and ref['inclusive_tail_rank']==1
    assert ref['semantic_p_value'] is False


def test_population_moment_oracle_matches_independent_monte_carlo():
    rng=np.random.default_rng(8);d=3;n=100000
    cov=np.array([[1,.3],[.3,2.]])
    raw=rng.multivariate_normal([0,0],cov,size=(n,d))
    x=raw[:,:,0]+1;y=raw[:,:,1]-2;b=rng.normal(size=(d,d))*.2;c=rng.normal(size=d)
    expected=expected_linear_loss(np.ones(d),-2*np.ones(d),np.eye(d),2*np.eye(d),.3*np.eye(d),b,c)
    measured=np.mean(np.sum((y-x@b-c)**2,axis=1))
    assert measured==pytest.approx(expected,rel=.01)


def test_known_estimand_distinguishes_estimator_truth_from_regime_label():
    ds=generate_dataset('C7','Y4',noise=.1,seed=11,train_families=4,validation_families=2,test_families=4)
    r=analyze_cell(ds['z'],ds['rows'],'C7',{'secondary':[],'lambda_grid':[0,.01]},return_models=True)
    truth=known_estimands(ds,r)
    assert np.isfinite(truth['pred']) and np.isfinite(truth['ctx'])
    assert 'conditional' in truth['truth_scope']


def test_local_infinite_bandwidth_is_state_conditional_mean_displacement():
    from bsg_next.estimators import fit_local_displacement
    rng=np.random.default_rng(65);x=rng.normal(size=(22,3));states=np.repeat([0,1],11)
    y=x+rng.normal(size=x.shape)+states[:,None]
    validation=(x[:4]+.1,y[:4]-.2,states[:4])
    model=fit_local_displacement(x,states,{'g':y},{'g':validation},chunk_size=2,max_scale_pairs=7)
    query=rng.normal(size=(5,3));labels=np.array([0,1,1,0,1])
    expected=query+np.stack([(y-x)[states==state].mean(0) for state in labels])
    assert np.allclose(model.predict(query,labels,'g',multiplier=np.inf),expected)
    assert all(v['sampled_pairs']==7 for v in model.metadata['states'].values())
    assert model.metadata['global_multiplier_across_actions']
    assert 0<model.metadata['smoother_trace_per_action']<=len(x)


def test_cached_local_uses_real_targets_without_changing_primary_or_fitting_test():
    from bsg_next.structural import pair_indices
    ds=generate_dataset('C7','Y4',noise=.1,seed=71,train_families=3,validation_families=2,test_families=3)
    z=ds['z'];rows=ds['rows'];config={'secondary':['local'],'lambda_grid':[0,.01],'local_chunk_size':2}
    result=analyze_cell(z,rows,'C7',config,return_models=True)
    primary=analyze_cell(z,rows,'C7',{'secondary':[],'lambda_grid':[0,.01]})
    assert result['primary_baseline']==primary['primary_baseline']
    assert result['family_orbit_prediction']==primary['family_orbit_prediction']
    assert result['family_orbit_context']==primary['family_orbit_context']
    local=result['_fitted']['local_displacement'];states=np.asarray([r['state_id'] for r in rows])
    actions={g:tuple(ds['group'].apply(g,s) for s in range(ds['group'].state_count)) for g in ds['group'].elements}
    errors=[]
    for g,(a,b) in pair_indices(rows,actions).items():
        mask=np.asarray([rows[i]['split']=='test' for i in a])&(states[a]!=states[b]);a,b=a[mask],b[mask]
        if len(a):errors.extend((np.sum((z[b]-local.predict(z[a],states[a],g))**2,1)/result['training_scale']).tolist())
    assert result['mean_losses']['local_displacement']==pytest.approx(np.mean(errors))
    assert len(result['secondary_family_orbit_prediction']['local_displacement_minus_primary'])==3
    changed=z.copy();changed[np.asarray([r['split']=='test' for r in rows])]+=np.arange(z.shape[1])*100
    other=analyze_cell(changed,rows,'C7',config,return_models=True)['_fitted']['local_displacement']
    assert other.metadata==local.metadata
    for state in local.sources:assert np.array_equal(other.sources[state],local.sources[state])
    for action in local.displacements:
        for state in local.sources:assert np.array_equal(other.displacements[action][state],local.displacements[action][state])


def test_holm_and_zero_energy_preserve_hypothesis_count():
    h=holm({'a':.01,'b':.04,'c':.2},.05)
    assert h['a']['p_adjusted']==pytest.approx(.03)
    assert h['a']['reject'] and not h['b']['reject']
    x=np.linspace(.1,.12,32)
    r=family_decisions({'C7':{'pred':x,'ctx':x,'eligible':False},'D4':{'pred':x,'ctx':x,'eligible':True}})
    assert r['test_count']==8 and r['contrasts']['C7.ctx']['status']=='uninformative'
    assert r['tests']['C7.ctx.superiority']['p']==1


def test_crossed_orbit_dependency_is_not_family_only():
    orbit=np.linspace(-.2,.2,16)
    table=np.tile(orbit,(32,1))+.02
    summary=crossed_summary(table)
    assert summary['df']==15
    assert summary['se']>0.02
    assert summary['se']==pytest.approx(orbit.std(ddof=1)/4)
    boot=bootstrap_sensitivity(table,replicates=100,seed=1,crossed=True)
    assert boot['sensitivity_only'] and boot['percentile_ci95'][0]<.02<boot['percentile_ci95'][1]


def test_synthetic_atomic_resume_and_identity_mismatch(tmp_path):
    config={'synthetic':{'train_families':3,'validation_families':2,'test_families':3,'lambda_grid':[0,.01]}}
    path=tmp_path/'run'
    a=run_synthetic(config,path,limit=1)
    seedfile=next((path/'seeds').glob('*.json'));before=seedfile.read_bytes()
    b=run_synthetic(config,path,limit=1)
    assert a['completed_datasets']==b['completed_datasets']==1
    assert seedfile.read_bytes()==before and a['no_inner_bootstrap']
    with pytest.raises(ValueError):run_synthetic({'synthetic':{'dimension':32}},path,limit=1)


def test_power_small_run_is_declared_assumption_not_measured_pilot(tmp_path):
    result=run_power({'power':{'replicates':16,'test_families':[8],'stress':['normal'],'correlations':[0.] }},tmp_path/'power')
    assert result['source']=='explicit_planning_assumptions_not_measured_pilot'
    assert result['selected_families'] is None


def test_archive_saved_operator_replay_does_not_refit(tmp_path):
    from bsg.data.c7 import build_dataset
    ds=generate_dataset('C7','Y1',noise=0,seed=9,train_families=3,validation_families=2,test_families=3)
    frame=build_dataset();z=np.tile(ds['mu'],(40,1));buffer=io.BytesIO()
    operators=np.stack([ds['true_operators'][g] for g in ds['group'].elements])
    np.savez(buffer,z=z,operators=operators,pca_components=np.eye(16),pca_mean=np.zeros(16),pca_explained=np.ones(16)/16)
    archive=tmp_path/'fixture.zip'
    with zipfile.ZipFile(archive,'w') as f:f.writestr('base.npz',buffer.getvalue())
    entry={'base_member':'base.npz','fields':{'group':'C7','latent_dim':16},
           'split':{'seed':101,'train_ids':list(range(28)),'validation_ids':list(range(28,34)),'test_ids':list(range(34,40))}}
    cell=load_archive_cell(archive,entry);rows=historical_replay(cell)
    assert len(rows)==12 and max(r['E_eq'] for r in rows)<1e-20
    assert cell['members']['base']['sha256']


def test_complete_cache_to_primary_report_and_secondary_rank_failure(tmp_path):
    from bsg_next.structural import run_structural
    from bsg_next.io import atomic_json,file_sha256
    from bsg_next.data import dataset_identity
    ds=generate_dataset('C7','Y4',noise=.1,seed=32,train_families=4,validation_families=2,test_families=3)
    cache=tmp_path/'cache';cache.mkdir()
    rows=[dict(r,domain='C7') for r in ds['rows']]
    (cache/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    np.save(cache/'H.npy',ds['z'].astype('float32'))
    atomic_json(cache/'manifest.json',{'schema_version':1,'status':'complete','domain':'C7','model':{'id':'fixture'},
        'n_rows':len(rows),'dataset_sha256':dataset_identity(rows),'rows_path':'rows.jsonl',
        'rows_sha256':file_sha256(cache/'rows.jsonl'),'layers':[0],
        'arrays':{'0':{'path':'H.npy','shape':list(ds['z'].shape),'dtype':'float32','sha256':file_sha256(cache/'H.npy')}}})
    config={'models':[{'id':'fixture','layers':[0]}],
            'structural':{'groups':['C7'],'q_primary':16,'q_secondary':[32],'secondary':['local'],'lambda_grid':[0,.01]}}
    result=run_structural(config,tmp_path/'out',cache_root=cache)
    assert result['panel_complete'] and result['cells']==2
    assert len(result['family_contrasts']['C7']['pred'])==3
    assert len(result['secondary_local_displacement']['C7']['family_orbit_local_minus_primary'])==3
    assert result['secondary_local_displacement']['C7']['formal_inference'] is False
    reports=[json.loads(p.read_text()) for p in (tmp_path/'out'/'cells').glob('*.json')]
    assert any(r[0].get('status')=='secondary_rank_failure' for r in reports)


def test_s3_zero_shot_requires_potential_features_not_test_centroids():
    ds=generate_dataset('C7','Y4',noise=.1,seed=3,train_families=3,validation_families=2,test_families=3)
    rows=[r for r in ds['rows'] if r['state_id']<6]
    z=np.asarray([v for v,r in zip(ds['z'],ds['rows']) if r['state_id']<6])
    with pytest.raises(ValueError,match='potential features'):
        analyze_cell(z,rows,'S3',{'secondary':[]})


def test_s3_power_uses_crossed_pilot_and_actual_primary_decisions(tmp_path):
    rng=np.random.default_rng(50)
    x=rng.normal(size=(12,1,2))*.03+rng.normal(size=(1,8,2))*.04+rng.normal(size=(12,8,2))*.01
    components,meta=crossed_variance_components(x)
    assert meta['pilot_orbit_clusters']==8 and np.trace(components['orbit'])>0
    data=np.stack([x,x+.1])
    vectorized=_crossed_vector_rejections(data)
    for i,table in enumerate(data):
        reference=crossed_decisions({'pred':table[:,:,0],'ctx':table[:,:,1]})
        expected=[reference['tests'][k]['reject'] for k in ['pred.superiority','ctx.superiority','pred.equivalence','ctx.equivalence']]
        assert np.array_equal(vectorized[i],expected)
    result=run_s3_power({'power':{'replicates':16,'test_families':[8],'test_orbits':[4],
                       'stress':['normal'],'s3_correlations':['estimated']}},tmp_path/'power',pilot_crossed=x)
    assert result['pilot_diagnostics']['pilot_family_clusters']==12
    assert result['scenarios'][0]['df']==3 and result['selected_families'] is None


def test_strict_gate_rejects_reduced_scientific_grid_and_technical_run(tmp_path):
    from bsg_next.gates import _expected,validate_calibration,CalibrationValidationError
    assert len(_expected({},'calibration'))==14400
    assert len(_expected({},'negative'))==60000
    with pytest.raises(CalibrationValidationError,match='5000'):
        _expected({'synthetic':{'negative_replicates':2}},'negative')
    with pytest.raises(CalibrationValidationError,match='reduced'):
        _expected({'synthetic':{'groups':['C7']}},'calibration')
    with pytest.raises(CalibrationValidationError,match='Technical'):
        validate_calibration({'technical_test':True},tmp_path,tmp_path,tmp_path)


def test_crossed_bootstrap_subset_is_executable_and_resumable(tmp_path):
    config={'technical_test':True,'synthetic':{'bootstrap_replicates':19,'crossed_bootstrap_datasets':2}}
    result=run_synthetic(config,tmp_path/'subset',mode='crossed_bootstrap',limit=1)
    assert result['sensitivity_only'] and result['completed_datasets']==1
    assert not result['all_complete'] and result['resamples_per_dataset']==19
    again=run_synthetic(config,tmp_path/'subset',mode='crossed_bootstrap',limit=1)
    assert again['cases']==result['cases']
