"""Descriptive validation comparability; never selects a winner from test scores."""
from __future__ import annotations
import json
from pathlib import Path


def comparable(left, right, factor=1.25):
    if left is None or right is None or min(left,right)<0:
        return False
    # Absolute floor is fixed, not selected from outcomes.
    low,high=sorted((max(left,1e-12),max(right,1e-12)))
    return high/low<=factor


def _entries(payload):
    result=payload['result']
    if payload['job']['component']=='compact':
        for track, group in result['tracks'].items():
            for record in group['models']:
                val=record.get('validation_selected_predictor') or {}
                yield (track,record['name']), {
                    'validation_nmse':val.get('pooled_changed_pair_mean_nmse'),
                    'validation_output_squared_norm':val.get('pooled_mean_prediction_squared_norm_over_training_energy'),
                    'coefficient_frobenius_norm':record.get('fit',{}).get('coefficient_frobenius_norm') if record.get('fit',{}).get('family') in ('ridge','affine') else None,
                    'evaluation':{split:{k:v for k,v in value.items() if k!='actions'} for split,value in record.get('evaluation',{}).items()},
                    'storage':record.get('fit',{}).get('storage')}
    else:
        for record in result['methods']:
            val=(record.get('validation_selected_predictor') or {}).get('pooled',{})
            yield ('context',record['name']), {
                'validation_nmse':val.get('mean_normalized_squared_error'),
                'validation_output_squared_norm':val.get('mean_output_squared_norm_over_train_energy'),
                'coefficient_frobenius_norm':record.get('metadata',{}).get('coefficient_frobenius_norm') if record['name'] in ('linear_transport_context','coordinate_fitted_operator_context') else None,
                'evaluation':{split:value['pooled'] for split,value in record.get('splits',{}).items()},
                'storage':record.get('metadata',{}).get('storage')}


def make_comparison(output, factor=1.25):
    records={}
    for path in sorted((Path(output)/'checkpoints').glob('*.json')):
        payload=json.loads(path.read_text()); job=payload['job']
        for (track,method),entry in _entries(payload):
            records.setdefault((job['cell'],job['component'],track,method),{})[job['registration']]=entry
    pairs=[]
    for (cell,component,track,method),arms in records.items():
        semantic=arms.get('semantic'); alternative=arms.get('consistent_swap01')
        if semantic is None or alternative is None: continue
        error_match=comparable(semantic['validation_nmse'],alternative['validation_nmse'],factor)
        output_match=comparable(semantic['validation_output_squared_norm'],alternative['validation_output_squared_norm'],factor)
        ln=semantic['coefficient_frobenius_norm'];rn=alternative['coefficient_frobenius_norm']
        coefficient_match=comparable(ln,rn,factor) if ln is not None and rn is not None else None
        pairs.append({'cell':cell,'component':component,'track':track,'method':method,
                      'validation_error_comparable':error_match,'validation_output_norm_comparable':output_match,
                      'coefficient_norm_comparable_if_linear':coefficient_match,
                      'within_fixed_validation_tolerance':error_match and output_match and coefficient_match is not False,
                      'semantic':semantic,'consistent_swap01':alternative})
    return {'role':'exploratory_validation_comparability_ledger','factor_tolerance':factor,
            'absolute_floor':1e-12,'matching_uses_test_outcomes':False,
            'pairs':pairs,'pairs_within_tolerance':sum(p['within_fixed_validation_tolerance'] for p in pairs),
            'interpretation':'All methods remain reported, whether comparable or not. Output norm matching is not a nonlinear Jacobian/operator norm match. Consistent alternative action is a coherent labeled reference, not a randomized significance null. No new significance decision.',
            'paper_eligible':False,'confirmation_authorized':False}
