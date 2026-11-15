"""Audited field estimators, explicit evaluation split and storage-curve extension.

Core generator code derived from compare_cached_generator_fields.py SHA256
bd0a8cec68d58ce7add26d63420711dc064ca055c89de6c5fb9cb3c29f9c508a.
Only the validation whitelist and explicit evaluation-split parameter change
in those reference routines. Old files and results remain unchanged.
"""
from collections import defaultdict
import math
import sys
BANDWIDTHS=(.125,.25,.5,1.,2.,4.,8.,float('inf'))
METHODS=('generator_ridge','generator_input_only_field')

class InputOnlyField:
    """Kernel displacement regression on paired training arrays; no state labels."""
    def __init__(self, x, targets):
        import numpy as np
        self.x = np.asarray(x, float).copy()
        self.delta = {g: np.asarray(y, float).copy() - self.x for g, y in targets.items()}
        assert all(v.shape == self.x.shape for v in self.delta.values())
        n = len(x); count = n * (n - 1) // 2
        ranks = np.unique(np.linspace(0, count - 1, min(4096, count), dtype=np.int64))
        starts = np.arange(n, dtype=np.int64) * (2 * n - np.arange(n) - 1) // 2
        left = np.searchsorted(starts, ranks, side='right') - 1
        right = left + 1 + ranks - starts[left]
        distances = np.linalg.norm(self.x[left] - self.x[right], axis=1)
        positive = distances[distances > 1e-12]
        self.scale = float(np.median(positive)) if len(positive) else 0.
        self.multiplier = float('inf')
        self.selection = {}

    def predict(self, query, action, multiplier=None):
        import numpy as np
        query = np.asarray(query, float)
        multiplier = self.multiplier if multiplier is None else multiplier
        result = query.copy()
        for start in range(0, len(query), 64):
            x = query[start:start + 64]
            if math.isinf(multiplier) or self.scale <= 0:
                weights = np.full((len(x), len(self.x)), 1 / len(self.x))
            else:
                distances = np.maximum(np.sum(x*x, 1)[:, None] + np.sum(self.x*self.x, 1)[None] - 2*x@self.x.T, 0.)
                logits = -distances / (2 * (self.scale * multiplier)**2)
                weights = np.exp(logits - logits.max(1, keepdims=True))
                weights /= weights.sum(1, keepdims=True)
            result[start:start + len(x)] += weights @ self.delta[action]
        return result

    def select(self, validation):
        import numpy as np
        scores = [float(np.mean([np.mean((self.predict(x, g, b) - y)**2)
                                 for g, (x, y) in validation.items()])) for b in BANDWIDTHS]
        best = min(scores)
        chosen = max(i for i, v in enumerate(scores) if v <= best + 1e-8)
        self.multiplier = BANDWIDTHS[chosen]
        self.selection = {'bandwidth_grid': ['inf' if math.isinf(b) else b for b in BANDWIDTHS],
                          'validation_losses': scores, 'selected': 'inf' if math.isinf(self.multiplier) else self.multiplier,
                          'scale': self.scale, 'fit_scope': 'paired_training_arrays_only',
                          'source_state_labels_in_fit_or_prediction': False,
                          'selection_scope': 'changed_validation_actions_only',
                          'retained_displacement_coordinates': sum(v.size for v in self.delta.values())}

def validate_rows(rows):
    splits = {r['split'] for r in rows}
    if not {'train', 'validation'}.issubset(splits) or not splits.issubset({'train', 'validation', 'pilot', 'test'}):
        raise ValueError('Expected disjoint train, validation and declared pilot/test partitions')
    families = {s: {r['family_id'] for r in rows if r['split'] == s} for s in splits}
    if any(families[a] & families[b] for a in splits for b in splits if a != b):
        raise ValueError('A family crosses partitions')

def word_catalog(group):
    """Freeze shortest positive-generator words, then explicit diagnostic words."""
    from bsg_next.structural import _generator_words
    words = _generator_words(group)
    assert set(words) == set(group.elements)
    for g, word in words.items():
        assert all(token in group.generators for token in word)
        assert group.evaluate_word(word) == g
    diagnostics = {'r7': (group.generators[0],)*7} if group.name == 'C7' else {
        'r4': (group.generators[0],)*4, 's2': (group.generators[1],)*2,
        'srs': (group.generators[1], group.generators[0], group.generators[1]),
        'rsr': (group.generators[0], group.generators[1], group.generators[0]),
        'rs': group.generators, 'sr': tuple(reversed(group.generators))}
    return words, diagnostics

def compose_prediction(x, word, predict, generators):
    value = x.copy()
    for token in word:
        if token not in generators:
            raise ValueError('Composition may only call fitted primitive predictors')
        value = predict(value, token)
    return value

def fit_generators(x, train_pairs, validation, group, grid):
    from bsg_next.estimators import select_action_ridge
    if set(train_pairs) != set(group.generators) or set(validation) != set(group.generators):
        raise ValueError('Generator-only fitting rejects any missing or nonprimitive action')
    ridge, selection = select_action_ridge(train_pairs, validation, grid)
    field = InputOnlyField(x, {g: train_pairs[g][1] for g in group.generators})
    field.select(validation)
    return ridge, selection, field

def compare_cell(z, rows, group, config, evaluation_split="pilot"):
    """Every selection uses primitive validation pairs, never composite targets."""
    import numpy as np
    from bsg_next.structural import pair_indices
    validate_rows(rows)
    train = np.array([r['split']=='train' for r in rows])
    val = np.array([r['split']=='validation' for r in rows])
    if evaluation_split not in ('pilot','test'):
        raise ValueError('Expected explicit pilot or test evaluation split')
    pilot = np.array([r['split']==evaluation_split for r in rows])
    if not pilot.any():
        raise ValueError('Evaluation split is empty')
    states = np.array([int(r['state_id']) for r in rows])
    indices = pair_indices(rows, {g: tuple(group.apply(g,s) for s in range(group.state_count)) for g in group.elements})
    train_pairs = {}; validation = {}
    for g in group.generators:
        a, b = indices[g]
        assert np.array_equal(a[train[a]], np.flatnonzero(train))
        train_pairs[g] = (z[a[train[a]]], z[b[train[a]]])
        mask = val[a] & (states[a]!=states[b])
        validation[g] = (z[a[mask]], z[b[mask]])
    ridge, selection, field = fit_generators(z[train], train_pairs, validation, group, config['lambda_grid'])
    predict = {'generator_ridge': lambda x,g: ridge[g].predict(x),
               'generator_input_only_field': lambda x,g: field.predict(x,g)}
    words, diagnostics = word_catalog(group)
    scale = float(np.mean(np.sum((z[train]-z[train].mean(0))**2,1)))
    if not math.isfinite(scale) or scale <= 1e-12:
        raise ValueError('Invalid training activation energy')
    pooled = {kind: {m: [] for m in METHODS} for kind in ['primitive','unseen_composite']}
    families = {kind: {m: defaultdict(list) for m in METHODS} for kind in pooled}
    action_results = []
    for g in group.elements:
        if g == group.identity:
            continue
        a,b = indices[g]; mask = pilot[a] & (states[a]!=states[b]); a,b = a[mask],b[mask]
        if not len(a):
            continue
        kind = 'primitive' if g in group.generators else 'unseen_composite'
        losses = {}
        for method in METHODS:
            prediction = compose_prediction(z[a],words[g],predict[method],group.generators)
            values = np.sum((prediction-z[b])**2,1)/scale
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite composed prediction; cell retained as error')
            losses[method] = float(np.mean(values)); pooled[kind][method].extend(values.tolist())
            for i,v in zip(a,values):
                families[kind][method][rows[i]['family_id']].append(float(v))
        action_results.append({'element': list(g) if isinstance(g,tuple) else g,
                               'word': list(words[g]), 'kind': kind, 'n_pairs': len(a), 'losses': losses})
    diagnostic_results = {}
    for name,word in diagnostics.items():
        element = group.evaluate_word(word); a,b = indices[element]; a,b = a[pilot[a]],b[pilot[a]]
        diagnostic_results[name] = {'word': list(word),'element': list(element) if isinstance(element,tuple) else element,
                                    'n_pairs': len(a),'mean_losses':{}}
        for method in METHODS:
            output = compose_prediction(z[a],word,predict[method],group.generators)
            diagnostic_results[name]['mean_losses'][method] = float(np.mean(np.sum((output-z[b])**2,1)/scale))
    order_tests = {}
    if group.name == 'D4_corner':
        for name,right,wrong in [('relation_srs_vs_rsr','srs','rsr'),('order_rs_vs_sr','rs','sr')]:
            right_element = group.evaluate_word(diagnostics[right]); wrong_element = group.evaluate_word(diagnostics[wrong])
            a,b = indices[right_element]
            mask = pilot[a] & np.array([group.apply(right_element,int(s)) != group.apply(wrong_element,int(s)) for s in states[a]])
            a,b = a[mask],b[mask]
            if not len(a):
                raise ValueError('Order diagnostic has no endpoint-discriminating cases')
            order_tests[name] = {'n_pairs':len(a),'coincident_endpoints_excluded':True,'methods':{}}
            for method in METHODS:
                values = {}
                for label,word_name in [('correct',right),('wrong',wrong)]:
                    pred = compose_prediction(z[a],diagnostics[word_name],predict[method],group.generators)
                    values[label] = float(np.mean(np.sum((pred-z[b])**2,1)/scale))
                order_tests[name]['methods'][method] = {**values,'correct_minus_wrong':values['correct']-values['wrong']}
    means = {kind:{m:float(np.mean(v)) for m,v in methods.items()} for kind,methods in pooled.items()}
    return {'mean_losses':means,'per_action':action_results,
            'family_losses':{kind:{m:{f:float(np.mean(v)) for f,v in fs.items()} for m,fs in methods.items()} for kind,methods in families.items()},
            'diagnostics':diagnostic_results,'order_tests':order_tests,'ridge_selection':selection,
            'kernel_selection':field.selection,
            'storage':{'ridge_matrix_coordinates':sum(v.coef.size for v in ridge.values()),
                       'kernel_source_coordinates':field.x.size,
                       'kernel_displacement_coordinates':sum(v.size for v in field.delta.values()),
                       'storage_not_independent_parameter_count':True},
            'fit_action_count':len(group.generators),'training_scale':scale,
            'intermediate_true_state_used':False,'intermediate_projection_or_snapping':False}

def self_test():
    import numpy as np
    from bsg_next.groups import get_group
    from bsg_next.structural import analyze_cell
    for domain in ['C7','D4_corner']:
        group = get_group(domain); eye = np.eye(group.state_count)
        matrices = {g: eye[[group.apply(g,s) for s in range(group.state_count)]] for g in group.generators}
        field = InputOnlyField(eye, {g:eye@matrix for g,matrix in matrices.items()});field.multiplier = .125
        words,diagnostics = word_catalog(group)
        for word in list(words.values())+list(diagnostics.values()):
            expected = eye[[group.apply_word(s,word) for s in range(group.state_count)]]
            for predictor in [lambda x,g:x@matrices[g],lambda x,g:field.predict(x,g)]:
                assert np.allclose(compose_prediction(eye,word,predictor,group.generators),expected,atol=1e-10)
        try:
            compose_prediction(eye,(next(g for g in group.elements if g not in group.generators and g!=group.identity),),lambda x,g:x,group.generators)
        except ValueError:
            pass
        else:
            raise AssertionError('Nonprimitive call accepted')
        primitive_pairs={g:(eye,eye@matrix) for g,matrix in matrices.items()}
        leaked=dict(primitive_pairs)
        leaked[next(g for g in group.elements if g not in group.generators and g!=group.identity)]=(eye,eye)
        try:
            fit_generators(eye,leaked,leaked,group,[0.])
        except ValueError:
            pass
        else:
            raise AssertionError('Unseen-element fitting pair accepted')
        # Compare the new actual-target aggregation against the existing generator
        # implementation on disclosed model-free coordinates, not LLM measurements.
        rng = np.random.default_rng(42); rows=[]; zs=[]
        for split,count in [('train',8),('validation',4),('pilot',4)]:
            for family in range(count):
                for paraphrase in range(2):
                    context = rng.normal(size=10)
                    for state in range(group.state_count):
                        rows.append({'split':split,'family_id':f'{split}_{family}', 'paraphrase_id':paraphrase,
                                     'orbit_id':'one','state_id':state})
                        zs.append(np.r_[eye[state],context] + rng.normal(scale=.01,size=group.state_count+10))
        z = np.asarray(zs); controls={'lambda_grid':[0.,.001,.01], 'secondary':['generator'], 'legal_action_reference':False}
        reference = analyze_cell(z,rows,group,controls,test_split='pilot')
        measured = compare_cell(z,rows,group,controls)
        assert np.isclose(measured['mean_losses']['unseen_composite']['generator_ridge'], reference['mean_losses']['generator_unseen_actual_target'],atol=1e-10)
        assert measured['fit_action_count']==len(group.generators)
        if domain=='D4_corner':
            assert measured['order_tests']['relation_srs_vs_rsr']['n_pairs']==4*2*2
            assert measured['order_tests']['order_rs_vs_sr']['n_pairs']==4*2*4
    assert 'torch' not in sys.modules and 'transformers' not in sys.modules
    print('SELF_TEST_OK: exact C7/D4 actions and word order for both predictors; primitive-only guard; legacy generator-target replay; coincident-endpoint exclusion; no model libraries loaded.',flush=True)


def farthest_first(x, count):
    """Unlabelled training-only support selection; ties use input row order."""
    import numpy as np
    count = min(int(count), len(x))
    if count == len(x):
        return np.arange(len(x))  # Preserve audited full-field scale sampling.
    chosen = [0]
    distance = np.sum((x-x[0])**2, axis=1)
    distance[0] = -np.inf
    while len(chosen) < count:
        index = int(np.argmax(distance)); chosen.append(index)
        distance = np.minimum(distance, np.sum((x-x[index])**2, axis=1))
        distance[chosen] = -np.inf
    return np.asarray(chosen, dtype=int)


def input_comparisons(z, rows, group, config, evaluation_split='test'):
    """Strong all-element controls plus separately labelled storage curves."""
    import copy
    import numpy as np
    from bsg_next.structural import pair_indices, _fit_actions
    from bsg_next.estimators import state_centroids, decode_states, fit_local_displacement
    validate_rows(rows)
    train=np.array([r['split']=='train' for r in rows])
    val=np.array([r['split']=='validation' for r in rows])
    evaluate=np.array([r['split']==evaluation_split for r in rows])
    if not evaluate.any(): raise ValueError('Empty evaluation partition')
    states=np.array([r['state_id'] for r in rows])
    indices=pair_indices(rows,{g:tuple(group.apply(g,s) for s in range(group.state_count)) for g in group.elements})
    targets={g:z[b[train[a]]] for g,(a,b) in indices.items()}
    for a,b in indices.values():
        assert np.array_equal(a[train[a]],np.flatnonzero(train))
    validation={}; oracle_validation={}
    for g,(a,b) in indices.items():
        mask=val[a]&(states[a]!=states[b]);a,b=a[mask],b[mask]
        if len(a):
            validation[g]=(z[a],z[b]);oracle_validation[g]=(z[a],z[b],states[a])
    primary,primary_selection=_fit_actions(z,rows,indices,config['lambda_grid'])
    x=z[train]; energy=float(np.mean(np.sum((x-x.mean(0))**2,axis=1)))
    mu=state_centroids(z,states,train);decoded=decode_states(z,mu)
    local=fit_local_displacement(x,states[train],targets,oracle_validation)
    scores=[float(np.mean([np.mean((local.predict(a,decode_states(a,mu),g,b)-y)**2)
                for g,(a,y) in validation.items()])) for b in BANDWIDTHS]
    selected=max(i for i,v in enumerate(scores) if v<=min(scores)+1e-8)
    tuned=copy.copy(local);tuned.multiplier=BANDWIDTHS[selected]
    controls={m:defaultdict(list) for m in ('primary_operator','oracle_local',
              'decoded_local_same_bandwidth','decoded_local_retuned')}
    for g,(a,b) in indices.items():
        mask=evaluate[a]&(states[a]!=states[b]);a,b=a[mask],b[mask]
        if not len(a):continue
        pp={'primary_operator':primary[g].predict(z[a]),'oracle_local':local.predict(z[a],states[a],g),
            'decoded_local_same_bandwidth':local.predict(z[a],decoded[a],g),
            'decoded_local_retuned':tuned.predict(z[a],decoded[a],g)}
        for method,pred in pp.items():
            loss=np.sum((pred-z[b])**2,axis=1)/energy
            for index,value in zip(a,loss):controls[method][rows[index]['family_id']].append(float(value))
    def pack(values):
        return {'mean_loss':float(np.mean([v for fs in values.values() for v in fs])),
                'family_losses':{f:float(np.mean(v)) for f,v in values.items()}}
    words,_=word_catalog(group);curves=[]
    for track in ('all_element','generator_only'):
        actions=list(group.elements) if track=='all_element' else list(group.generators)
        matched=int(len(actions)*z.shape[1]/(len(actions)+1))
        budgets=sorted({min(len(x),k) for k in (8,16,32,64,128,256,len(x),matched)})
        for k in budgets:
            support=farthest_first(x,k)
            field=InputOnlyField(x[support],{g:targets[g][support] for g in actions})
            field.select({g:xy for g,xy in validation.items() if g in actions})
            values={kind:defaultdict(list) for kind in ('all_changed',) if track=='all_element'}
            if track=='generator_only':values={kind:defaultdict(list) for kind in ('primitive','unseen_composite')}
            for g,(a,b) in indices.items():
                mask=evaluate[a]&(states[a]!=states[b]);a,b=a[mask],b[mask]
                if not len(a):continue
                if track=='all_element':pred=field.predict(z[a],g);kind='all_changed'
                else:
                    pred=compose_prediction(z[a],words[g],field.predict,group.generators)
                    kind='primitive' if g in group.generators else 'unseen_composite'
                loss=np.sum((pred-z[b])**2,axis=1)/energy
                if not np.isfinite(loss).all():raise ValueError('Nonfinite storage-curve prediction')
                for index,value in zip(a,loss):values[kind][rows[index]['family_id']].append(float(value))
            count=field.x.size+sum(v.size for v in field.delta.values())
            curves.append({'track':track,'K':k,'full_training_support':k==len(x),
                'coordinate_storage_matched_budget_K':matched,'selected_training_indices':support.tolist(),
                'selection':field.selection,'counted_float64_coordinates':count,'counted_predictor_array_bytes':count*8,
                'linear_map_array_bytes':len(actions)*z.shape[1]**2*8,
                'capacity_or_total_RAM_matched':False,'results':{kind:pack(v) for kind,v in values.items()}})
    return {'controls':{m:pack(v) for m,v in controls.items()},'primary_selection':primary_selection,
        'validation_source_decode_accuracy':float(np.mean(decoded[val]==states[val])),
        'evaluation_source_decode_accuracy':float(np.mean(decoded[evaluate]==states[evaluate])),
        'decoded_local_validation_losses':scores,'decoded_local_selected_bandwidth':
            'inf' if math.isinf(tuned.multiplier) else tuned.multiplier,'storage_curves':curves,
        'support_selection':'training-only label-free farthest-first, first row seed, row-order ties; full_N retains original order',
        'formal_inference':False}
