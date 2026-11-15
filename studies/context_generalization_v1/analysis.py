"""Pure NumPy estimators and paired-family summaries; no model imports."""
from __future__ import annotations
import numpy as np


def blocks(z, rows, domain, split, expected_families=None, *, row_domain=None):
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 2 or len(z) != len(rows) or not np.isfinite(z).all():
        raise ValueError('Coordinates must be finite real rows')
    p = {'C7': 7, 'D4': 4}[domain]
    row_domain = domain if row_domain is None else row_domain
    if row_domain not in {'C7': ('C7',), 'D4': ('D4', 'D4_corner')}[domain]:
        raise ValueError('Unsupported row domain for action')
    selected = {}
    ids = set()
    for i, row in enumerate(rows):
        rid = row['row_id']
        if rid in ids:
            raise ValueError('Duplicate row identity')
        ids.add(rid)
        if row['domain'] != row_domain:
            raise ValueError('Mixed domains')
        if row['split'] != split:
            continue
        f, para, state = row['family_id'], row['paraphrase_id'], row['state_id']
        if type(para) is not int or para not in (0,1) or type(state) is not int or not 0 <= state < p:
            raise ValueError('Invalid state or paired rendering')
        key = (f, para, state)
        if key in selected:
            raise ValueError('Duplicate family/rendering/state')
        selected[key] = i
    families = sorted({k[0] for k in selected})
    if not families or (expected_families is not None and families != list(expected_families)):
        raise ValueError('Family identities differ')
    keys = [(f, para, s) for f in families for para in (0,1) for s in range(p)]
    if set(keys) != set(selected):
        raise ValueError('Incomplete paired state blocks')
    return z[[selected[k] for k in keys]].reshape(len(families), 2, p, z.shape[1]), families


def action_tables(domain):
    if domain == 'C7':
        return [np.array([(i+k)%7 for i in range(7)]) for k in range(1,7)]
    if domain == 'D4':
        return [np.array([(k+(1-i if b else i))%4 for i in range(4)])
                for b in (0,1) for k in range(4) if (k,b) != (0,0)]
    raise ValueError('Unsupported domain')


def ridge(x, y, penalty):
    if x.ndim != 2 or y.shape != x.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Invalid fitting arrays')
    if penalty < 0 or not np.isfinite(penalty):
        raise ValueError('Invalid penalty')
    if penalty == 0:
        return np.linalg.lstsq(x, y, rcond=1e-12)[0]
    scale = float(np.sum(x*x) / x.shape[1])
    if scale <= np.finfo(float).tiny:
        raise ValueError('Degenerate training source energy')
    return np.linalg.solve(x.T@x + penalty*scale*np.eye(x.shape[1]), x.T@y)


def fit_predictors(train, validation, domain, grid):
    """No test argument exists. Train contains all states, including fixed ones."""
    if train.ndim != 4 or validation.ndim != 4 or train.shape[1:] != validation.shape[1:] or train.shape[1] != 2:
        raise ValueError('Expected paired family/state coordinate tensors')
    if not np.isfinite(train).all() or not np.isfinite(validation).all():
        raise ValueError('Nonfinite fit data')
    u = train[:,0]-train[:,1]; uv = validation[:,0]-validation[:,1]
    energy = float(np.mean(np.sum(u*u,axis=-1)))
    if not np.isfinite(energy) or energy <= np.finfo(float).tiny:
        raise ValueError('Undefined training context-difference normalization')
    matrices = {'coordinate': [], 'coordinate_nativeval': [], 'difference': []}; ledger=[]
    for action, table in enumerate(action_tables(domain)):
        changed = np.arange(len(table)) != table
        for kind, x, y, xv, yv in (
            ('coordinate', train, train[:,:,table], validation[:,:,changed], validation[:,:,table[changed]]),
            ('difference', u, u[:,table], uv[:,changed], uv[:,table[changed]])):
            d = train.shape[-1]
            x,y,xv,yv = [a.reshape(-1,d) for a in (x,y,xv,yv)]
            candidates = []
            for penalty in grid:
                R = ridge(x,y,float(penalty))
                native_loss = float(np.mean(np.sum((xv@R-yv)**2,axis=1)))
                # Both PRIMARY arms select on exactly the same old-validation
                # context differences, even though their training targets differ.
                loss = float(np.mean(np.sum((uv[:,changed]@R-uv[:,table[changed]])**2,axis=-1)))
                if not np.isfinite(loss) or not np.isfinite(native_loss) or not np.isfinite(R).all():
                    raise ValueError('Nonfinite candidate fit')
                candidates.append((loss,float(penalty),R,native_loss))
            best = min(range(len(candidates)),key=lambda j:candidates[j][0])
            matrices[kind].append(candidates[best][2])
            ledger.append({'action_index':action,'training_objective':kind,
                           'selection_objective':'context_difference_validation_mse',
                           'selected_grid_index':best,'selected_lambda':candidates[best][1],
                           'validation_candidates':[{'lambda':c[1],'context_mse':c[0], 'native_objective_mse':c[3]} for c in candidates]})
            if kind=='coordinate':
                native = min(range(len(candidates)),key=lambda j:candidates[j][3])
                matrices['coordinate_nativeval'].append(candidates[native][2])
                ledger.append({'action_index':action,'training_objective':'coordinate',
                               'arm':'secondary_coordinate_nativeval',
                               'selection_objective':'coordinate_validation_mse',
                               'selected_grid_index':native,'selected_lambda':candidates[native][1],
                               'validation_candidates':[{'lambda':c[1],'mse':c[3]} for c in candidates]})
    params={k:np.stack(v) for k,v in matrices.items()}
    return params, {'training_difference_energy':energy,'selection_ledger':ledger,
                    'test_used_for_fit_or_selection':False,
                    'normalization':'mean old-train uncentered squared context difference',
                    'fit_pairs':'all training states per nonidentity action',
                    'selection_pairs':'changed states on old validation only',
                    'tie_rule':'first exact minimum in prespecified grid order'}


def evaluate(params, metadata, test, domain, families):
    if test.ndim != 4 or test.shape[0] != len(families) or test.shape[1] != 2 or not np.isfinite(test).all():
        raise ValueError('Invalid held-out data')
    u = test[:,0]-test[:,1]; energy=metadata['training_difference_energy']
    if not np.isfinite(energy) or energy <= np.finfo(float).tiny:
        raise ValueError('Invalid frozen training energy')
    losses={k:[] for k in ('identity','coordinate','difference','coordinate_nativeval')}
    tables=action_tables(domain)
    for kind in ('coordinate','difference','coordinate_nativeval'):
        if params[kind].shape != (len(tables),test.shape[-1],test.shape[-1]) or not np.isfinite(params[kind]).all():
            raise ValueError('Invalid frozen predictor parameters')
    for ai,table in enumerate(tables):
        changed=np.arange(len(table)) != table
        x=u[:,changed]; y=u[:,table[changed]]
        for kind in losses:
            pred=x if kind=='identity' else x@params[kind][ai]
            losses[kind].append(np.mean(np.sum((pred-y)**2,axis=-1),axis=1)/energy)
    per={k:np.mean(v,axis=0) for k,v in losses.items()}
    if not all(np.isfinite(v).all() for v in per.values()):
        raise ValueError('Nonfinite evaluation')
    records=[]
    for i,family in enumerate(families):
        records.append({'family_id':family,**{k:float(v[i]) for k,v in per.items()},
                        'difference_minus_coordinate':float(per['difference'][i]-per['coordinate'][i])})
    return {'status':'complete','per_family':records,
            'means':{k:float(v.mean()) for k,v in per.items()},
            'action_aggregation':'equal nonidentity actions after equal changed states',
            'negative_primary_effect_favors':'difference_fitted'}


def aggregate(cells, families, expected=48, draws=2000, seed=2026102901):
    complete=[c for c in cells if c.get('status')=='complete']
    if len(cells)!=expected or len(complete)!=expected or len({c['cell'] for c in cells})!=expected:
        return {'status':'incomplete_panel','planned_cells':expected,'completed_cells':len(complete),
                'primary_effect':None,'reason':'No cell deletion or replacement is allowed'}
    family_vectors={}
    for domain in ('C7','D4'):
        values=[]
        for c in complete:
            if c['domain']!=domain:continue
            if [r['family_id'] for r in c['per_family']]!=families:
                raise ValueError('Family order differs across cells')
            values.append([r['difference_minus_coordinate'] for r in c['per_family']])
        if len(values)!=expected//2:
            raise ValueError('Unequal or incomplete domain panel')
        family_vectors[domain]=np.asarray(values).mean(axis=0)
    family_vectors['primary_equal_domain_panel']=(family_vectors['C7']+family_vectors['D4'])/2
    rng=np.random.default_rng(seed)
    sample=rng.integers(0,len(families),size=(draws,len(families)))
    summaries={}
    for name,values in family_vectors.items():
        means=values[sample].mean(axis=1)
        summaries[name]={'mean_difference_minus_coordinate':float(values.mean()),
                         'family_bootstrap_percentiles_2_5_97_5':np.quantile(means,[.025,.975]).tolist(),
                         'n_authored_families':len(families)}
    return {'status':'complete','planned_cells':expected,'completed_cells':expected,'contrasts':summaries,
            'bootstrap_draws':draws,'bootstrap_seed':seed,
            'scope':'conditional sensitivity over this fixed authored family bank and fixed model/layer panel',
            'population_confidence_certificate':False,'significance_or_equivalence_decision':False,
            'independent_human_authorship_claim':False}
