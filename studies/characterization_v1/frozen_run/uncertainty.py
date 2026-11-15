"""Prespecified descriptive cluster-bootstrap sensitivity, never decisions."""
from collections import defaultdict
import numpy as np
from bsg_next.statistics import bootstrap_sensitivity


def summarize(legacy,fields,expected_conditions=24,replicates=10000,seed=314159):
    def interval(vector,crossed=False):
        value=bootstrap_sensitivity(np.asarray(vector,float),replicates=replicates,seed=seed,crossed=crossed)
        value.update(formal_significance=False,coverage_certified_for_new_comparison=False,
                     fixed_model_panel=True)
        return value
    def collapse(entries):
        if not entries or any(len(v)!=expected_conditions for v in entries.values()):
            raise ValueError('Incomplete or duplicate fixed-panel family coverage')
        return {k:float(np.mean(v)) for k,v in sorted(entries.items())}
    legacy_groups=defaultdict(list)
    for record in legacy:legacy_groups[record['group']+'|'+record['test_split']].append(record)
    original={}
    for key,records in sorted(legacy_groups.items()):
        entry={}
        for endpoint,field in [('prediction','family_orbit_prediction'),('context','family_orbit_context')]:
            grouped=defaultdict(list)
            for record in records:
                for pair,value in record[field].items():grouped[pair].append(value)
            collapsed=collapse(grouped)
            families=sorted({p.split('|')[0] for p in collapsed})
            orbits=sorted({p.split('|')[1] for p in collapsed})
            crossed=key.startswith('S3_role|')
            if crossed:
                values=np.array([[collapsed[f+'|'+o] for o in orbits] for f in families])
            else:
                values=np.array([np.mean([collapsed[p] for p in collapsed if p.split('|')[0]==f]) for f in families])
            entry[endpoint]={'families':len(families),'orbits':len(orbits),'eligible':
                all(r['eligible'] for r in records) if endpoint=='context' else True,
                'contrast_direction':'validation_selected_baseline_minus_linear_operator',
                'sensitivity_interval':interval(values,crossed)}
        original[key]=entry
    by_group=defaultdict(list)
    for r in fields:by_group[r['group']].append(r)
    curves={}
    for group,records in sorted(by_group.items()):
        pooled=defaultdict(lambda:defaultdict(list))
        for r in records:
            for curve in r['input_comparisons']['storage_curves']:
                for kind,values in curve['results'].items():
                    if curve['track']=='all_element':
                        linear=r['input_comparisons']['controls']['primary_operator']['family_losses']
                    else:
                        linear=r['generator']['family_losses'][kind]['generator_ridge']
                    for family,value in values['family_losses'].items():
                        pooled[(curve['track'],curve['K'],kind)][family].append(value-linear[family])
        curves[group]=[]
        for (track,k,kind),entry in sorted(pooled.items()):
            collapsed=collapse(entry)
            curves[group].append({'track':track,'K':k,'endpoint':kind,'families':len(collapsed),
                'contrast_direction':'kernel_minus_linear_error; negative favors kernel',
                'sensitivity_interval':interval(list(collapsed.values()))})
    return {'scope':'Prespecified sensitivity intervals for a fixed six-model/four-layer panel. Not calibrated new hypothesis tests; no powered or significance certificate.',
        'paired_cluster_rule':'Average fixed model/layer conditions within family (and S3 entity orbit), then resample whole families; independently resample S3 family and entity indices.',
        'seed':seed,'replicates':replicates,'multiple_comparison_adjustment':False,
        'legacy_contrasts':original,'storage_curve_contrasts':curves}
