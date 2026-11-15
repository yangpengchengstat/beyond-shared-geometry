"""Read-only diagnostics of frozen AFS-D inputs; never fits or reconstructs PCA.

Requires NumPy only. Reads explicit archives, writes derived CSV/JSON under --output.
The frozen protocol.json must exist before execution. No bsg module is executed.
"""
import argparse, csv, hashlib, io, json, platform, tarfile, zipfile
from pathlib import Path
import numpy as np


def digest(data):
    return hashlib.sha256(data).hexdigest()


def filehash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--internal-archive', type=Path, required=True)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((a.output/'protocol.json').read_text())
    assert protocol['ill_conditioned_diagnostic']['threshold'] == 1e-8
    sources = {}
    rows, identity, missing = [], [], []
    archive = a.repo/'AFS_D_rank_aware_001_full_runpod_output.tar.gz'
    sources[str(archive)] = filehash(archive)
    sources[str(a.internal_archive)] = filehash(a.internal_archive)
    assert sources[str(a.internal_archive)] == '5fe7234572120073d8dfab2e8dc0d3373a741f8d48abadc92288f98ab16721f4'
    with tarfile.open(archive, 'r:gz') as t:
        members = {m.name: t.extractfile(m).read() for m in t if m.isfile()}
    prefix = 'run_outputs/algebra_falsification/afs_d_rank_aware_001/'
    original_ranks = list(csv.DictReader(io.StringIO(members[prefix+'afs_d/embedding_rank_diagnostics.csv'].decode())))
    availability = list(csv.DictReader(io.StringIO(members[prefix+'provenance/internal_activation_availability.csv'].decode())))
    ledger = json.loads(members[prefix+'run_ledger.json'])
    for source, content in members.items():
        if source.endswith(('.npz', 'manifest.json', 'config_snapshot.yaml', 'embedding_rank_diagnostics.csv', 'internal_activation_availability.csv', 'run_ledger.json')):
            sources[str(archive)+'!'+source] = digest(content)
    completed = [u['fields'] for k,u in ledger['units'].items()
                 if k.split(':')[0] in ('embedding_analysis','internal_rank_matched_analysis') and u['status']=='COMPLETE']
    assert len(completed)==100
    def mask_for(n, seed):
        # Exact source splits.py algorithm: template-major rows, 40 templates.
        ids=np.arange(40); np.random.default_rng(seed).shuffle(ids)
        return np.isin(np.repeat(np.arange(40),n),ids[:28])

    def diagnose(matrix, meta, kind, bank, source, memberhash, q, center):
        original_dtype = str(matrix.dtype)
        x=np.asarray(matrix,dtype=np.float64)
        if center: x=x-x.mean(axis=0,keepdims=True)
        # Values only: no PCA components computed, saved or replaced.
        s=np.linalg.svd(x,compute_uv=False,full_matrices=False)
        tol=float(max(x.shape)*np.finfo(np.float64).eps*s[0])
        rank=int(np.count_nonzero(s>tol))
        sq=float(s[q-1]); ratio=sq/float(s[0]); total=float(np.sum(x*x))
        status='verified_deficient' if rank<q else ('borderline_or_ill_conditioned' if ratio<=1e-8 else 'verified_supported')
        row=dict(study='Study1_AFS_D_original',model=meta['model'],domain=meta['group'],readout=meta['readout'],split=meta['split'],bank=bank,
                 pca_identity='historical train-only full SVD; fitted basis not archived',input_identity=memberhash,
                 matrix_kind=kind,shape='x'.join(map(str,x.shape)),centering='arithmetic row mean' if center else 'none',weights='uniform rows',
                 q=q,numerical_rank=rank,sigma_max=float(s[0]),sigma_min_retained=sq,sigma_min_full=float(s[-1]),sigma_retained_ratio=ratio,
                 input_dtype=original_dtype,compute_dtype='float64',tau_abs=0.0,tau_rel=max(x.shape)*np.finfo(np.float64).eps,threshold=tol,
                 threshold_rule='max(shape)*eps(float64)*sigma_max; strict >',historical_rule='numpy.linalg.matrix_rank(np.asarray(..., dtype=float))',
                 evr_q=float(np.dot(s[:q],s[:q])/total) if total else '',evr_denominator='full centered original matrix Frobenius squared',
                 evidence_source=source,status=status,interpretation='rank suffices for q leading directions; not a remeasurement of saved projected coordinates',
                 original_result='')
        rows.append(row)
        print('DIAGNOSTIC',meta,kind,rank,ratio,flush=True)
        return row

    for name,payload in sorted(members.items()):
        if not name.endswith('.npz'): continue
        with np.load(io.BytesIO(payload),allow_pickle=False) as z:
            h=z['hidden']; meta=json.loads(str(z['metadata'].item()))
        n={'C7':7,'D4':4}[meta['group']]; q=n-1
        assert h.shape[0]==40*n
        # Verify context identity per state before selecting representatives.
        by=h.reshape(40,n,h.shape[1])
        state_unique=[int(np.unique(by[:,i,:],axis=0).shape[0]) for i in range(n)]
        all_identical=bool(np.array_equal(by,np.broadcast_to(by[:1],by.shape)))
        identity.append(dict(model=meta['model'],domain=meta['group'],shape=list(h.shape),unique_rows=int(np.unique(h,axis=0).shape[0]),
                             per_state_unique_rows=state_unique,bitwise_identical_across_40_contexts=all_identical,
                             representative_selection='first observed context only AFTER bitwise equality check; no context averaging',
                             evidence_source=str(archive)+'!'+name,sha256=digest(payload)))
        if not all_identical or state_unique != [1]*n: raise RuntimeError('No unique lexical vector per state; stop unique E inference')
        basic=dict(model=meta['model'],group=meta['group'],readout='embedding_0',split='all_unique_states')
        diagnose(by[0],basic,'unique_lexical_E_centered','unique_state_bank',str(archive)+'!'+name,digest(payload),q,True)
        for seed in [101,303,909,2701,4243]:
            basic=dict(model=meta['model'],group=meta['group'],readout='embedding_0',split=seed)
            row=diagnose(h[mask_for(n,seed)],basic,'raw_training_H_centered','train',str(archive)+'!'+name,digest(payload),q,True)
            hist=next(r for r in original_ranks if r['model']==meta['model'] and r['group']==meta['group'] and int(r['split_seed'])==seed)
            assert row['numerical_rank']==int(hist['centered_training_rank'])==q
            row['original_result']='PASS; historical centered_training_rank='+hist['centered_training_rank']
    with zipfile.ZipFile(a.internal_archive) as z:
        for v in availability:
            name='run_outputs/activation_cache/'+v['cache_key']+'.npz'
            payload=z.read(name); assert digest(payload)==v['sha256']
            sources[str(a.internal_archive)+'!'+name]=digest(payload)
            with np.load(io.BytesIO(payload),allow_pickle=False) as ar:
                h=ar['activations']; meta=json.loads(str(ar['metadata'].item()))
            assert meta['model_id']==v['model'] and meta['group_task']=='structural_'+v['group'] and int(meta['layer'])==int(v['layer'])
            n={'C7':7,'D4':4}[v['group']]; q=n-1
            assert h.shape[0]==40*n
            for seed in [101,303,909,2701,4243]:
                basic=dict(model=v['model'],group=v['group'],readout='block_'+v['layer'],split=seed)
                row=diagnose(h[mask_for(n,seed)],basic,'raw_training_H_centered','train',str(a.internal_archive)+'!'+name,digest(payload),q,True)
                row['original_result']='COMPLETE matched internal fit; observed rank guard in code'
    # Deliberately no reconstruction of projected coordinates or fitted operators.
    for f in completed:
        for bank in ['train','validation','test']:
            n={'C7':7,'D4':4}[f['group']]; q=int(f['latent_dim']); count={'train':28,'validation':6,'test':6}[bank]*n
            rows.append(dict(study='Study1_AFS_D_original',model=f['model'],domain=f['group'],readout='embedding_0' if f['layer']==0 else 'block_'+str(f['layer']),split=f['split_seed'],bank=bank,
                             pca_identity='not persisted by AFS-D metrics-only checkpoints',input_identity='',matrix_kind='retained_Z',shape=f'{count}x{q} (configured; array unavailable)',centering='original frozen training mean',weights='uniform rows',q=q,
                             numerical_rank='',evidence_source=str(archive)+'!'+prefix+'run_ledger.json',status='unavailable',
                             interpretation='Metrics and COMPLETE status retained; no fitted PCA or projected coordinates persisted. Raw-input feasibility is not direct saved-coordinate validation.',
                             evr_q='',evr_denominator='NA: projected array unavailable'))
    replay=a.repo/'incoming/rank_recovery_v1/from-zero-v1-rank-recovery.results.zip'
    replay_members=[]; replay_check=[]
    with zipfile.ZipFile(replay) as z:
        for info in z.infolist():
            if '/historical/runs/afs_d/' in info.filename and info.filename.endswith(('.npz','.npy')):
                replay_members.append(info.filename)
                b=z.read(info.filename)
                original=next(p for k,p in members.items() if k.endswith('/'+Path(info.filename).name))
                replay_check.append({'member':info.filename,'sha256':digest(b),'byte_identical_to_original_embedding_checkpoint':b==original})
        d=z.read('from-zero-v1-rank-recovery/historical/runs/afs_d/afs_d/embedding_rank_diagnostics.csv')
        replay_check.append({'member':'historical/runs/afs_d/afs_d/embedding_rank_diagnostics.csv','sha256':digest(d),'byte_identical_to_original':d==members[prefix+'afs_d/embedding_rank_diagnostics.csv']})
    assert len(replay_members)==4
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (a.output/'lexical_dimension_rank_audit.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    for filename,obj in [('lexical_identity_checks.json',identity),('replay_archive_check.json',replay_check),('input_sha256_before.json',sources)]:
        (a.output/filename).write_text(json.dumps(obj,indent=2)+'\n')
    preserve={str(archive):filehash(archive),str(a.internal_archive):filehash(a.internal_archive)}
    assert all(sources[k]==v for k,v in preserve.items())
    (a.output/'preservation_check.json').write_text(json.dumps({'checked_files':preserve,'unchanged':True,'matrix_objects_read_only':True,'numpy_version':np.__version__,'python':platform.python_version(),'protocol_sha256':filehash(a.output/'protocol.json'),'new_pca_or_operator_fits':0},indent=2)+'\n')
    print('COMPLETE',len(rows),'rows; original and replay embedding payloads compared; zero fitting',flush=True)


if __name__=='__main__': main()
