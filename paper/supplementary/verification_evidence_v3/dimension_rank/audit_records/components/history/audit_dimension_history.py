#!/usr/bin/env python3
"""Read-only lineage and dimension-rule evidence; never imports experimental code."""
from pathlib import Path
import argparse, hashlib, json, subprocess, zipfile

def sha(b): return hashlib.sha256(b).hexdigest()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--repo',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args()
    root=a.repo.resolve(); release=root/'submission/tmlr_20261101_final_edit_v1/beyond-shared-geometry'
    paths=['configs/main_experiment.yaml','docs/FINAL_EXPERIMENT_DESIGN.md','docs/EXPERIMENT_PROTOCOL.md','docs/PRE_FULL_PROTOCOL.md','src/bsg/experiments/main_config.py','docs/ALGEBRA_FALSIFICATION_PROTOCOL.md','configs/algebra_falsification.yaml','runpod_output_by_date/result_9_6_2026/Beyond_Shared_Geometry_main_experiment_results/manifest.json']
    release_paths=['studies/characterization_v1/frozen_run/study_config.json','studies/characterization_v1/frozen_run/README.txt','studies/characterization_v1/frozen_run/source_manifest.json','studies/characterization_v1/frozen_run/src/bsg_next/structural.py','studies/reproduction_v1/characterization.py','studies/compact_context_v1/protocol.json','studies/compact_context_v1/common.py','studies/spectral_finite_bank_v1/protocol.json','studies/spectral_extensions_v1/protocol.json']
    manifest=[]
    for base,ps in [(root,paths),(release,release_paths)]:
        for p in ps:
            f=base/p; b=f.read_bytes(); manifest.append({'path':str(f.relative_to(root)),'bytes':len(b),'sha256':sha(b)})
    cmds=[['git','rev-parse','HEAD'],['git','log','--format=%H %aI %s','--reverse'],['git','show','ef5c7aa:configs/main_experiment.yaml'],['git','show','30ae942:docs/FINAL_EXPERIMENT_DESIGN.md']]
    git={}
    for cmd in cmds:
        r=subprocess.run(cmd,cwd=root,capture_output=True,text=True,check=True);git[' '.join(cmd)]=r.stdout
    archive_checks=[]
    rr=root/'incoming/rank_recovery_v1/from-zero-v1-rank-recovery.results.zip'
    with zipfile.ZipFile(rr) as z:
        member='parent-from-zero-v1/characterization/study_contract.json'; b=z.read(member);c=json.loads(b)
        archive_checks.append({'archive':str(rr.relative_to(root)),'member':member,'sha256':sha(b),'contract_id':c.get('study_id')})
        for key,p in [('frozen_source_manifest_sha256','studies/characterization_v1/frozen_run/source_manifest.json'),('driver_sha256','studies/reproduction_v1/characterization.py'),('reference_configuration_sha256','studies/characterization_v1/frozen_run/study_config.json')]:
            h=sha((release/p).read_bytes());archive_checks.append({'contract_key':key,'path':p,'observed_sha256':h,'expected_sha256':c[key],'equal':h==c[key]})
    rr=root/'incoming/tmlr_all_v2/tmlr-all-v2.20261030T203725Z-5489fd.results.zip'
    with zipfile.ZipFile(rr) as z:
        for p in ('studies/spectral_finite_bank_v1/protocol.json','studies/spectral_extensions_v1/protocol.json'):
            b=z.read('source_snapshot/'+p);h=sha(b);archive_checks.append({'archive':str(rr.relative_to(root)),'member':'source_snapshot/'+p,'sha256':h,'same_as_release':b==(release/p).read_bytes()})
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/'history_evidence.json').write_text(json.dumps({'repo':str(root),'release':str(release),'files':manifest,'git_evidence':git,'archive_checks':archive_checks},indent=2)+'\n')
    print(json.dumps({'read_only':True,'source_file_count':len(manifest),'archive_checks':archive_checks},indent=2))
if __name__=='__main__':main()
