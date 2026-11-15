"""Build the additive recovery patch without modifying the original release."""
from pathlib import Path
import hashlib,json,sys,zipfile
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'studies/reproduction_v1'))
from receipt import source_identity

def build():
    parent=source_identity()
    files=sorted(p for p in HERE.iterdir() if p.suffix in ('.py','.sh','.md') and p.is_file())
    manifest={'parent_source_manifest_sha256':parent,
              'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
              'original_release_modified':False,'original_q16_D':'INFEASIBLE_RANK_DEFICIENT_PRESERVED'}
    (HERE/'source_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    destination=ROOT/'dist/2026-10-22-rank-recovery-v1';destination.mkdir(parents=True,exist_ok=True)
    path=destination/'bsg-rank-recovery-v1.zip'
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in [*files,HERE/'source_manifest.json']:
            item=zipfile.ZipInfo(str(p.relative_to(ROOT)),date_time=(2026,10,22,0,0,0))
            item.create_system=3;item.external_attr=(0o100755 if p.suffix=='.sh' else 0o100644)<<16
            z.writestr(item,p.read_bytes(),compress_type=zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(path) as z:
        if z.testzip():raise ValueError('Patch CRC failure')
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_name(path.name+'.sha256').write_text(digest+'  '+path.name+'\n')
    print(json.dumps({'path':str(path),'sha256':digest,'bytes':path.stat().st_size,
        'parent_source_manifest_sha256':parent,'GPU_recovery_executed':False},indent=2))

if __name__=='__main__':build()
