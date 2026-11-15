"""Copy the already-existing nested FULL archive; no model or analysis execution."""
import argparse, hashlib, shutil, zipfile
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--repo',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
a.output.parent.mkdir(parents=True,exist_ok=True)
expected='5fe7234572120073d8dfab2e8dc0d3373a741f8d48abadc92288f98ab16721f4'
if not a.output.exists():
    with zipfile.ZipFile(a.repo/'runpod_packages/Beyond_Shared_Geometry_afs_d_rank_aware_immutable_inputs.zip') as z:
        with z.open('immutable/run_outputs_9_4_2026_full3.zip') as src, a.output.open('xb') as dst:
            shutil.copyfileobj(src,dst)
h=hashlib.sha256()
with a.output.open('rb') as f:
    for b in iter(lambda:f.read(2**20),b''):h.update(b)
if h.hexdigest()!=expected:raise RuntimeError('Nested input checksum mismatch')
print('VERIFIED_EXISTING_INPUT_COPY',h.hexdigest())
