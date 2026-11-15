#!/usr/bin/env python3
"""Package the verified delivery; standard library only, no analysis or compiler."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import zipfile
from check_release import tex_dependencies
from verify_delivery import verify
from verify_submission_surface import verify as verify_surface

STAMP = (2026, 11, 8, 0, 0, 0)
EXCLUDED = {'.git', '__pycache__', '.pytest_cache', '.venv', 'venv', 'node_modules', 'dist', 'build'}
SECRET = re.compile(rb'(?:\b(?:hf_|ghp_|github_pat_|sk-proj-)[A-Za-z0-9_-]{20,}\b|\bAKIA[A-Z0-9]{16}\b|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)')
IDENTITY = re.compile(rb'Pengcheng|pengchengyang|LLM_PJ|/Users/|/home/|github\.com/[^\s/]+/beyond-shared-geometry', re.I)
PRIVATE_PATH = re.compile(rb'/Users/[A-Za-z0-9_.-]+/|/home/[A-Za-z0-9_.-]+/|/private/(?:tmp|var)/')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def validate_members(members, anonymous=False):
    for name, raw in members.items():
        p = Path(name)
        if p.is_absolute() or '..' in p.parts or any(s in EXCLUDED for s in p.parts):
            raise ValueError('Unsafe archive path: ' + name)
        if len(raw) >= 100_000_000:
            raise ValueError('File exceeds public source limit: ' + name)
        if SECRET.search(raw) or PRIVATE_PATH.search(raw):
            raise ValueError('Private information in package: ' + name)
        if anonymous and IDENTITY.search(raw):
            raise ValueError('Identifying content in review package: ' + name)


def write_zip(path, members):
    with zipfile.ZipFile(path, 'x', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name, raw in sorted(members.items()):
            info = zipfile.ZipInfo(name, STAMP)
            info.create_system = 3
            info.external_attr = (0o100755 if name.endswith('.sh') else 0o100644) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, raw, compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
    with zipfile.ZipFile(path) as z:
        if z.testzip() is not None:
            raise ValueError('ZIP CRC failure')
        for name, raw in members.items():
            if sha(z.read(name)) != sha(raw):
                raise ValueError('ZIP member mismatch: ' + name)
    return {'files': len(members), 'bytes': path.stat().st_size, 'sha256': sha(path.read_bytes())}


def paper_members(root):
    paper = root / 'paper'
    members = {p.relative_to(paper).as_posix(): p.read_bytes() for p in tex_dependencies(root)}
    # Include the saved-data and presentation chains, not just TeX inputs.
    for folder in ('supplementary', 'generated', 'figures'):
        for p in sorted((paper / folder).rglob('*')):
            if any(s in EXCLUDED for s in p.relative_to(paper).parts) or p.suffix == '.pyc':
                continue
            if p.is_symlink():
                raise ValueError('Linked paper member')
            if p.is_file():
                members[p.relative_to(paper).as_posix()] = p.read_bytes()
    for name in ('TMLR_TEMPLATE_LICENSE.txt', 'OVERLEAF_README.txt', 'submission_metadata.json'):
        members[name] = (paper / name).read_bytes()
    if (paper / 'main.pdf').exists():
        # verify(root) already requires a fresh source-bound successful receipt.
        members['main.pdf'] = (paper / 'main.pdf').read_bytes()
    return members


def package(root, output):
    report = verify(root)
    if report['status'] != 'PASS':
        raise ValueError('Static verification failed: ' + json.dumps(report['errors']))
    surface = verify_surface(root)
    if surface['status'] != 'PASS':
        raise ValueError('Manuscript surface verification failed: ' + json.dumps(surface))
    if output.exists():
        raise FileExistsError('Choose a new output directory; existing releases are never overwritten')
    public = {}
    for p in sorted(root.rglob('*')):
        rel = p.relative_to(root)
        if any(s in EXCLUDED for s in rel.parts) or p.name == '.DS_Store' or p.suffix == '.pyc':
            continue
        if p.is_symlink():
            raise ValueError('Linked public member: ' + str(rel))
        if p.is_file():
            public['beyond-shared-geometry/' + rel.as_posix()] = p.read_bytes()
    anon = paper_members(root)
    anon['CONTENTS.sha256.json'] = (json.dumps({n: sha(b) for n, b in sorted(anon.items())}, indent=2)+'\n').encode()
    review = {'paper/' + n: b for n, b in anon.items()}
    for p in (root / 'review_reproduction').rglob('*'):
        if p.is_file() and not any(s in EXCLUDED for s in p.parts) and p.suffix != '.pyc':
            review[p.relative_to(root).as_posix()] = p.read_bytes()
    review['README_REVIEW.txt'] = (
        'Anonymous review source and saved-evidence supplement\n'
        'Compile paper/main.tex from scratch with pdfLaTeX + BibTeX.\n'
        'No manuscript PDF is supplied unless a fresh source-bound receipt exists.\n'
        'Read paper/OVERLEAF_README.txt and review_reproduction/README.md.\n'
        'The paper supplementary directory includes current full-precision endpoints,\n'
        'table/figure renderers, dimension/rank audit records and preserved stops.\n'
        'Historical validation and stage receipts remain historical, not current build claims.\n'
        'The 8/16-dimensional and independent orthogonal analyses are complete.\n'
        'C7 canonical-action family-risk LOFO is complete; all-image registration LOFO is not run.\n'
        'These files support saved-result reproduction, not reconstruction of all raw activations.\n'
        'No models or experiments run as part of packaging. Author review remains pending.\n'
    ).encode()
    validate_members(public)
    validate_members(anon, anonymous=True)
    validate_members(review, anonymous=True)
    output.mkdir(parents=True)
    reports = {}
    for name, members in [('github-source.zip', public), ('overleaf_source.zip', anon),
                          ('anonymous-review-source.zip', review)]:
        reports[name] = write_zip(output / name, members)
    for name in ('overleaf_source.zip', 'anonymous-review-source.zip'):
        if reports[name]['bytes'] >= 100_000_000:
            raise ValueError('Review archive exceeds 100 MB: ' + name)
    (output / 'SHA256SUMS.txt').write_text(''.join(r['sha256']+'  '+n+'\n' for n, r in reports.items()))
    record = {'status': 'PASS', 'scope': 'CRC, member hashes, source verification and static privacy scans',
              'archives': reports, 'scientific_analysis_run': False, 'models_run': False,
              'author_review_status': 'pending', 'submission_ready': False,
              'final_pdf_review': 'NOT_RUN', 'metadata': 'github/release/current_release_status.json'}
    (output / 'PACKAGE_CHECK.json').write_text(json.dumps(record, indent=2)+'\n')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(package(Path(__file__).resolve().parents[1], args.output.resolve()), indent=2))
