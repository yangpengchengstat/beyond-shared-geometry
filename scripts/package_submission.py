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

STAMP = (2026, 11, 13, 0, 0, 0)
EXCLUDED = {'.git', '__pycache__', '.pytest_cache', '.venv', 'venv', 'node_modules', 'dist', 'build'}
PUBLIC_FILE_LIMIT = 100 * 1024 * 1024
REVIEW_DIRECTORIES = ('studies/template_centering_v1', 'results/template_centering_v1',
                      'studies/spectral_dimension_sensitivity_v1/core')
SECRET = re.compile(rb'(?:\b(?:hf_|ghp_|github_pat_|sk-proj-)[A-Za-z0-9_-]{20,}\b|\bAKIA[A-Z0-9]{16}\b|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)')
IDENTITY = re.compile(rb'Pengcheng|pengchengyang|LLM_PJ|/Users/|/home/|github\.com/[^\s/]+/beyond-shared-geometry', re.I)
PRIVATE_PATH = re.compile(rb'/Users/[A-Za-z0-9_.-]+/|/home/[A-Za-z0-9_.-]+/|/private/(?:tmp|var)/')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def validate_members(members, anonymous=False):
    for name, raw in members.items():
        p = Path(name)
        if p.is_absolute() or '..' in p.parts or '\\' in name or any(s in EXCLUDED for s in p.parts):
            raise ValueError('Unsafe archive path: ' + name)
        if len(raw) >= PUBLIC_FILE_LIMIT:
            raise ValueError('File reaches the 100 MiB public source limit: ' + name)
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


def review_members(root, paper):
    """Select the anonymous evidence closure; never execute its analysis code."""
    members = {'paper/' + name: raw for name, raw in paper.items()}
    for folder in ('review_reproduction',) + REVIEW_DIRECTORIES:
        base = root / folder
        if (not base.is_dir() or base.is_symlink() or
                not base.resolve().is_relative_to(root.resolve())):
            raise ValueError('Missing or linked review directory: ' + folder)
        for path in sorted(base.rglob('*')):
            rel = path.relative_to(root)
            if any(part in EXCLUDED for part in rel.parts) or path.suffix == '.pyc' or path.name == '.DS_Store':
                continue
            if path.is_symlink():
                raise ValueError('Linked review member: ' + rel.as_posix())
            if path.is_file():
                members[rel.as_posix()] = path.read_bytes()
    selection = 'data/retained_coordinates/anonymous_files_c7_d4.json'
    manifest_path = root / selection
    if (not manifest_path.is_file() or manifest_path.is_symlink() or
            not manifest_path.resolve().is_relative_to(root.resolve())):
        raise ValueError('Missing or linked anonymous C7/D4 file selection')
    selection_record = json.loads(manifest_path.read_text())
    declared = selection_record.get('files')
    if isinstance(declared, list):
        hashes = selection_record.get('sha256', {})
        if not isinstance(hashes, dict) or any(name not in hashes for name in declared):
            raise ValueError('Anonymous C7/D4 selection lacks member checksums')
        files = {name: hashes[name] for name in declared}
        if len(files) != len(declared):
            raise ValueError('Duplicate anonymous C7/D4 data path')
    elif isinstance(declared, dict):
        files = dict(declared)
    else:
        raise ValueError('Anonymous C7/D4 files must be a path list or hash mapping')
    if not files:
        raise ValueError('Empty anonymous C7/D4 file selection')
    if any(not isinstance(name, str) or not isinstance(value, str) or
           re.fullmatch(r'[0-9a-f]{64}', value) is None for name, value in files.items()):
        raise ValueError('Anonymous C7/D4 data requires SHA256 member checksums')
    files.setdefault(selection, None)
    for name, expected in sorted(files.items()):
        path = Path(name)
        if (path.is_absolute() or '..' in path.parts or '\\' in name or
                not name.startswith('data/retained_coordinates/')):
            raise ValueError('Unsafe anonymous C7/D4 data path: ' + name)
        source = root / path
        if (source.is_symlink() or not source.is_file() or
                not source.resolve().is_relative_to(root.resolve())):
            raise ValueError('Missing or linked anonymous C7/D4 data member: ' + name)
        raw = source.read_bytes()
        if expected is not None and sha(raw) != expected:
            raise ValueError('Anonymous C7/D4 data checksum mismatch: ' + name)
        members[name] = raw
    # The project owner's license identity is withheld only in the blind-review
    # export. Preserve every permission/disclaimer byte and the public original.
    license_name = 'studies/spectral_dimension_sensitivity_v1/core/_vendor/LICENSE'
    original = members[license_name]
    notice = b'Copyright (c) 2026 Pengcheng Yang'
    replacement = b'Copyright (c) 2026 [author identity withheld for anonymous review]'
    if original.count(notice) != 1:
        raise ValueError('Unexpected project-owned license identity')
    anonymous = original.replace(notice, replacement)
    overlay = json.loads((root / 'release/publication_derivatives_v6.json').read_text())
    binding = overlay.get('anonymous_exports', {}).get(license_name, {})
    if (binding.get('original_sha256') != sha(original) or
            binding.get('exported_sha256') != sha(anonymous)):
        raise ValueError('Anonymous license derivative is not explicitly bound')
    members[license_name] = anonymous
    members['ANONYMOUS_EXPORTS.json'] = (json.dumps({
        'scope': 'Blind-review export only; public source identities and original scientific locks remain unchanged.',
        'files': {license_name: {'original_sha256': sha(original),
                               'exported_sha256': sha(anonymous),
                               'change': 'Withhold the project author identity in the copyright notice; retain its year and all MIT permission and disclaimer text.'}},
        'mathematical_source_bytes_changed': False,
    }, indent=2)+'\n').encode()
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
    review = review_members(root, anon)
    review['README_REVIEW.txt'] = (
        'Anonymous review source and saved-evidence supplement\n'
        'Compile paper/main.tex from scratch with XeLaTeX + BibTeX, matching the locally tested engine family.\n'
        'The official template also supports pdfLaTeX.\n'
        'No manuscript PDF is supplied unless a fresh source-bound receipt exists.\n'
        'Read paper/OVERLEAF_README.txt and review_reproduction/README.md.\n'
        'The paper supplementary directory includes current full-precision endpoints,\n'
        'table/figure renderers, dimension/rank audit records and preserved stops.\n'
        'Historical validation and stage receipts remain historical, not current build claims.\n'
        'The 8/16-dimensional and independent orthogonal analyses are complete.\n'
        'C7 canonical-action family-risk LOFO is complete; all-image registration LOFO is not run.\n'
        'The retained-coordinate supplement includes the C7/D4 data selection, centering code,\n'
        'its existing spectral-kernel dependency and saved centering outcomes at repository-relative paths.\n'
        'S3 coordinate objects are not selected for this anonymous supplement.\n'
        'These files do not reconstruct all original raw activations; analyses remain opt-in.\n'
        'No models or experiments run as part of packaging. Author review remains pending.\n'
    ).encode()
    review['CONTENTS.sha256.json'] = (json.dumps(
        {name: sha(raw) for name, raw in sorted(review.items())}, indent=2)+'\n').encode()
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
              'final_pdf_review': 'Not repeated by packager; see github/release/PDF_QA_V6.json',
              'metadata': 'github/release/current_release_status.json'}
    (output / 'PACKAGE_CHECK.json').write_text(json.dumps(record, indent=2)+'\n')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(package(Path(__file__).resolve().parents[1], args.output.resolve()), indent=2))
