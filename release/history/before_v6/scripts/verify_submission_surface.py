#!/usr/bin/env python3
"""Read-only, stdlib verification of the current manuscript and packaged evidence.

This checks source structure and saved-record coverage; it does not compile TeX,
execute scientific workers, recompute numerical results, or certify a final PDF.
Prints JSON to stdout. Exit 0 means all configured static checks passed.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def escaped(text: str, index: int) -> bool:
    j = index - 1
    while j >= 0 and text[j] == '\\':
        j -= 1
    return (index - j - 1) % 2 == 1


def strip_comments(text: str) -> str:
    """Blank comments while preserving offsets/line numbers and escaped percent."""
    chars = list(text)
    i = 0
    while i < len(text):
        if text[i] == '%' and not escaped(text, i):
            j = text.find('\n', i)
            if j < 0:
                j = len(text)
            chars[i:j] = ' ' * (j - i)
            i = j
        else:
            i += 1
    return ''.join(chars)


def group_at(text: str, index: int, opener='{', closer='}'):
    while index < len(text) and text[index].isspace():
        index += 1
    if index >= len(text) or text[index] != opener:
        return None
    start = index
    level = 1
    index += 1
    while index < len(text):
        if not escaped(text, index):
            if text[index] == opener:
                level += 1
            elif text[index] == closer:
                level -= 1
                if level == 0:
                    return text[start + 1:index], index + 1
        index += 1
    raise ValueError('Unbalanced TeX argument at offset ' + str(start))


def commands(text: str):
    """Yield literal control words and balanced optional/first mandatory arguments."""
    for match in re.finditer(r'\\([A-Za-z]+)(\*)?', text):
        if escaped(text, match.start()):
            continue
        pos = match.end()
        options = []
        while True:
            option = group_at(text, pos, '[', ']')
            if option is None:
                break
            options.append(option[0])
            pos = option[1]
        arg = group_at(text, pos)
        yield {'name': match.group(1), 'start': match.start(), 'options': options,
               'arg': arg[0] if arg else None, 'end': arg[1] if arg else pos}


def safe_file(base: Path, value: str, suffix='') -> Path:
    p = Path(value)
    if p.is_absolute() or '..' in p.parts or '\\' in value or '#' in value:
        raise ValueError('Nonportable or dynamic input path: ' + value)
    if suffix and not p.suffix:
        p = p.with_suffix(suffix)
    target = base / p
    if target.is_symlink() or not target.resolve().is_relative_to(base.resolve()):
        raise ValueError('Input escapes project: ' + value)
    return target


def tex_closure(paper: Path):
    files = {}
    occurrences = []
    errors = []
    stack = []

    def visit(path):
        rel = str(path.relative_to(paper))
        if path in stack:
            errors.append('Recursive TeX input: ' + rel)
            return
        if not path.is_file():
            errors.append('Missing TeX input: ' + rel)
            return
        text = strip_comments(path.read_text())
        files[rel] = text
        occurrences.append((rel, text))
        stack.append(path)
        for cmd in commands(text):
            if cmd['name'] in ('input', 'include'):
                if cmd['arg'] is None:
                    errors.append('Unbraced/dynamic input requires review: ' + rel)
                    continue
                try:
                    visit(safe_file(paper, cmd['arg'], '.tex'))
                except ValueError as e:
                    errors.append(rel + ': ' + str(e))
        stack.pop()

    visit(paper / 'main.tex')
    return files, occurrences, errors


def float_checks(text: str, source: str):
    """Inspect literal table/figure environments without interpreting their content."""
    stack = []
    checks = []
    errors = []
    for cmd in commands(text):
        env = cmd['arg']
        if cmd['name'] == 'begin' and env in ('table', 'table*', 'figure', 'figure*'):
            stack.append((env, cmd['start'], cmd['end']))
        elif cmd['name'] == 'end' and env in ('table', 'table*', 'figure', 'figure*'):
            if not stack or stack[-1][0] != env:
                errors.append(source + ': unmatched float end ' + str(env))
                continue
            _, start, begin_end = stack.pop()
            body = text[begin_end:cmd['start']]
            body_commands = list(commands(body))
            captions = [c for c in body_commands if c['name'] == 'caption']
            labels = [c['arg'] for c in body_commands if c['name'] == 'label']
            kind = 'table' if env.startswith('table') else 'figure'
            if kind == 'table':
                content = [c['start'] for c in body_commands if c['name'] in ('input', 'include') or
                           (c['name'] == 'begin' and c['arg'] in ('tabular', 'tabular*', 'tabularx', 'longtable'))]
                correct = len(captions) == 1 and bool(content) and captions[0]['start'] < min(content)
            else:
                content = [c['start'] for c in body_commands if c['name'] == 'includegraphics']
                correct = len(captions) == 1 and bool(content) and captions[0]['start'] > max(content)
            check = {'source': source, 'line': text[:start].count('\n') + 1,
                     'kind': kind, 'labels': labels, 'caption_order_correct': correct}
            checks.append(check)
            if not correct:
                errors.append(source + ':' + str(check['line']) + ': incorrect or uncheckable ' + kind + ' caption order')
    if stack:
        errors.append(source + ': unclosed float environment')
    return checks, errors


def inspect_tex(paper: Path, expected_tables=None, expected_figures=None):
    files, occurrences, errors = tex_closure(paper)
    labels, refs, anchors, links, cites, floats = [], [], [], [], [], []
    bib_files, graphics = set(), set()
    for source, text in occurrences:
        fs, es = float_checks(text, source)
        floats.extend(fs)
        errors.extend(es)
        for cmd in commands(text):
            name, arg = cmd['name'], cmd['arg']
            if name == 'label' and arg is not None:
                labels.append(arg.strip())
            elif name in ('ref', 'eqref', 'pageref', 'autoref', 'nameref', 'cref', 'Cref', 'crefrange', 'Crefrange') and arg:
                refs.extend(k.strip() for k in arg.split(','))
                if name.lower() == 'crefrange':
                    second = group_at(text, cmd['end'])
                    if second:
                        refs.append(second[0].strip())
            elif name == 'hyperref' and cmd['options']:
                refs.extend(k.strip() for k in cmd['options'][0].split(','))
            elif name == 'hypertarget' and arg:
                anchors.append(arg)
            elif name == 'hyperlink' and arg:
                links.append(arg)
            elif name.startswith('cite') and arg:
                cites.extend(k.strip() for k in arg.split(',') if k.strip() != '*')
            elif name == 'bibliography' and arg:
                for value in arg.split(','):
                    try:
                        bib_files.add(safe_file(paper, value.strip(), '.bib'))
                    except ValueError as e:
                        errors.append(str(e))
            elif name == 'includegraphics' and arg:
                try:
                    target = safe_file(paper, arg)
                    choices = [target] if target.suffix else [target.with_suffix(s) for s in ('.pdf', '.png', '.jpg', '.jpeg')]
                    found = next((p for p in choices if p.is_file()), None)
                    if found is None:
                        errors.append('Missing graphic: ' + arg)
                    else:
                        graphics.add(str(found.relative_to(paper)))
                except ValueError as e:
                    errors.append(str(e))
    keys = []
    for path in sorted(bib_files):
        if not path.is_file():
            errors.append('Missing bibliography: ' + str(path.relative_to(paper)))
            continue
        keys.extend(re.findall(r'@(?!(?:comment|string|preamble)\b)\w+\s*\{\s*([^,\s]+)', strip_comments(path.read_text()), re.I))
    errors.extend('Missing reference label: ' + k for k in sorted(set(refs) - set(labels)))
    errors.extend('Duplicate label: ' + k for k, count in sorted(Counter(labels).items()) if count > 1)
    errors.extend('Missing hyperlink target: ' + k for k in sorted(set(links) - set(anchors)))
    errors.extend('Missing citation key: ' + k for k in sorted(set(cites) - set(keys)))
    errors.extend('Duplicate bibliography key: ' + k for k, count in sorted(Counter(keys).items()) if count > 1)
    nt = sum(f['kind'] == 'table' for f in floats)
    nf = sum(f['kind'] == 'figure' for f in floats)
    if expected_tables is not None and nt != expected_tables:
        errors.append(f'Expected {expected_tables} active tables; found {nt}')
    if expected_figures is not None and nf != expected_figures:
        errors.append(f'Expected {expected_figures} active figures; found {nf}')
    return {'errors': errors, 'active_tex_files': sorted(files), 'tables': nt, 'figures': nf,
            'floats': floats, 'labels': len(labels), 'citation_keys': len(set(cites)), 'graphics': sorted(graphics)}


def check_saved_completion(root: Path, release: dict):
    expected = {'independent_orthogonality': 'complete', 'spectral_dimension_8_16': 'complete',
                'c7_canonical_family_risk_lofo': 'complete', 'all_image_registration_lofo': 'not_run'}
    states = release.get('scientific_completion')
    if states is None:
        return {'status': 'BLOCKED', 'reason': 'Current release metadata lacks scientific_completion; historical study snapshots are not substituted.'}
    errors = [f'{k}: expected {v}, found {states.get(k)!r}' for k, v in expected.items() if states.get(k) != v]
    coverage = {}
    for name, rel, field in (
            ('independent_orthogonality', 'results/orthogonal_comparator_v1/cell_summary.csv', 'status'),
            ('spectral_dimension_8_16', 'results/spectral_dimension_sensitivity_v1/cell_summary.csv', 'canonical_status')):
        path = root / rel
        if not path.is_file():
            errors.append('Missing saved coverage record: ' + rel)
            continue
        with path.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        dimensions = dict(Counter(row['dimension'] for row in rows))
        good = len(rows) == 96 and dimensions == {'8': 48, '16': 48} and all(row[field] == 'complete' for row in rows)
        coverage[name] = {'rows': len(rows), 'dimensions': dimensions, 'complete': good}
        if not good:
            errors.append('Saved coverage/status disagrees: ' + name)
    family = root / 'results/c7_family_influence_v1/summary.json'
    if family.is_file():
        data = json.loads(family.read_text())
        cells = data.get('cell_summaries', [])
        good = len(cells) == 24 and all(c.get('completed') == 64 and c.get('failed') == 0 for c in cells)
        coverage['c7_canonical_family_risk_lofo'] = {'cells': len(cells), 'complete': good}
        if not good:
            errors.append('C7 family-risk saved coverage is incomplete')
    else:
        errors.append('Missing saved C7 family-risk summary')
    return {'status': 'FAIL' if errors else 'PASS', 'metadata': states, 'saved_record_coverage': coverage,
            'errors': errors, 'scope': 'Saved row counts/status fields only; no experiment or numerical recomputation.'}


def check_evidence(root: Path):
    evidence = root / 'paper/supplementary/verification_evidence_v3'
    primary = evidence / 'source_manifest.json'
    if not primary.is_file():
        return {'status': 'BLOCKED', 'reason': 'Manuscript evidence mirror/source_manifest.json is not yet present.'}
    required = {'dimension_rank/' + s for s in ('dimension_rank_audit.csv', 'dimension_sensitivity_existing.csv',
                'dimension_sensitivity_summary.csv', 'dimension_sensitivity_coverage.csv')}
    primary_data = json.loads(primary.read_text())
    actual = {s['path'] for s in primary_data.get('sources', [])}
    errors = ['Missing declared required evidence: ' + s for s in sorted(required - actual)]
    checked = []
    # Validate only manifests declaring packaged identities, not historical manifests
    # describing unavailable source activations or intentionally logical locations.
    for manifest in sorted(evidence.rglob('*.json')):
        if manifest.stat().st_size > 5_000_000:
            continue
        try:
            data = json.loads(manifest.read_text())
        except (UnicodeError, json.JSONDecodeError):
            continue
        entries = data.get('sources', []) if isinstance(data, dict) else []
        for entry in entries:
            if not isinstance(entry, dict) or 'packaged_sha256' not in entry:
                continue
            try:
                path = safe_file(manifest.parent, entry['path'])
                if not path.is_file():
                    errors.append('Missing evidence: ' + entry['path'])
                elif digest(path) != entry['packaged_sha256']:
                    errors.append('Evidence hash mismatch: ' + entry['path'])
                elif 'packaged_bytes' in entry and path.stat().st_size != entry['packaged_bytes']:
                    errors.append('Evidence size mismatch: ' + entry['path'])
                checked.append(str(path.relative_to(root)))
            except (KeyError, ValueError) as e:
                errors.append(str(e))
    return {'status': 'FAIL' if errors else 'PASS', 'files_checked': checked, 'errors': errors,
            'scope': 'Portable manifest paths and packaged identities; original-to-sanitized numeric equivalence is recorded separately.'}


def verify(root: Path):
    root = root.resolve()
    tex = inspect_tex(root / 'paper', expected_tables=35, expected_figures=9)
    errors = list(tex['errors'])
    main = strip_comments((root / 'paper/main.tex').read_text())
    main_commands = list(commands(main))
    classes = [c for c in main_commands if c['name'] == 'documentclass']
    packages = [c for c in main_commands if c['name'] == 'usepackage' and c['arg'] and 'tmlr' in c['arg'].split(',')]
    authors = [c['arg'] for c in main_commands if c['name'] == 'author']
    if len(classes) != 1 or classes[0]['arg'] != 'article' or '10pt' not in ','.join(classes[0]['options']).split(','):
        errors.append('Expected official 10pt article setup')
    if len(packages) != 1 or packages[0]['options']:
        errors.append('Expected anonymous tmlr package without accepted/preprint options')
    if authors != ['Anonymous authors']:
        errors.append('Author field is not the expected anonymous submission form')
    if any(c['name'] == 'raggedbottom' for text in tex_closure(root / 'paper')[0].values() for c in commands(text)):
        errors.append('raggedbottom overrides official page layout')
    provenance = json.loads((root / 'provenance/template_source.json').read_text())
    templates = []
    for name in ('tmlr.sty', 'tmlr.bst', 'fancyhdr.sty'):
        path = root / 'paper' / name
        target = provenance.get('support_sha256', {}).get(name)
        ok = bool(target) and path.is_file() and digest(path) == target
        templates.append({'file': 'paper/' + name, 'matches_recorded_official_hash': ok})
        if not ok:
            errors.append('Official template identity not verified: ' + name)
    release = json.loads((root / 'release/current_release_status.json').read_text())
    if release.get('new_model_execution') is not False or release.get('new_scientific_analysis') is not False:
        errors.append('Release must preserve no-new-execution flags')
    completion = check_saved_completion(root, release)
    evidence = check_evidence(root)
    errors.extend(completion.get('errors', []))
    errors.extend(evidence.get('errors', []))
    blocked = [item['reason'] for item in (completion, evidence) if item['status'] == 'BLOCKED']
    return {'status': 'FAIL' if errors else ('BLOCKED' if blocked else 'PASS'),
            'scope': 'Read-only static manuscript surface, template identity, portable evidence and saved completion records; not PDF acceptance.',
            'tex': tex, 'template_provenance': templates, 'scientific_completion': completion,
            'packaged_evidence': evidence, 'errors': errors, 'blocked': blocked,
            'compiler_executed': False, 'scientific_analysis_executed': False, 'files_modified': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        result = verify(args.root)
    except (OSError, ValueError, KeyError) as e:
        result = {'status': 'FAIL', 'errors': [str(e)], 'compiler_executed': False, 'scientific_analysis_executed': False}
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
