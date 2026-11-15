"""Inspect the compiled anonymous manuscript; requires the optional paper extra."""
from pathlib import Path
import hashlib
import json
import re
import sys

import pymupdf

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    metadata = json.loads((ROOT / 'release/release_manifest.json').read_text())
    paper = ROOT / 'paper'
    pdf = paper / 'main.pdf'
    doc = pymupdf.open(pdf)
    errors = []
    text_pages = [page.get_text() for page in doc]
    text = '\n'.join(text_pages)
    from check_release import tex_dependencies
    sources = sorted(tex_dependencies(ROOT))
    stale = [p.relative_to(ROOT).as_posix() for p in sources if p.stat().st_mtime_ns > pdf.stat().st_mtime_ns]
    if stale:
        errors.append('Compiled PDF predates source files: ' + ', '.join(stale))
    aux = (paper / 'main.aux').read_text() if (paper / 'main.aux').is_file() else ''
    def label_page(name):
        match = re.search(r'\\newlabel\{' + re.escape(name) + r'\}\{\{[^}]*\}\{(\d+)\}', aux)
        return int(match.group(1)) if match else None
    boundary = label_page('sec:main-end')
    if boundary is None:
        boundary = next((i + 1 for i, value in enumerate(text_pages) if re.search(r'^Impact Statement\s*$', value, re.M)), None)
    if boundary is None:
        errors.append('Main-text boundary label is missing.')
    # TMLR has no fixed page cap; content and readability determine appropriate length.
    main_figures = {name: label_page(name) for name in ['fig:budget-frontier']}
    for name, page in main_figures.items():
        if page and boundary and page > boundary:
            errors.append(f'Main figure {name} occurs after the main-text boundary.')
    main_tables = {'tab:main-findings': label_page('tab:main-findings')}
    for name, page in main_tables.items():
        if page is None or boundary is None or page > boundary:
            errors.append(f'Core evidence table {name} is missing or outside the main text.')
    if 'Under review by ICML' in text or 'International Conference on Machine Learning' in text[:1500]:
        errors.append('ICML submission header remains in the TMLR draft.')
    if 'Title Suppressed' in text or '??' in text or '\ufffd' in text:
        errors.append('Suppressed running title, unresolved reference, or replacement glyph detected.')
    if re.search(r'Pengcheng|/Users/|/home/', text, re.I):
        errors.append('Identifying author or private host path detected in manuscript text.')
    author = doc.metadata.get('author', '')
    if author and 'anonymous' not in author.lower():
        errors.append('PDF author metadata is not anonymous.')
    outside = []
    for number, page in enumerate(doc, 1):
        for block in page.get_text('dict')['blocks']:
            for line in block.get('lines', []):
                for span in line.get('spans', []):
                    x0, y0, x1, y1 = span['bbox']
                    if x0 < 0 or y0 < 0 or x1 > page.rect.width + 1 or y1 > page.rect.height + 1:
                        outside.append(number)
    if outside:
        errors.append('Text lies outside page bounds on pages: ' + str(sorted(set(outside))))
    log = (paper / 'main.log').read_text(errors='replace') if (paper / 'main.log').is_file() else ''
    if re.search(r'Overfull \\[hv]box|Citation .+ undefined|Reference .+ undefined|Font shape .+ undefined', log):
        errors.append('LaTeX log contains an overflow, undefined reference/citation, or missing font shape.')
    fonts = sorted({font[3] for page in doc for font in page.get_fonts()})
    if not fonts:
        errors.append('No embedded text fonts were found in the compiled PDF.')
    report = {
        'pdf_path': 'paper/main.pdf', 'pdf_sha256': digest(pdf),
        'source_sha256': {p.relative_to(ROOT).as_posix(): digest(p) for p in sources},
        'main_text_last_page': boundary, 'main_figure_pages': main_figures,
        'main_table_pages': main_tables,
        'total_pages': len(doc), 'bytes': pdf.stat().st_size,
        'target_venue': 'TMLR', 'template_status': 'official_tmlr_unmodified',
        'submission_ready': metadata.get('submission_ready', False),
        'pending_analyses': metadata.get('pending_analyses', []),
        'submission_readiness_blockers': metadata.get('submission_readiness_blockers', []),
        'author_review_status': metadata.get('author_review_status'),
        'checks': {'anonymous_author_metadata': author, 'outside_page_text': sorted(set(outside)), 'fonts': fonts},
        'errors': errors,
        'scope': 'Automated structural checks of the TMLR manuscript; separate visual and final author review required. These checks do not establish scientific validity or author approval.',
    }
    (ROOT / 'release' / 'pdf_qa.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'pages': len(doc), 'main_text_last_page': boundary, 'errors': errors}))
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
