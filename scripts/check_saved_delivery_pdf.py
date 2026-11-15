"""Check an unchanged saved PDF with pdfplumber/pypdf; never compile or run a study."""
from pathlib import Path
import hashlib
import json
import re
import pdfplumber
from pypdf import PdfReader
from check_release import tex_dependencies

ROOT = Path(__file__).resolve().parents[1]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    pdf = ROOT / 'paper/main.pdf'
    dependencies = sorted(tex_dependencies(ROOT))
    errors = []
    pages = []
    outside = []
    fonts = set()
    with pdfplumber.open(pdf) as doc:
        for n, page in enumerate(doc.pages, 1):
            pages.append(page.extract_text() or '')
            for ch in page.chars:
                fonts.add(ch['fontname'])
                if ch['x0'] < -1 or ch['top'] < -1 or ch['x1'] > page.width + 1 or ch['bottom'] > page.height + 1:
                    outside.append(n)
    text = '\n'.join(pages)
    release = json.loads((ROOT/'release/release_manifest.json').read_text())
    expected_pages = release.get('expected_pdf_pages')
    if expected_pages is not None and len(pages) != expected_pages:
        errors.append('PDF page count differs from the final compiled revision')
    if '??' in text or '\ufffd' in text: errors.append('Unresolved reference or replacement glyph in extracted text')
    if re.search(r'Pengcheng|/Users/|/home/',text,re.I): errors.append('Identifying/private text in manuscript')
    if outside: errors.append('Text outside page bounds: ' + str(sorted(set(outside))))
    if not fonts: errors.append('No text fonts detected')
    metadata = PdfReader(pdf).metadata or {}
    author = str(metadata.get('/Author',''))
    if author and 'anonymous' not in author.lower(): errors.append('PDF author metadata is not anonymous')
    auxpath = ROOT/'paper/main.aux'
    aux = auxpath.read_text() if auxpath.exists() else ''
    def page_for(label):
        match = re.search(r'\\newlabel\{' + re.escape(label) + r'\}\{\{[^}]*\}\{(\d+)\}', aux)
        return int(match[1]) if match else None
    boundary = page_for('sec:main-end') or next((i+1 for i,t in enumerate(pages) if re.search(r'^Impact Statement\s*$',t,re.M)),None)
    table = page_for('tab:core-risk-summary') or next((i+1 for i,t in enumerate(pages) if re.search(r'Table\s+1[:.]',t)), None)
    if not boundary or not table or table > boundary: errors.append('Missing/outside-main main evidence table')
    if 'Table 32' not in text or 'dimension sensitivity' not in text.lower(): errors.append('Latest dimension sensitivity table not found')
    release = json.loads((ROOT/'release/release_manifest.json').read_text())
    report = {
        'pdf_path':'paper/main.pdf','pdf_sha256':sha(pdf),
        'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in dependencies},
        'total_pages':len(pages),'bytes':pdf.stat().st_size,
        'main_text_last_page':boundary,'main_table_pages':{'tab:core-risk-summary':table},
        'main_figure_pages':{'fig:context-followup':page_for('fig:context-followup')},
        'target_venue':'TMLR','template_status':'official_tmlr_unmodified',
        'submission_ready':release['submission_ready'],
        'pending_analyses':release['pending_analyses'],
        'submission_readiness_blockers':release['submission_readiness_blockers'],
        'author_review_status':release['author_review_status'],
        'checks':{'anonymous_author_metadata':author,'outside_page_text':sorted(set(outside)),'fonts':sorted(fonts),'table32_text_found':'Table 32' in text},
        'saved_compile_attestation':'COMPILE_CHECK.md',
        'fresh_compile_performed_for_packaging':False,
        'current_revision_compiled_before_packaging':True,
        'pdf_structural_check':'scripts/check_saved_delivery_pdf.py (pdfplumber/pypdf)',
        'scope':'Structural checks of the final compiled revision. Compile and rendered-page review are recorded separately in the revision report; this packaging checker does not compile or certify scientific validity.',
        'errors':errors}
    (ROOT/'release/pdf_qa.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'pages':len(pages),'errors':errors,'pdf_sha256':report['pdf_sha256']}))
    return 1 if errors else 0
if __name__ == '__main__': raise SystemExit(main())
