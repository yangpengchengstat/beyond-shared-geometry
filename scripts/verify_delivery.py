#!/usr/bin/env python3
"""Read-only stdlib checks; never imports scientific workers or runs a compiler."""
from pathlib import Path
import ast, hashlib, json, re, sys
ROOT=Path(__file__).resolve().parents[1]
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def safe(root,rel):
 p=Path(rel)
 if p.is_absolute() or '..' in p.parts:raise ValueError('Nonrelative manifest path: '+rel)
 q=root/p
 if q.is_symlink() or not q.resolve().is_relative_to(root.resolve()):raise ValueError('Escaping path: '+rel)
 return q
def verify_current_pdf(root,required_sources):
 """A PDF is current only with a successful, hash-bound compilation receipt.

 This checks the receipt's consistency, not whether a compiler was actually run.
 Only the build workflow may create that receipt after a real successful build.
 """
 pdf=root/'paper/main.pdf'
 if not pdf.exists():return []
 receipt=root/'release/current_compile.json'
 if not receipt.is_file():return ['Unverified active manuscript PDF: missing current_compile.json']
 try:
  record=json.loads(receipt.read_text())
  if record.get('status')!='PASS' or record.get('pdf_path')!='paper/main.pdf':
   return ['Unverified active manuscript PDF: unsuccessful or mismatched receipt']
  errors=[]
  if record.get('pdf_sha256')!=digest(pdf):errors.append('Active PDF checksum mismatch')
  sources=record.get('source_files',{})
  if not isinstance(sources,dict):return ['Current compile source_files is not a mapping']
  missing=required_sources-set(sources)
  errors+=['Current compile receipt missing source '+p for p in sorted(missing)]
  for rel,h in sources.items():
   p=safe(root,rel)
   if not p.is_file() or digest(p)!=h:errors.append('Current compile source mismatch '+rel)
  return errors
 except (ValueError,KeyError,TypeError,OSError) as e:
  return ['Invalid current compile receipt: '+str(e)]
def verify(root):
 manifest=json.loads((root/'release/current_files.json').read_text())
 errors=[]
 for rel,h in manifest['files'].items():
  p=safe(root,rel)
  if not p.is_file() or digest(p)!=h:errors.append('Changed/missing: '+rel)
 syntax_count=0
 for p in root.rglob('*.py'):
  if 'history' in p.relative_to(root).parts:continue
  try:ast.parse(p.read_text(),filename=str(p.relative_to(root)));syntax_count+=1
  except SyntaxError as e:errors.append(str(e))
 locked=json.loads((root/'release/source_manifest.json').read_text())['files']
 for rel,h in locked.items():
  p=safe(root,rel)
  if not p.is_file() or digest(p)!=h:errors.append('Scientific source lock: '+rel)
 registry=json.loads((root/'results/registry.json').read_text())['entries']
 for item in registry:
  p=safe(root,item['path'])
  if not p.is_file() or digest(p)!=item['sha256'] or p.stat().st_size!=item['bytes']:errors.append('Frozen evidence: '+item['path'])
 paper=root/'paper';pending=[paper/'main.tex'];seen=set();labels=[];refs=[];cites=set();graphics=[]
 while pending:
  p=pending.pop()
  if p in seen:continue
  seen.add(p)
  if not p.is_file():errors.append('Missing TeX '+str(p.relative_to(root)));continue
  s=re.sub(r'(?<!\\)%[^\n]*','',p.read_text())
  labels+=re.findall(r'\\label\{([^{}]+)\}',s)
  for names in re.findall(r'\\(?:[cC]ref|eqref|ref)\*?\{([^{}]+)\}',s):refs+=names.split(',')
  for names in re.findall(r'\\cite\w*\*?(?:\[[^\]]*\]){0,2}\{([^{}]+)\}',s):cites.update(names.split(','))
  for n in re.findall(r'\\(?:input|include)\{([^{}]+)\}',s):
   q=safe(paper,n if Path(n).suffix else n+'.tex');pending.append(q)
  for n in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^{}]+)\}',s):
   q=safe(paper,n)
   candidates=[Path(str(q)+suffix) for suffix in ('','.pdf','.png','.jpg')]
   found=next((p for p in candidates if p.is_file()),None)
   if found is None:errors.append('Missing graphic '+n)
   else:graphics.append(found)
 keys=set(re.findall(r'@\w+\s*\{\s*([^,\s]+)',(paper/'references.bib').read_text()))
 errors+=['Missing citation '+k for k in sorted(cites-keys)]
 errors+=['Missing label '+k for k in sorted(set(refs)-set(labels))]
 errors+=['Duplicate label '+k for k in sorted(set(labels)) if labels.count(k)>1]
 status=json.loads((root/'release/current_release_status.json').read_text())
 if status['submission_ready'] or status['author_review_status']!='pending':errors.append('Unexpected readiness promotion')
 build_sources={str(p.relative_to(root)) for p in seen|set(graphics)}
 build_sources|={str(p.relative_to(root)) for p in paper.iterdir() if p.is_file() and p.suffix in ('.bib','.sty','.bst')}
 errors+=verify_current_pdf(root,build_sources)
 return {'status':'FAIL' if errors else 'PASS','scope':'hashes, frozen identities, Python syntax, static TeX references and any current-PDF source binding only','files_checked':len(manifest['files']),'scientific_source_locks':len(locked),'frozen_registry_entries':len(registry),'python_files_parsed':syntax_count,'active_tex_files':len(seen),'citation_keys':len(cites),'labels':len(labels),'errors':errors,'software_tests_run':False,'numerical_analysis_run':False,'compile':status.get('normal_clean_build','BLOCKED'),'submission_ready':False}
if __name__=='__main__':
 try:result=verify(ROOT)
 except Exception as e:result={'status':'FAIL','errors':[str(e)]}
 print(json.dumps(result,indent=2));sys.exit(0 if result['status']=='PASS' else 1)
