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
   q=safe(paper,n);graphics.append(q)
   if not any(Path(str(q)+suffix).is_file() for suffix in ('','.pdf','.png','.jpg')):errors.append('Missing graphic '+n)
 keys=set(re.findall(r'@\w+\s*\{\s*([^,\s]+)',(paper/'references.bib').read_text()))
 errors+=['Missing citation '+k for k in sorted(cites-keys)]
 errors+=['Missing label '+k for k in sorted(set(refs)-set(labels))]
 errors+=['Duplicate label '+k for k in sorted(set(labels)) if labels.count(k)>1]
 status=json.loads((root/'release/current_release_status.json').read_text())
 if status['submission_ready'] or status['author_review_status']!='pending':errors.append('Unexpected readiness promotion')
 if (paper/'main.pdf').exists():errors.append('Unverified active manuscript PDF')
 return {'status':'FAIL' if errors else 'PASS','scope':'hashes, frozen identities, Python syntax and static TeX references only','files_checked':len(manifest['files']),'scientific_source_locks':len(locked),'frozen_registry_entries':len(registry),'python_files_parsed':syntax_count,'active_tex_files':len(seen),'citation_keys':len(cites),'labels':len(labels),'errors':errors,'software_tests_run':False,'numerical_analysis_run':False,'compile':'BLOCKED','submission_ready':False}
if __name__=='__main__':
 try:result=verify(ROOT)
 except Exception as e:result={'status':'FAIL','errors':[str(e)]}
 print(json.dumps(result,indent=2));sys.exit(0 if result['status']=='PASS' else 1)
