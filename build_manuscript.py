#!/usr/bin/env python3
"""Clean manuscript build only; never runs scientific code or a model."""
from pathlib import Path
import argparse, datetime, hashlib, json, os, shutil, subprocess

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, default=ROOT/'paper')
parser.add_argument('--build', type=Path, required=True)
parser.add_argument('--tex-bin', type=Path)
args = parser.parse_args()
env = os.environ.copy()
if args.tex_bin: env['PATH'] = str(args.tex_bin) + os.pathsep + env['PATH']
programs = {n: shutil.which(n, path=env['PATH']) for n in ['pdflatex','bibtex']}
if not all(programs.values()):
    raise SystemExit('BLOCKED: pdfLaTeX and BibTeX are required; no installation is attempted.')
if args.build.exists(): raise SystemExit('Use a new, empty build path; refusing to overwrite.')
shutil.copytree(args.source, args.build, ignore=shutil.ignore_patterns('main.pdf','main.bbl','main.aux','main.log','main.blg','main.out','*.pyc','__pycache__'))
start=datetime.datetime.now(datetime.timezone.utc).isoformat()
commands=[]
for index,command in enumerate([
    [programs['pdflatex'],'-no-shell-escape','-interaction=nonstopmode','-halt-on-error','-file-line-error','main.tex'],
    [programs['bibtex'],'main'],
    [programs['pdflatex'],'-no-shell-escape','-interaction=nonstopmode','-halt-on-error','-file-line-error','main.tex'],
    [programs['pdflatex'],'-no-shell-escape','-interaction=nonstopmode','-halt-on-error','-file-line-error','main.tex'],
]):
    with (args.build/f'pass_{index+1}.txt').open('wb') as log:
        result=subprocess.run(command,cwd=args.build,env=env,stdout=log,stderr=subprocess.STDOUT)
    commands.append({'command':command,'exit_code':result.returncode})
    if result.returncode:break
report={'started_utc':start,'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'programs':programs,'commands':commands,'inherited_bbl_or_pdf_used':False,
        'new_model_execution':False,'new_scientific_analysis':False,
        'status':'PASS' if len(commands)==4 and all(x['exit_code']==0 for x in commands) else 'FAIL'}
for name in ['main.pdf','main.bbl','references.bib']:
    p=args.build/name
    if p.exists():report[name+'_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
(args.build/'BUILD_RECORD.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
raise SystemExit(0 if report['status']=='PASS' else 1)
