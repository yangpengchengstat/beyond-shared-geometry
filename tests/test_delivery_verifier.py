"""Packaging-only negative fixtures; no scientific dependencies or execution."""
import importlib.util, json, tempfile, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('delivery_verifier',ROOT/'scripts/verify_delivery.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
class DeliveryChecks(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
  for d in ['release','results','paper']: (self.root/d).mkdir()
  for name,data in {'release/current_files.json':{'files':{}},'release/source_manifest.json':{'files':{}},'results/registry.json':{'entries':[]},'release/current_release_status.json':{'submission_ready':False,'author_review_status':'pending'}}.items():
   (self.root/name).write_text(json.dumps(data))
  (self.root/'paper/main.tex').write_text('Document')
  (self.root/'paper/references.bib').write_text('')
 def tearDown(self):self.tmp.cleanup()
 def test_minimal_source(self):self.assertEqual(module.verify(self.root)['status'],'PASS')
 def test_checksum_mismatch(self):
  (self.root/'release/current_files.json').write_text(json.dumps({'files':{'paper/main.tex':'0'*64}}))
  self.assertEqual(module.verify(self.root)['status'],'FAIL')
 def test_missing_citation(self):
  (self.root/'paper/main.tex').write_text(r'\cite{absent}')
  self.assertEqual(module.verify(self.root)['status'],'FAIL')
 def test_unverified_pdf(self):
  (self.root/'paper/main.pdf').write_bytes(b'unverified')
  self.assertEqual(module.verify(self.root)['status'],'FAIL')
 def write_pdf_receipt(self):
  # Synthetic bytes test identity binding only, never a successful TeX build.
  (self.root/'paper/main.pdf').write_bytes(b'pdf-binding-fixture')
  record={'status':'PASS','pdf_path':'paper/main.pdf','pdf_sha256':module.digest(self.root/'paper/main.pdf'),'source_files':{n:module.digest(self.root/n) for n in ('paper/main.tex','paper/references.bib')}}
  (self.root/'release/current_compile.json').write_text(json.dumps(record))
  return record
 def test_current_pdf_bound_to_sources(self):
  self.write_pdf_receipt()
  self.assertEqual(module.verify(self.root)['status'],'PASS')
 def test_pdf_receipt_rejects_later_source_edit(self):
  self.write_pdf_receipt();(self.root/'paper/main.tex').write_text('Changed after build')
  self.assertEqual(module.verify(self.root)['status'],'FAIL')
 def test_pdf_receipt_must_cover_active_bibliography(self):
  record=self.write_pdf_receipt();del record['source_files']['paper/references.bib']
  (self.root/'release/current_compile.json').write_text(json.dumps(record))
  self.assertEqual(module.verify(self.root)['status'],'FAIL')
 def test_pdf_receipt_rejects_replaced_pdf(self):
  self.write_pdf_receipt();(self.root/'paper/main.pdf').write_bytes(b'another PDF')
  self.assertEqual(module.verify(self.root)['status'],'FAIL')
 def test_parent_path_rejected(self):
  with self.assertRaises(ValueError):module.safe(self.root,'../outside')
 def test_absolute_path_rejected(self):
  with self.assertRaises(ValueError):module.safe(self.root,'/outside')
if __name__=='__main__':unittest.main()
