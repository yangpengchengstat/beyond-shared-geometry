"""Archived schema regression tests; all arrays and texts are software fixtures."""
from pathlib import Path
import copy
import io
import json
import sys
import unittest
import zipfile
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run
from archive_rows import restore_rows, _canonical, _sha
from analysis import blocks


class ArchivedRowsTests(unittest.TestCase):
    def fixture(self, mutate=None):
        metadata=[{'domain':'D4_corner','row_id':f'D4_corner/f/p{p}/lexical_D4_corner/s{s}',
                   'family_id':'f','paraphrase_id':p,'state_id':s,'orbit_id':'lexical_D4_corner',
                   'split':'train'} for p in (0,1) for s in range(4)]
        full=[dict(r,text=f'Fixture {i} café') for i,r in enumerate(metadata)]
        merged={'dataset_sha256':_sha(_canonical(full)),
                'rows_sha256':_sha(b''.join(_canonical(r)+b'\n' for r in full)),
                'descriptor':{'dataset_sha256':_sha(_canonical(full))},'n_rows':8}
        stored=list(reversed(copy.deepcopy(full)))
        if mutate:mutate(metadata,stored,merged)
        raw=b''.join(_canonical(r)+b'\n' for r in stored)
        manifest={'status':'sealed','rows_path':'rows.jsonl','rows_sha256':_sha(raw),
                  'dataset_sha256':_sha(_canonical(stored))}
        buf=io.BytesIO()
        with zipfile.ZipFile(buf,'w') as z:
            z.writestr('fixture/datasets/D4/manifest.json',_canonical(manifest))
            z.writestr('fixture/datasets/D4/rows.jsonl',raw)
        buf.seek(0)
        return metadata,full,merged,buf

    def restore(self, mutate=None):
        metadata,full,merged,buf=self.fixture(mutate)
        with zipfile.ZipFile(buf) as z:
            restored,record=restore_rows(run.input_adapter,z,run.input_adapter.zip_index(z),
                {'archive_prefix':'fixture'},{'domain':'D4'},metadata,merged)
        return metadata,full,restored,record

    def test_restore_by_id_preserves_original_order_and_ids(self):
        metadata,full,got,record=self.restore()
        self.assertEqual(got,full);self.assertFalse(record['prompt_regeneration'])
        z=np.arange(8*3).reshape(8,3)
        result,_=blocks(z,metadata,'D4','train',['f'],row_domain='D4_corner')
        np.testing.assert_array_equal(result.reshape(8,3),z)
        self.assertNotIn('text',metadata[0])

    def test_modified_text_rejected_despite_resealed_dataset(self):
        with self.assertRaisesRegex(ValueError,'extraction dataset identity'):
            self.restore(lambda m,r,e:r[0].update(text='Changed'))

    def test_missing_extra_and_duplicate_rows_rejected(self):
        mutations=[lambda m,r,e:r.pop(),lambda m,r,e:r.append(dict(r[0])),
                   lambda m,r,e:r.append(dict(r[0],row_id='extra'))]
        for mutate in mutations:
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):self.restore(mutate)

    def test_metadata_and_extraction_identity_mismatch_rejected(self):
        for mutate in [lambda m,r,e:m[0].update(state_id=3),
                       lambda m,r,e:e.update(rows_sha256='0'*64),
                       lambda m,r,e:e['descriptor'].update(dataset_sha256='0'*64)]:
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):self.restore(mutate)

    def test_empty_prompt_rejected(self):
        with self.assertRaisesRegex(ValueError,'text or row identity'):
            self.restore(lambda m,r,e:r[0].update(text=''))

    def test_domain_alias_is_explicit_and_does_not_accept_d4_free(self):
        metadata,_,_,_=self.fixture()
        z=np.ones((8,3))
        with self.assertRaisesRegex(ValueError,'Mixed domains'):blocks(z,metadata,'D4','train')
        with self.assertRaisesRegex(ValueError,'Unsupported row domain'):
            blocks(z,metadata,'D4','train',row_domain='D4_free8')


if __name__=='__main__':unittest.main()
