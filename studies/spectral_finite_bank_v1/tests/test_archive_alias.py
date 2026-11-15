"""Regression for the pinned archive's four-corner row names."""
from pathlib import Path
import copy
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import runner


class AliasTests(unittest.TestCase):
    def fixture(self):
        rows=[{'domain':'D4_corner','row_id':f'D4_corner/f/p{p}/lexical_D4_corner/s{s}',
               'family_id':'f','paraphrase_id':p,'state_id':s,'orbit_id':'lexical_D4_corner',
               'split':'test'} for p in (0,1) for s in range(4)]
        spec={'archive_row_domains':{'C7':'C7','D4':'D4_corner'},'partition':'test',
              'test_families':['f'],'paraphrase_ids':[0,1],'context_blocks_per_cell':2}
        return rows,spec

    def test_keeps_original_ids_and_indices(self):
        rows,spec=self.fixture();original=copy.deepcopy(rows)
        indices,context=runner._selection(rows,'D4',spec)
        self.assertEqual(indices,[[0,1,2,3],[4,5,6,7]])
        self.assertEqual(context[0]['orbit_id'],'lexical_D4_corner')
        self.assertEqual(context[0]['row_ids'],[r['row_id'] for r in rows[:4]])
        self.assertEqual(rows,original)

    def test_alias_does_not_accept_wrong_domain_or_renamed_id(self):
        for key,value in [('domain','D4_free8'),('domain','D4'),('orbit_id','lexical_D4'),
                          ('row_id','D4/f/p0/lexical_D4/s0')]:
            rows,spec=self.fixture();rows[0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):runner._selection(rows,'D4',spec)


if __name__=='__main__':unittest.main()
