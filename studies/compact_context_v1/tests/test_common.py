"""Independent action algebra, pairing and leakage regression checks."""
import copy
import json
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import action_catalog, make_cell, development_subset


def fixture(domain="C7"):
    states = {"C7": 7, "D4": 4, "S3": 6}[domain]
    rows = []
    for split in ("train", "validation", "pilot", "test", "support", "reserved"):
        for p in (0, 1):
            for state in range(states):
                rows.append(dict(row_id=f"{split}/{p}/{state}", family_id=f"f_{split}",
                                 orbit_id="orbit", state_id=state, paraphrase_id=p,
                                 split=split, domain=domain))
    z = np.random.default_rng(1021).normal(size=(len(rows), 16))
    return z, rows


class CommonTests(unittest.TestCase):
    def test_literal_generator_semantics(self):
        _, _, generators, tables, _ = action_catalog("C7")
        self.assertEqual(tables[generators[0]], (1, 2, 3, 4, 5, 6, 0))
        _, _, generators, tables, _ = action_catalog("D4")
        self.assertEqual(tables[generators[0]], (1, 2, 3, 0))
        self.assertEqual(tables[generators[1]], (1, 0, 3, 2))
        r, s = (tables[g] for g in generators)
        self.assertNotEqual(tuple(s[r[i]] for i in range(4)), tuple(r[s[i]] for i in range(4)))

    def test_every_word_and_conjugate_preserves_composition(self):
        for domain in ("C7", "D4", "S3"):
            with self.subTest(domain=domain):
                z, rows = fixture(domain)
                original = make_cell(z, rows, domain)
                alternate = make_cell(z, rows, domain, "consistent_swap01")
                n = len(original.tables[original.identity])
                self.assertNotEqual(set(original.tables.values()), set(alternate.tables.values()))
                self.assertEqual(len(set(original.tables.values())), len(original.elements))
                for target, word in original.words.items():
                    for cell in (original, alternate):
                        result = tuple(range(n))
                        for generator in word:
                            result = tuple(cell.tables[generator][result[i]] for i in range(n))
                        self.assertEqual(result, cell.tables[target])
                for a in original.elements:
                    for b in original.elements:
                        product = tuple(original.tables[b][original.tables[a][i]] for i in range(n))
                        c = next(g for g in original.elements if original.tables[g] == product)
                        self.assertEqual(alternate.tables[c],
                                         tuple(alternate.tables[b][alternate.tables[a][i]] for i in range(n)))
                for g in original.elements:
                    self.assertEqual(sum(i == j for i, j in enumerate(original.tables[g])),
                                     sum(i == j for i, j in enumerate(alternate.tables[g])))
                    for cell in (original, alternate):
                        for i, target in enumerate(cell.targets[g]):
                            self.assertEqual(rows[target]["split"], rows[i]["split"])
                            self.assertEqual(rows[target]["state_id"], cell.tables[g][rows[i]["state_id"]])

    def test_s3_family_reuse_is_allowed_only_across_distinct_orbits(self):
        z, rows = fixture("S3")
        for row in rows:
            row["family_id"] = "shared_family"
            row["orbit_id"] = "orbit_" + row["split"]
        make_cell(z, rows, "S3")
        for row in rows:
            row["orbit_id"] = "same_orbit"
        with self.assertRaisesRegex(ValueError, "crosses partitions"):
            make_cell(z, rows, "S3")

    def test_development_subsetting_removes_evaluated_coordinates(self):
        z, rows = fixture()
        kept, metadata = development_subset(z, rows)
        self.assertEqual(set(r["split"] for r in metadata), {"train", "validation", "pilot"})
        self.assertEqual(len(kept), 42)
        excluded = np.array([r["split"] not in ("train", "validation", "pilot") for r in rows])
        changed = z.copy()
        changed[excluded] = 1e120
        other, other_metadata = development_subset(changed, rows)
        np.testing.assert_array_equal(kept, other)
        self.assertEqual(metadata, other_metadata)
        make_cell(kept, metadata, "C7")

    def test_incomplete_or_duplicate_orbit_is_rejected(self):
        z, rows = fixture()
        with self.assertRaisesRegex(ValueError, "Incomplete state orbit"):
            make_cell(z[1:], rows[1:], "C7")
        duplicated = copy.deepcopy(rows)
        duplicated[1]["row_id"] = duplicated[0]["row_id"]
        with self.assertRaisesRegex(ValueError, "Duplicate row"):
            make_cell(z, duplicated, "C7")
        duplicated = copy.deepcopy(rows)
        duplicated[1]["state_id"] = duplicated[0]["state_id"]
        with self.assertRaisesRegex(ValueError, "Duplicate context/state"):
            make_cell(z, duplicated, "C7")

    def test_bad_state_partition_registration_and_finite_input_rejected(self):
        z, rows = fixture()
        for value in (-1, 7, 0.5):
            damaged = copy.deepcopy(rows)
            damaged[0]["state_id"] = value
            with self.subTest(state=value), self.assertRaises(ValueError):
                make_cell(z, damaged, "C7")
        unknown = copy.deepcopy(rows)
        for row in unknown:
            if row["split"] == "test":
                row["split"] = "unknown"
        with self.assertRaisesRegex(ValueError, "Unknown partition"):
            make_cell(z, unknown, "C7")
        with self.assertRaisesRegex(ValueError, "Unknown action"):
            make_cell(z, rows, "C7", "unregistered")
        damaged = z.copy()
        damaged[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            make_cell(damaged, rows, "C7")
        with self.assertRaisesRegex(ValueError, "finite"):
            make_cell(z[:-1], rows, "C7")
        train_only = [i for i, r in enumerate(rows) if r["split"] == "train"]
        with self.assertRaisesRegex(ValueError, "validation"):
            make_cell(z[train_only], [rows[i] for i in train_only], "C7")


if __name__ == "__main__":
    unittest.main()
