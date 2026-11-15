from bsg.data.c7 import TEMPLATES as C7_TEMPLATES, build_dataset as build_c7
from bsg.data.d4 import TEMPLATES as D4_TEMPLATES, build_dataset as build_d4
from bsg.data.splits import split_templates
from bsg.data.splits import split_templates_three_way


def test_template_splits_have_no_leakage():
    for frame in [build_c7(), build_d4()]:
        train_mask, test_mask, train_ids, test_ids = split_templates(frame, seed=42, train_fraction=0.70)
        assert set(train_ids).isdisjoint(test_ids)
        assert train_mask.any()
        assert test_mask.any()


def test_final_three_way_template_splits_are_exact_disjoint_and_deterministic():
    for frame, order in ((build_c7(), 7), (build_d4(), 4)):
        assert frame.template_id.nunique() == 40
        for seed in (101, 303, 909, 2701, 4243):
            first = split_templates_three_way(
                frame, seed=seed, train_templates=28, validation_templates=6, test_templates=6
            )
            second = split_templates_three_way(
                frame, seed=seed, train_templates=28, validation_templates=6, test_templates=6
            )
            assert first.identity == second.identity
            assert (len(first.train_ids), len(first.validation_ids), len(first.test_ids)) == (28, 6, 6)
            assert (first.train_mask.sum(), first.validation_mask.sum(), first.test_mask.sum()) == (28 * order, 6 * order, 6 * order)
            assert set(first.train_ids).isdisjoint(first.validation_ids)
            assert set(first.train_ids).isdisjoint(first.test_ids)
            assert set(first.validation_ids).isdisjoint(first.test_ids)


def test_final_template_pools_have_one_placeholder_and_no_normalized_duplicates():
    for templates in (C7_TEMPLATES, D4_TEMPLATES):
        assert len(templates) == 40
        assert all(template.count("{}") == 1 for template in templates)
        normalized = [" ".join(template.lower().split()) for template in templates]
        assert len(set(normalized)) == len(normalized)
