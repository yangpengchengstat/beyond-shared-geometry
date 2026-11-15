"""Comparability flags use validation only and retain unmatched methods."""
from pathlib import Path
import sys
import json
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare import comparable, make_comparison, _entries


def payload(registration, test_value=1.):
    methods = []
    for name, error, norm in [('matched', .1, 2.), ('unmatched', .1 if registration == 'semantic' else .3, 2.)]:
        methods.append({'name': name,
            'validation_selected_predictor': {
                'pooled_changed_pair_mean_nmse': error,
                'pooled_mean_prediction_squared_norm_over_training_energy': norm},
            'fit': {'family': 'ridge', 'coefficient_frobenius_norm': 1., 'storage': {'parameter_bytes': 1024}},
            'evaluation': {'test': {'equal_action_mean_nmse': test_value}}})
    return {'job': {'cell': 'fixture__C7__6', 'component': 'compact', 'registration': registration},
            'result': {'tracks': {'generator_only': {'models': methods}}}}


class CompareTests(unittest.TestCase):
    def test_fixed_factor_floor_and_invalid_values(self):
        self.assertTrue(comparable(1., 1.25))
        self.assertFalse(comparable(1., 1.250001))
        self.assertTrue(comparable(0., 1e-15))
        self.assertFalse(comparable(None, .1))
        self.assertFalse(comparable(-.1, .1))
        self.assertFalse(comparable(float('nan'), .1))
        self.assertFalse(comparable(float('inf'), .1))

    def test_test_scores_cannot_change_comparability_or_remove_methods(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / 'checkpoints'
            directory.mkdir()
            for registration in ('semantic', 'consistent_swap01'):
                (directory / (registration + '.json')).write_text(json.dumps(payload(registration)))
            before = make_comparison(tmp)
            self.assertEqual(len(before['pairs']), 2)
            self.assertEqual(before['pairs_within_tolerance'], 1)
            self.assertFalse(before['matching_uses_test_outcomes'])
            for registration, score in (('semantic', 1e10), ('consistent_swap01', 1e-10)):
                (directory / (registration + '.json')).write_text(json.dumps(payload(registration, score)))
            after = make_comparison(tmp)
            self.assertEqual([(r['method'], r['within_fixed_validation_tolerance']) for r in before['pairs']],
                             [(r['method'], r['within_fixed_validation_tolerance']) for r in after['pairs']])
            self.assertEqual(len(after['pairs']), 2)

    def test_linear_coefficient_norm_blocks_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / 'checkpoints'; directory.mkdir()
            for registration in ('semantic', 'consistent_swap01'):
                item = payload(registration)
                if registration == 'consistent_swap01':
                    for method in item['result']['tracks']['generator_only']['models']:
                        method['fit']['coefficient_frobenius_norm'] = 3.
                (directory / (registration + '.json')).write_text(json.dumps(item))
            comparison = make_comparison(tmp)
            self.assertEqual(comparison['pairs_within_tolerance'], 0)
            self.assertTrue(all(r['coefficient_norm_comparable_if_linear'] is False for r in comparison['pairs']))

    def test_nonlinear_coefficients_are_not_claimed_operator_norms(self):
        item = payload('semantic')
        for method in item['result']['tracks']['generator_only']['models']:
            method['fit']['family'] = 'rff'
        self.assertTrue(all(entry['coefficient_frobenius_norm'] is None for _,entry in _entries(item)))

    def test_context_linear_norm_is_available_for_comparison(self):
        item = {'job': {'component': 'context'}, 'result': {'methods': [{
            'name': 'linear_transport_context',
            'validation_selected_predictor': {'pooled': {'mean_normalized_squared_error': .2,
                'mean_output_squared_norm_over_train_energy': .8}},
            'metadata': {'coefficient_frobenius_norm': 4.}, 'splits': {}}]}}
        entries = list(_entries(item))
        self.assertEqual(entries[0][1]['coefficient_frobenius_norm'], 4.)


if __name__ == '__main__':
    unittest.main()
