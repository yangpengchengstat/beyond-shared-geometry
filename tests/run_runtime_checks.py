from __future__ import annotations

import importlib.util
from pathlib import Path


RUNTIME_CHECKS = [
    ("test_afs_d.py", "test_afs_d_config_and_matrix_are_frozen"),
    ("test_afs_d.py", "test_afs_d_generator_and_operator_units_are_aggregated_without_pseudoreplication"),
    ("test_afs_d.py", "test_afs_d_pairing_keeps_depth_fixed_and_bootstrap_is_deterministic"),
    ("test_afs_d.py", "test_afs_d_dimension_mismatched_pairing_is_rejected"),
    ("test_afs_d.py", "test_afs_d_outcome_rules_cover_a_b_c_d_and_mixed"),
    ("test_afs_d.py", "test_afs_d_output_guard_and_resume_require_no_model_when_extractions_are_complete"),
    ("test_afs_d.py", "test_afs_d_rank_diagnostics_select_frozen_dimensions_without_metrics"),
    ("test_afs_d.py", "test_afs_d_insufficient_rank_aborts_before_analysis"),
    ("test_afs_d.py", "test_afs_d_observed_pca_rank_guard_remains_active"),
    ("test_afs_d.py", "test_afs_d_internal_metadata_uses_rank_matched_dimensions"),
    ("test_algebra_falsification.py", "test_abc_only_execution_manifest_preserves_frozen_matrix_and_omits_d"),
    ("test_algebra_falsification.py", "test_afs_tier_matrices_and_rng_namespaces_are_exact"),
    ("test_algebra_falsification.py", "test_c7_generator_indexing_cycle_and_train_only_fit"),
    ("test_algebra_falsification.py", "test_d4_canonical_words_row_order_and_wrong_words_are_exact"),
    ("test_algebra_falsification.py", "test_permutation_null_excludes_exact_automorphisms_and_preserves_operators"),
    ("test_algebra_falsification.py", "test_scramble_preserves_multisets_splits_pca_and_never_reads_heldout_rows"),
    ("test_algebra_falsification.py", "test_d4_scramble_uses_exact_action_for_every_source_and_element"),
    ("test_algebra_falsification.py", "test_embedding_layer_zero_architecture_and_span_contract"),
    ("test_algebra_falsification.py", "test_primary_cell_coverage_rejects_missing_cells"),
    ("test_afs_historical_isolation.py", "test_historical_import_validates_hash_and_cannot_write_source"),
    ("test_afs_historical_isolation.py", "test_historical_import_rejects_bad_checksum_before_loading"),
    ("test_checkpoint_hydration.py", "test_fresh_and_restored_checkpoint_state_are_exact_and_aggregate_schemas_survive"),
    ("test_checkpoint_hydration.py", "test_fresh_structural_computation_populates_the_same_cross_model_lookup"),
    ("test_checkpoint_hydration.py", "test_new_process_hydrates_complete_structural_state_and_retries_cross_model"),
    ("test_checkpoint_hydration.py", "test_manifest_provenance_hydrates_exact_and_legacy_model_load_markers"),
    ("test_checkpoint_hydration.py", "test_empty_promoted_structural_csv_never_enters_postprocessing_only_resume"),
    ("test_bootstrap_performance.py", "test_indexed_bootstrap_matches_reference_draws_cis_and_gate_direction"),
    ("test_bootstrap_performance.py", "test_indexed_spearman_bootstrap_matches_reference_sampling_semantics"),
    ("test_bootstrap_performance.py", "test_indexed_bootstrap_core_like_fixture_is_materially_faster"),
    ("test_c7.py", "test_c7_identity_inverse_r7_and_composition"),
    ("test_d4.py", "test_d4_relations_noncommutativity_inverse_and_composition"),
    ("test_composition_order.py", "test_row_vector_composition_order_c7_and_d4"),
    ("test_composition_order.py", "test_wrong_d4_order_fails"),
    ("test_template_split.py", "test_template_splits_have_no_leakage"),
    ("test_template_split.py", "test_final_three_way_template_splits_are_exact_disjoint_and_deterministic"),
    ("test_template_split.py", "test_final_template_pools_have_one_placeholder_and_no_normalized_duplicates"),
    ("test_pca.py", "test_pca_fit_uses_training_rows_only"),
    ("test_pca.py", "test_validation_and_test_rows_cannot_change_training_pca"),
    ("test_pca.py", "test_d4_smoke_centered_pca_rank_supports_16_but_rejects_20"),
    ("test_pca.py", "test_observed_centered_rank_is_checked_before_pca_fit"),
    ("test_main_experiment_contract.py", "test_main_config_is_reproducible_and_exact"),
    ("test_main_experiment_contract.py", "test_smoke_tier_exercises_every_stage_contract"),
    ("test_main_experiment_contract.py", "test_manifest_complete_and_output_tree_canonical"),
    ("test_main_experiment_contract.py", "test_activation_cache_key_and_round_trip"),
    ("test_main_experiment_contract.py", "test_intervention_preserves_outside_subspace_residual"),
    ("test_main_experiment_contract.py", "test_mean_difference_baseline_uses_training_rows_only"),
    ("test_main_experiment_contract.py", "test_hierarchical_bootstrap_and_pairing_reproducible"),
    ("test_main_experiment_contract.py", "test_cross_model_and_irrep_diagnostics_exact_cases"),
    ("test_main_experiment_contract.py", "test_archive_is_never_imported_by_active_runtime"),
    ("test_main_experiment_contract.py", "test_protocol_documents_cycle_and_terminal_layer_rules"),
    ("test_final_runtime_contracts.py", "test_seed_namespaces_are_separate_and_deterministic_full"),
    ("test_final_runtime_contracts.py", "test_smoke_uses_approved_rank_feasible_primary_dimension"),
    ("test_final_runtime_contracts.py", "test_terminal_layers_are_never_causal_and_pair_matrix_is_exact"),
    ("test_final_runtime_contracts.py", "test_d4_srs_exact_toy_case_and_reversed_order_failure"),
    ("test_final_runtime_contracts.py", "test_stale_cache_metadata_is_rejected"),
    ("test_final_runtime_contracts.py", "test_checkpoint_resume_skips_verified_and_recomputes_corrupt_output"),
    ("test_final_runtime_contracts.py", "test_clean_tree_full_guard_and_stable_artifact_names"),
    ("test_final_runtime_contracts.py", "test_final_artifact_schemas_are_generated_from_saved_tables"),
    ("test_final_runtime_contracts.py", "test_finite_smoke_telemetry_produces_finite_preliminary_full_projection"),
    ("test_final_runtime_contracts.py", "test_nonfinite_runtime_timing_is_rejected_before_projection"),
    ("test_final_runtime_contracts.py", "test_completed_smoke_scientific_ledger_enters_postprocessing_only_resume"),
    ("test_final_runtime_contracts.py", "test_result_packaging_telemetry_preserves_manifest_checksum_integrity"),
    ("test_pre_full_revision.py", "test_balanced_orbit_global_element_average_cancels_but_source_conditioned_does_not"),
    ("test_pre_full_revision.py", "test_source_conditioned_additive_is_train_only_and_uses_exact_d4_action"),
    ("test_pre_full_revision.py", "test_deterministic_duplicates_do_not_change_weight_but_random_seeds_remain"),
    ("test_pre_full_revision.py", "test_joint_vs_independent_ridge_paired_inference_is_emitted"),
    ("test_pre_full_revision.py", "test_depth_gate_pairs_deterministic_learned_rows_with_random_replications"),
    ("test_pre_full_revision.py", "test_gate_wording_distinguishes_not_evaluable_and_nonsignificant_association"),
    ("test_pre_full_revision.py", "test_d4_word_products_distinguish_true_relation_and_wrong_order"),
    ("test_pre_full_revision.py", "test_d4_true_relation_is_paired_against_wrong_order_distance"),
    ("test_pre_full_revision.py", "test_pre_full_and_revised_full_workload_matrices_are_exact"),
    ("test_pre_full_revision.py", "test_revised_full_gate_accepts_pre_full_and_rejects_historical_core"),
    ("test_pre_full_revision.py", "test_packaged_pre_full_authorization_is_explicit_and_self_contained"),
    ("test_pre_full_revision.py", "test_full_compound_checkpoint_keys_bind_controls_words_and_roles_only_prospectively"),
    ("test_pre_full_revision.py", "test_c7_transformation_and_cycle_have_separate_figure_sources"),
]


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> int:
    root = Path(__file__).parent
    modules, total = {}, 0
    for filename, name in RUNTIME_CHECKS:
        module = modules.setdefault(filename, _load_module(root / filename))
        getattr(module, name)()
        print(f"PASS {filename}::{name}")
        total += 1
    print(f"{total} RunPod runtime checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
