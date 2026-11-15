from __future__ import annotations

import time

import numpy as np
import pandas as pd

from bsg.statistics.hierarchical import (
    _clustered_draw,
    _clustered_draw_indexed,
    _prepare_clustered_index,
    _spearman,
    hierarchical_mean_ci,
    hierarchical_spearman_ci,
)


def _core_like_frame() -> pd.DataFrame:
    rng = np.random.default_rng(9)
    rows = []
    for split_index, split_seed in enumerate((101, 303, 909)):
        test_ids = np.arange(split_index * 6, split_index * 6 + 6)
        for optimization_seed in (42, 202, 777):
            for template_id in test_ids:
                for _matched_observation in range(56):
                    rows.append((split_seed, optimization_seed, int(template_id), float(rng.normal() - 0.2)))
    return pd.DataFrame(
        rows,
        columns=["split_seed", "optimization_seed", "template_id", "value"],
    )


def test_indexed_bootstrap_matches_reference_draws_cis_and_gate_direction():
    frame = _core_like_frame()
    index = _prepare_clustered_index(frame, ("value",))
    reference_rng = np.random.default_rng(314159)
    indexed_rng = np.random.default_rng(314159)
    reference = np.asarray([_clustered_draw(frame, "value", reference_rng) for _ in range(100)])
    indexed = np.asarray([_clustered_draw_indexed(index, indexed_rng) for _ in range(100)])
    np.testing.assert_allclose(indexed, reference, rtol=0, atol=1e-12)

    optimized = hierarchical_mean_ci(frame, "value", n_boot=100, seed=314159)
    assert optimized["mean"] == float(frame.value.mean())
    np.testing.assert_allclose(
        [optimized["ci_low"], optimized["ci_high"]],
        np.quantile(reference, [0.025, 0.975]),
        rtol=0,
        atol=1e-12,
    )
    assert (optimized["ci_high"] < 0) == (np.quantile(reference, 0.975) < 0)


def test_indexed_spearman_bootstrap_matches_reference_sampling_semantics():
    frame = _core_like_frame()
    frame["second_value"] = frame["value"] * 0.7 + np.sin(np.arange(len(frame)))
    rng = np.random.default_rng(314159)
    split_ids = np.array(sorted(frame.split_seed.unique()))
    reference = []
    for _ in range(40):
        pieces = []
        for split in rng.choice(split_ids, size=len(split_ids), replace=True):
            split_frame = frame[frame.split_seed == split]
            template_ids = np.array(sorted(split_frame.template_id.unique()))
            for template in rng.choice(template_ids, size=len(template_ids), replace=True):
                pieces.append(split_frame[split_frame.template_id == template])
        sampled = pd.concat(pieces, ignore_index=True)
        reference.append(_spearman(sampled.value, sampled.second_value))

    optimized = hierarchical_spearman_ci(
        frame, "value", "second_value", n_boot=40, seed=314159
    )
    np.testing.assert_allclose(
        [optimized["ci_low"], optimized["ci_high"]],
        np.quantile(reference, [0.025, 0.975]),
        rtol=0,
        atol=1e-12,
    )


def test_indexed_bootstrap_core_like_fixture_is_materially_faster():
    frame = _core_like_frame()
    index = _prepare_clustered_index(frame, ("value",))
    reference_rng = np.random.default_rng(17)
    indexed_rng = np.random.default_rng(17)

    started = time.perf_counter()
    for _ in range(20):
        _clustered_draw(frame, "value", reference_rng)
    reference_seconds = time.perf_counter() - started

    started = time.perf_counter()
    for _ in range(20):
        _clustered_draw_indexed(index, indexed_rng)
    indexed_seconds = time.perf_counter() - started

    assert indexed_seconds * 5 < reference_seconds
