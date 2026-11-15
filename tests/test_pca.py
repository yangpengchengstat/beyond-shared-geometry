import numpy as np

from bsg.representations.pca import PCASubspace
from bsg.experiments.fitting import ensure_observed_pca_rank, ensure_pca_rank


def test_pca_fit_uses_training_rows_only():
    train = np.arange(40, dtype=float).reshape(10, 4)
    all_rows = np.vstack([train, np.full((3, 4), 1000.0)])
    pca = PCASubspace(2, seed=42)
    pca.fit_transform_train_apply_all(train, all_rows)
    assert pca._fit_rows == len(train)


def test_validation_and_test_rows_cannot_change_training_pca():
    rng = np.random.default_rng(7)
    train = rng.normal(size=(20, 18))
    validation = rng.normal(size=(4, 18))
    test = rng.normal(size=(4, 18))
    a = PCASubspace(16).fit(train)
    b = PCASubspace(16).fit(train)
    b.transform(np.vstack([train, validation * 1000, test * -1000]))
    np.testing.assert_allclose(a.mean_, b.mean_)
    np.testing.assert_allclose(np.abs(a.components_), np.abs(b.components_))


def test_d4_smoke_centered_pca_rank_supports_16_but_rejects_20():
    ensure_pca_rank(16, n_train=5 * 4, n_features=2048)
    try:
        ensure_pca_rank(20, n_train=5 * 4, n_features=2048)
    except ValueError as error:
        assert "train PCA rank 19" in str(error)
    else:
        raise AssertionError("D4 smoke rank guard accepted an infeasible latent dimension")


def test_observed_centered_rank_is_checked_before_pca_fit():
    train = np.tile(np.arange(6, dtype=float), (20, 1))
    try:
        ensure_observed_pca_rank(1, train)
    except ValueError as error:
        assert "observed centered training rank 0" in str(error)
    else:
        raise AssertionError("rank-deficient training activations were accepted")
