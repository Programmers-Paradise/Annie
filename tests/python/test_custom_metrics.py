import numpy as np
import pytest

from rust_annie import AnnIndex, Distance, list_metrics, register_metric

from _helpers import exact_knn


def test_builtins_listed():
    assert {"euclidean", "cosine", "manhattan", "chebyshev"} <= set(list_metrics())


def test_custom_metric_receives_numpy_arrays():
    seen = []

    def metric(a, b):
        seen.append((type(a), a.dtype))
        return float(np.abs(a - b).sum())

    register_metric("test_receives_arrays", metric)
    idx = AnnIndex.new_with_metric(2, "test_receives_arrays")
    idx.add(np.eye(2), [0, 1])
    idx.search([1.0, 0.0], k=1)
    assert seen and seen[0] == (np.ndarray, np.float32)


def test_mahalanobis_matches_reference(rng):
    cov = np.array([[2.0, 0.5, 0.0], [0.5, 1.0, 0.2], [0.0, 0.2, 0.5]])
    vi = np.linalg.inv(cov)

    def mahalanobis(a, b):
        d = a.astype(np.float64) - b
        return float(np.sqrt(d @ vi @ d))

    register_metric("test_mahalanobis", mahalanobis)
    data = rng.standard_normal((100, 3)).astype(np.float32)
    ids = np.arange(100)
    idx = AnnIndex.new_with_metric(3, "test_mahalanobis")
    idx.add(data, ids)
    q = rng.standard_normal(3).astype(np.float32)
    got_ids, got_d = idx.search(q, k=5)
    want_ids, want_d = exact_knn(data, ids, q, 5, mahalanobis)
    np.testing.assert_array_equal(got_ids, want_ids)
    np.testing.assert_allclose(got_d, want_d, rtol=1e-5)
    b_ids, _ = idx.search_batch(np.stack([q, q]), k=5)
    np.testing.assert_array_equal(b_ids[1], want_ids)


def test_custom_clone_matches_builtin(dataset):
    data, ids, queries = dataset
    register_metric("test_l1_clone", lambda a, b: float(np.abs(a - b).sum()))
    builtin = AnnIndex(16, Distance.MANHATTAN)
    custom = AnnIndex(16, Distance.custom("test_l1_clone"))
    builtin.add(data, ids)
    custom.add(data, ids)
    for q in queries[:3]:
        b = builtin.search(q, k=5)
        c = custom.search(q, k=5)
        np.testing.assert_array_equal(b[0], c[0])
        np.testing.assert_allclose(b[1], c[1], rtol=1e-5)


def test_unregistered_metric_is_an_error():
    with pytest.raises(ValueError, match="register_metric"):
        AnnIndex.new_with_metric(2, "definitely_not_registered")
    idx = AnnIndex(2, Distance.custom("also_not_registered"))
    idx.add(np.zeros((1, 2)), [0])
    with pytest.raises(ValueError, match="register_metric"):
        idx.search([0.0, 0.0], k=1)


def test_exception_in_metric_propagates():
    def broken(a, b):
        raise ZeroDivisionError("boom")

    register_metric("test_broken", broken)
    idx = AnnIndex.new_with_metric(2, "test_broken")
    idx.add(np.zeros((1, 2)), [0])
    with pytest.raises(RuntimeError, match="boom"):
        idx.search([0.0, 0.0], k=1)


def test_register_validation():
    with pytest.raises(TypeError):
        register_metric("test_not_callable", 42)
    with pytest.raises(ValueError):
        register_metric("euclidean", lambda a, b: 0.0)
