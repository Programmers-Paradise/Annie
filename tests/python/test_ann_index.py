import numpy as np
import pytest

from rust_annie import AnnIndex, Distance

from _helpers import euclidean, exact_knn


def test_search_matches_numpy(dataset):
    data, ids, queries = dataset
    idx = AnnIndex(16, Distance.EUCLIDEAN)
    idx.add(data, ids)
    for q in queries:
        got_ids, got_d = idx.search(q, k=10)
        want_ids, want_d = exact_knn(data, ids, q, 10, euclidean)
        np.testing.assert_array_equal(got_ids, want_ids)
        np.testing.assert_allclose(got_d, want_d, rtol=1e-5)


def test_default_metric_is_euclidean():
    idx = AnnIndex(4)
    assert idx.metric == Distance.EUCLIDEAN


def test_result_types_and_shapes(dataset):
    data, ids, queries = dataset
    idx = AnnIndex(16)
    idx.add(data, ids)
    got_ids, got_d = idx.search(queries[0], k=5)
    assert got_ids.dtype == np.int64 and got_d.dtype == np.float32
    assert got_ids.shape == (5,) and got_d.shape == (5,)
    assert np.all(np.diff(got_d) >= 0)


def test_self_query_has_zero_distance(dataset):
    data, ids, _ = dataset
    idx = AnnIndex(16)
    idx.add(data, ids)
    got_ids, got_d = idx.search(data[7], k=1)
    assert got_ids[0] == ids[7]
    assert got_d[0] == 0.0


def test_batch_matches_single(dataset):
    data, ids, queries = dataset
    for metric in [Distance.EUCLIDEAN, Distance.COSINE, Distance.MANHATTAN]:
        idx = AnnIndex(16, metric)
        idx.add(data, ids)
        b_ids, b_d = idx.search_batch(queries, k=7)
        assert b_ids.shape == (20, 7) and b_d.shape == (20, 7)
        for i, q in enumerate(queries):
            s_ids, s_d = idx.search(q, k=7)
            np.testing.assert_array_equal(b_ids[i], s_ids)
            np.testing.assert_allclose(b_d[i], s_d, rtol=1e-6)


def test_k_larger_than_index():
    idx = AnnIndex(2)
    idx.add(np.eye(2, dtype=np.float32), np.array([3, 4]))
    got_ids, _ = idx.search(np.zeros(2), k=10)
    assert sorted(got_ids.tolist()) == [3, 4]
    b_ids, b_d = idx.search_batch(np.zeros((2, 2)), k=4)
    assert b_ids.shape == (2, 4)
    assert (b_ids[:, 2:] == -1).all() and np.isinf(b_d[:, 2:]).all()


def test_empty_index_returns_nothing():
    idx = AnnIndex(3)
    got_ids, got_d = idx.search(np.zeros(3), k=5)
    assert len(got_ids) == 0 and len(got_d) == 0


def test_accepts_float64_and_lists():
    idx = AnnIndex(3)
    idx.add([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], [0, 1])
    got_ids, _ = idx.search([0.9, 0.9, 0.9], k=1)
    assert got_ids[0] == 1
    idx.add(np.ones((1, 3)) * 5, np.array([2], dtype=np.int32))
    assert len(idx) == 3


def test_len_dim_contains_ids_repr(dataset):
    data, ids, _ = dataset
    idx = AnnIndex(16, Distance.COSINE)
    assert len(idx) == 0
    idx.add(data, ids)
    assert len(idx) == idx.len() == 500
    assert idx.dim() == 16
    assert 1000 in idx and 5 not in idx
    np.testing.assert_array_equal(idx.ids(), ids)
    assert "AnnIndex(dim=16, metric=cosine, size=500)" == repr(idx)


@pytest.mark.parametrize(
    "data, ids, exc",
    [
        (np.zeros((2, 3)), [0], ValueError),  # length mismatch
        (np.zeros((1, 4)), [0], ValueError),  # wrong dim
        (np.zeros((1, 3)), [-1], ValueError),  # negative id
        (np.zeros((2, 3)), [5, 5], ValueError),  # duplicate in batch
        (np.full((1, 3), np.nan), [0], ValueError),  # NaN
    ],
)
def test_add_validation(data, ids, exc):
    idx = AnnIndex(3)
    with pytest.raises(exc):
        idx.add(data, ids)
    assert len(idx) == 0


def test_duplicate_against_existing_is_atomic():
    idx = AnnIndex(2)
    idx.add(np.zeros((1, 2)), [1])
    with pytest.raises(ValueError, match="1"):
        idx.add(np.ones((2, 2)), [2, 1])
    assert len(idx) == 1 and 2 not in idx


def test_search_validation():
    idx = AnnIndex(3)
    idx.add(np.zeros((1, 3)), [0])
    with pytest.raises(ValueError):
        idx.search(np.zeros(2), k=1)
    with pytest.raises(ValueError):
        idx.search(np.zeros(3), k=0)
    with pytest.raises(ValueError):
        idx.search(np.array([np.inf, 0, 0]), k=1)
    with pytest.raises(ValueError):
        idx.search_batch(np.zeros((2, 4)), k=1)


@pytest.mark.parametrize("dim", [0, 70000])
def test_invalid_dimension(dim):
    with pytest.raises(ValueError):
        AnnIndex(dim)


def test_remove(dataset):
    data, ids, queries = dataset
    idx = AnnIndex(16)
    idx.add(data, ids)
    nearest = idx.search(queries[0], k=1)[0][0]
    idx.remove([int(nearest)])
    assert len(idx) == 499 and nearest not in idx
    assert nearest not in idx.search(queries[0], k=50)[0]
    with pytest.raises(KeyError):
        idx.remove([int(nearest)])
    with pytest.raises(KeyError):
        idx.remove([1001, 999999])
    assert 1001 in idx, "failed remove must not partially apply"


def test_remove_then_readd_same_id():
    idx = AnnIndex(2)
    idx.add(np.zeros((1, 2)), [7])
    idx.remove([7])
    idx.add(np.ones((1, 2)), [7])
    got_ids, got_d = idx.search(np.ones(2), k=1)
    assert got_ids[0] == 7 and got_d[0] == 0.0


def test_auto_compaction(dataset):
    data, ids, queries = dataset
    idx = AnnIndex(16)
    idx.set_max_deleted_ratio(0.1)
    idx.add(data, ids)
    idx.remove(ids[:40].tolist())
    assert idx.deleted_count() == 40 and idx.capacity() == 500
    idx.remove(ids[40:60].tolist())  # crosses 10% -> compacts
    assert idx.deleted_count() == 0 and idx.capacity() == 440
    idx.validate()
    want, _ = exact_knn(data[60:], ids[60:], queries[0], 5, euclidean)
    np.testing.assert_array_equal(idx.search(queries[0], k=5)[0], want)


def test_manual_compact():
    idx = AnnIndex(2)
    idx.add(np.eye(2), [0, 1])
    idx.set_max_deleted_ratio(1.0)
    idx.remove([0])
    assert idx.capacity() == 2
    idx.compact()
    assert idx.capacity() == 1
    idx.validate()


def test_update():
    idx = AnnIndex(2)
    idx.add(np.array([[0.0, 0.0], [10.0, 10.0]]), [0, 1])
    idx.update(1, [0.1, 0.1])
    assert idx.search([0.2, 0.2], k=1)[0][0] == 1
    with pytest.raises(KeyError):
        idx.update(99, [0.0, 0.0])
    with pytest.raises(ValueError):
        idx.update(1, [0.0, 0.0, 0.0])


def test_version_increments():
    idx = AnnIndex(2)
    v0 = idx.version()
    idx.add(np.zeros((1, 2)), [0])
    assert idx.version() > v0


def test_add_batch_with_progress(dataset):
    data, ids, _ = dataset
    idx = AnnIndex(16)
    calls = []
    idx.add_batch_with_progress(data, ids, lambda done, total: calls.append((done, total)), chunk_size=200)
    assert calls == [(200, 500), (400, 500), (500, 500)]
    assert len(idx) == 500


def test_get_info_and_validate(dataset):
    data, ids, _ = dataset
    idx = AnnIndex(16)
    idx.add(data, ids)
    info = idx.get_info()
    assert info["type"] == "brute" and info["size"] == "500" and info["metric"] == "euclidean"
    idx.validate()
