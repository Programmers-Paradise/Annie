import numpy as np
import pytest

from rust_annie import Distance, Filter, HnswConfig, PyHnswIndex

from _helpers import euclidean, exact_knn, recall


@pytest.fixture
def clustered(rng):
    centers = rng.standard_normal((20, 32)) * 5
    data = (centers[rng.integers(0, 20, 3000)] + rng.standard_normal((3000, 32))).astype(np.float32)
    ids = rng.permutation(np.arange(10_000, 13_000)).astype(np.int64)
    queries = (centers[rng.integers(0, 20, 50)] + rng.standard_normal((50, 32))).astype(np.float32)
    return data, ids, queries


def test_returns_user_ids_and_true_distances(clustered):
    data, ids, queries = clustered
    idx = PyHnswIndex(32)
    idx.add(data, ids)
    got_ids, got_d = idx.search(queries[0], k=10)
    assert set(got_ids.tolist()) <= set(ids.tolist())
    pos = {int(i): n for n, i in enumerate(ids)}
    for i, d in zip(got_ids, got_d):
        assert d == pytest.approx(euclidean(queries[0], data[pos[int(i)]]), rel=1e-4)
    assert np.all(np.diff(got_d) >= 0)


@pytest.mark.parametrize("metric", [Distance.EUCLIDEAN, Distance.COSINE, Distance.MANHATTAN])
def test_recall(clustered, metric):
    data, ids, queries = clustered
    ref = {
        Distance.EUCLIDEAN: euclidean,
        Distance.COSINE: lambda a, b: 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)),
        Distance.MANHATTAN: lambda a, b: float(np.abs(a - b).sum()),
    }[metric]
    idx = PyHnswIndex(32, metric=metric)
    idx.add(data, ids)
    r = np.mean(
        [recall(idx.search(q, k=10, ef=100)[0], exact_knn(data, ids, q, 10, ref)[0]) for q in queries]
    )
    assert r >= 0.9, f"recall@10 = {r:.3f}"


def test_higher_ef_does_not_reduce_recall(clustered):
    data, ids, queries = clustered
    idx = PyHnswIndex(32, m=8, ef_construction=40)
    idx.add(data, ids)
    truth = [exact_knn(data, ids, q, 10, euclidean)[0] for q in queries]
    low = np.mean([recall(idx.search(q, k=10, ef=10)[0], t) for q, t in zip(queries, truth)])
    high = np.mean([recall(idx.search(q, k=10, ef=400)[0], t) for q, t in zip(queries, truth)])
    assert high > low and high >= 0.95


def test_batch_matches_single(clustered):
    data, ids, queries = clustered
    idx = PyHnswIndex(32)
    idx.add(data, ids)
    b_ids, b_d = idx.search_batch(queries, k=5)
    assert b_ids.shape == (50, 5)
    for i in range(5):
        s_ids, s_d = idx.search(queries[i], k=5)
        np.testing.assert_array_equal(b_ids[i], s_ids)


def test_incremental_add_and_capacity_hint(rng):
    idx = PyHnswIndex(8, max_elements=10)
    for start in range(0, 100, 25):
        idx.add(rng.standard_normal((25, 8)), np.arange(start, start + 25))
    assert len(idx) == 100
    idx.validate()


def test_remove(clustered):
    data, ids, queries = clustered
    idx = PyHnswIndex(32)
    idx.add(data, ids)
    first = idx.search(queries[0], k=5)[0]
    idx.remove(first.tolist())
    assert len(idx) == 2995
    again = idx.search(queries[0], k=5)[0]
    assert not set(first.tolist()) & set(again.tolist())
    assert len(again) == 5
    with pytest.raises(KeyError):
        idx.remove([int(first[0])])


def test_duplicate_and_invalid_input():
    idx = PyHnswIndex(2)
    idx.add(np.zeros((1, 2)), [1])
    with pytest.raises(ValueError):
        idx.add(np.ones((1, 2)), [1])
    with pytest.raises(ValueError):
        idx.add(np.ones((1, 3)), [2])
    with pytest.raises(ValueError):
        idx.search(np.zeros(2), k=0)


@pytest.mark.parametrize("kwargs", [{"m": 1}, {"m": 300}, {"ef_construction": 0}, {"ef_search": 0}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        PyHnswIndex(4, **kwargs)


def test_unsupported_metric():
    with pytest.raises(ValueError, match="HNSW supports"):
        PyHnswIndex(4, metric=Distance.CHEBYSHEV)


def test_config_object():
    cfg = HnswConfig(m=8, ef_construction=50, ef_search=20)
    idx = PyHnswIndex(4, config=cfg)
    assert idx.config.m == 8 and idx.ef_search == 20
    idx.ef_search = 99
    assert idx.get_info()["ef_search"] == "99"
    with pytest.raises(ValueError):
        HnswConfig(m=0).validate()


def test_empty_search():
    ids, d = PyHnswIndex(3).search(np.zeros(3), k=3)
    assert len(ids) == 0


def test_filter_with_removed(clustered):
    data, ids, queries = clustered
    idx = PyHnswIndex(32)
    idx.add(data, ids)
    allowed = ids[ids % 2 == 0]
    idx.remove(allowed[:10].tolist())
    got = idx.search(queries[0], k=10, filter=Filter.id_set(allowed.tolist()))[0]
    assert all(i % 2 == 0 for i in got)
    assert not set(got.tolist()) & set(allowed[:10].tolist())


@pytest.mark.parametrize("seed", range(20))
def test_selective_filter_is_exact(seed):
    """Regression: selective filters used to starve graph traversal and miss matches."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((2000, 8)).astype(np.float32)
    ids = np.arange(2000)
    idx = PyHnswIndex(8)
    idx.add(data, ids)
    q = data[0]
    keep = ids % 10 == 0
    want, _ = exact_knn(data[keep], ids[keep], q, 5, euclidean)
    got, _ = idx.search(q, k=5, filter=lambda i: i % 10 == 0)
    np.testing.assert_array_equal(got, want)
    got_b, _ = idx.search_batch(q[None, :], k=5, filter=Filter.id_set(ids[keep].tolist()))
    np.testing.assert_array_equal(got_b[0], want)
