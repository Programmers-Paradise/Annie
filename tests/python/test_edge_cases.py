"""Regression tests for edge cases that used to crash or behave inconsistently."""

import sys

import numpy as np
import pytest

from rust_annie import AnnIndex, Distance, Index, PyHnswIndex, ThreadSafeAnnIndex

HUGE_K = 10**12


def _indexes():
    return [AnnIndex(4), ThreadSafeAnnIndex(4), PyHnswIndex(4), Index("brute", 4), Index("hnsw", 4)]


@pytest.mark.parametrize("idx", _indexes(), ids=lambda i: type(i).__name__)
def test_huge_k_returns_all_matches(idx):
    # k far beyond the index size used to abort the Python process while
    # pre-allocating a buffer of size 2k.
    data = np.random.default_rng(0).random((10, 4), dtype=np.float32)
    idx.add(data, np.arange(10))
    ids, dists = idx.search(data[3], k=HUGE_K)
    assert len(ids) == 10 and ids[0] == 3 and np.all(np.diff(dists) >= 0)


@pytest.mark.parametrize("idx", _indexes(), ids=lambda i: type(i).__name__)
def test_huge_k_batch_raises_memory_error(idx):
    data = np.random.default_rng(0).random((10, 4), dtype=np.float32)
    idx.add(data, np.arange(10))
    with pytest.raises(MemoryError):
        idx.search_batch(data[:2], k=HUGE_K)
    ids, _ = idx.search_batch(data[:2], k=12)
    assert ids.shape == (2, 12) and np.all(ids[:, 10:] == -1)


def test_hnsw_huge_ef():
    idx = PyHnswIndex(4)
    idx.add(np.random.default_rng(0).random((10, 4)), np.arange(10))
    assert len(idx.search(np.zeros(4), k=3, ef=HUGE_K)[0]) == 3


def test_hnsw_cosine_rejects_zero_vectors():
    idx = PyHnswIndex(3, metric=Distance.COSINE)
    with pytest.raises(ValueError, match="zero vector"):
        idx.add(np.array([[1, 0, 0], [0, 0, 0]], dtype=np.float32), [1, 2])
    assert len(idx) == 0, "a rejected batch must leave the index unchanged"


def test_cosine_zero_query_consistent_across_indexes():
    data = np.random.default_rng(1).standard_normal((50, 3)).astype(np.float32)
    results = []
    for idx in (AnnIndex(3, Distance.COSINE), PyHnswIndex(3, metric=Distance.COSINE)):
        idx.add(data, np.arange(50))
        results.append(idx.search(np.zeros(3), k=5))
    for ids, dists in results:
        np.testing.assert_array_equal(ids, np.arange(5))  # all tie at 1; ties broken by ID
        np.testing.assert_allclose(dists, 1.0)


def test_threadsafe_has_inspection_methods():
    idx = ThreadSafeAnnIndex(2, Distance.MANHATTAN)
    idx.add(np.eye(2), [7, 8])
    assert idx.metric == Distance.MANHATTAN
    np.testing.assert_array_equal(idx.ids(), [7, 8])
    idx.set_max_deleted_ratio(1.0)
    idx.remove([7])
    assert idx.deleted_count() == 1 and idx.capacity() == 2
    assert idx.get_info()["size"] == "1"
    idx.validate()
    before = idx.version()
    idx.compact()
    assert idx.version() > before, "compact() must count as a mutation"
    assert idx.capacity() == 1


def test_save_is_atomic(tmp_path):
    path = tmp_path / "index.annie"
    idx = AnnIndex(2)
    idx.add(np.eye(2), [1, 2])
    idx.save(str(path))
    idx.add(np.ones((1, 2)), [3])
    idx.save(str(path))  # overwrite
    assert len(AnnIndex.load(str(path))) == 3
    assert [p.name for p in tmp_path.iterdir()] == ["index.annie"], "no temporary files left"
    # A save that fails part-way leaving the old file intact is tested in Rust
    # (storage::tests::failed_save_keeps_existing_file).


@pytest.mark.skipif(sys.platform == "win32", reason="Windows may refuse concurrent renames onto one file")
def test_concurrent_saves_to_same_path(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    path = str(tmp_path / "shared.annie")
    idx = ThreadSafeAnnIndex(8)
    idx.add(np.random.default_rng(0).random((2000, 8)), np.arange(2000))
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda _: idx.save(path), range(32)))
    assert len(ThreadSafeAnnIndex.load(path)) == 2000
    assert [p.name for p in tmp_path.iterdir()] == ["shared.annie"]
