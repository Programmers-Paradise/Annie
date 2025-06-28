import numpy as np
import pytest
from rust_annie import AnnIndex, Distance


@pytest.fixture
def sample_data():
    vecs = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    ids = np.array([10, 11, 12, 13, 14, 15], dtype=np.int64)
    return vecs, ids


def test_add_and_search(sample_data):
    vecs, ids = sample_data
    index = AnnIndex(dim=3, metric=Distance.Euclidean)
    index.add(vecs, ids)

    query = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    ids_out, dists = index.search(query, k=3)

    assert len(ids_out) == 3
    assert all(isinstance(i, (int, np.integer)) for i in ids_out)
    assert np.isclose(dists[0], 0.0)


def test_search_batch(sample_data):
    vecs, ids = sample_data
    index = AnnIndex(dim=3, metric=Distance.Manhattan)
    index.add(vecs, ids)

    queries = np.array([
        [1.0, 2.0, 3.0],
        [0.0, 0.0, 0.0]
    ], dtype=np.float32)

    ids_out, dists = index.search_batch(queries, k=2)
    assert ids_out.shape == (2, 2)
    assert dists.shape == (2, 2)


def test_filter_callback(sample_data):
    vecs, ids = sample_data
    index = AnnIndex(dim=3, metric=Distance.Cosine)
    index.add(vecs, ids)

    query = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    def only_even_ids(id):
        return id % 2 == 0

    ids_out, dists = index.search_filter_py(query, k=3, filter_fn=only_even_ids)

    assert len(ids_out) == len(dists)
    assert all(id % 2 == 0 for id in ids_out)
    assert all(isinstance(d, float) for d in dists)


def test_save_and_load(tmp_path, sample_data):
    vecs, ids = sample_data
    index = AnnIndex(dim=3, metric=Distance.Euclidean)
    index.add(vecs, ids)

    index.save(tmp_path / "test_index")

    loaded = AnnIndex.load(tmp_path / "test_index")
    query = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    ids_orig, _ = index.search(query, k=2)
    ids_loaded, _ = loaded.search(query, k=2)

    assert list(ids_orig) == list(ids_loaded)


def test_minkowski_distance():
    index = AnnIndex.new_minkowski(dim=3, p=3.0)
    vecs = np.array([
        [1.0, 2.0, 2.0],
        [2.0, 3.0, 1.0],
    ], dtype=np.float32)
    ids = np.array([1, 2], dtype=np.int64)
    index.add(vecs, ids)

    query = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    ids_out, dists = index.search(query, k=2)

    assert len(ids_out) == 2
    assert all(isinstance(d, float) for d in dists)


def test_search_more_than_available(sample_data):
    vecs, ids = sample_data
    index = AnnIndex.new(3, Distance.Euclidean)
    index.add(vecs, ids)

    query = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    ids_out, dists = index.search(query, k=10)
    assert len(ids_out) <= len(vecs)
    assert len(ids_out) == len(dists)


def test_search_empty_index():
    index = AnnIndex.new(3, Distance.Euclidean)
    query = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    with pytest.raises(RuntimeError):
        index.search(query, k=2)


def test_dimension_mismatch(sample_data):
    vecs, ids = sample_data
    index = AnnIndex.new(3, Distance.Euclidean)
    index.add(vecs, ids)

    bad_query = np.array([1.0, 2.0], dtype=np.float32)
    with pytest.raises(ValueError):
        index.search(bad_query, k=1)


def test_load_invalid_path():
    with pytest.raises(RuntimeError):
        AnnIndex.load("non_existent_index_file")
