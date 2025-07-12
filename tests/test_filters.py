import pytest
import numpy as np
from rust_annie import AnnIndex, Distance


@pytest.fixture
def sample_data():
    vecs = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    ids = np.array([10, 11, 12, 13, 14, 15], dtype=np.int64)
    return vecs, ids


def test_add_and_search(sample_data):
    vecs, ids = sample_data
    index = AnnIndex(dim=3, metric=Distance.Euclidean())
    index.add(vecs, ids)

    query = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    ids_out, dists = index.search(query, k=3)

    assert len(ids_out) == 3
    assert len(dists) == 3


def test_batch_search(sample_data):
    vecs, ids = sample_data
    index = AnnIndex(3, Distance.Euclidean())
    index.add(vecs, ids)

    queries = np.stack([vecs[0], vecs[1]])
    ids_out, dists = index.search_batch(queries, k=2)

    assert ids_out.shape == (2, 2)
    assert dists.shape == (2, 2)


def test_remove(sample_data):
    vecs, ids = sample_data
    index = AnnIndex(3, Distance.Cosine())
    index.add(vecs, ids)

    index.remove([11, 14])
    query = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    ids_out, _ = index.search(query, k=3)

    assert 11 not in ids_out
    assert 14 not in ids_out


def test_repr(sample_data):
    vecs, ids = sample_data
    index = AnnIndex(3, Distance.Cosine())
    index.add(vecs, ids)

    r = repr(index)
    assert "AnnIndex" in r
    assert "Cosine" in r
    assert "entries=6" in r


def test_len(sample_data):
    vecs, ids = sample_data
    index = AnnIndex(3, Distance.Euclidean())
    index.add(vecs, ids)
    assert len(index) == 6
    assert index.__len__() == 6


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
    assert len(dists) == 2


def test_search_more_than_available(sample_data):
    vecs, ids = sample_data
    index = AnnIndex(dim=3, metric=Distance.Euclidean())
    index.add(vecs, ids)

    query = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    k = 10  # Requesting more than available
    ids_out, dists = index.search(query, k=min(k, len(vecs)))

    assert len(ids_out) <= len(vecs)
    assert len(dists) == len(ids_out)


def test_search_empty_index():
    index = AnnIndex(dim=3, metric=Distance.Euclidean())
    query = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    with pytest.raises(Exception, match="EmptyIndex"):
        index.search(query, k=2)


def test_load_invalid_path():
    with pytest.raises(Exception, match="Failed to open file"):
        AnnIndex.load("non_existent_index_file")
