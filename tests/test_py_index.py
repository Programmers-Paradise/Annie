import numpy as np
import pytest
from rust_annie import Index, Distance

def test_brute_index_add_search():
    index = Index("brute", 4, Distance.EUCLIDEAN)
    assert isinstance(index, Index)

    # Add data points
    index.add_item([1.0, 2.0, 3.0, 4.0])
    index.add_item([5.0, 6.0, 7.0, 8.0])
    index.build()

    # Search
    query = np.array([1.1, 2.1, 3.1, 4.1], dtype=np.float32)
    ids, dists = index.search(query, k=1)

    assert isinstance(ids, np.ndarray)
    assert isinstance(dists, np.ndarray)
    assert ids.shape == (1,)
    assert dists.shape == (1,)

def test_hnsw_index_add_search():
    index = Index("hnsw", 4, Distance.EUCLIDEAN)
    assert isinstance(index, Index)

    # Add data points
    index.add_item([1.0, 2.0, 3.0, 4.0])
    index.add_item([5.0, 6.0, 7.0, 8.0])
    index.build()

    # Search
    query = np.array([1.1, 2.1, 3.1, 4.1], dtype=np.float32)
    ids, dists = index.search(query, k=1)

    assert isinstance(ids, np.ndarray)
    assert isinstance(dists, np.ndarray)
    assert ids.shape == (1,)
    assert dists.shape == (1,)
    assert dists[0] == 0.0  # distances are mocked as 0.0 for HNSW currently

def test_invalid_backend():
    with pytest.raises(ValueError) as excinfo:
        Index("dummy", 4, Distance.EUCLIDEAN)
    assert "Unknown backend" in str(excinfo.value)
