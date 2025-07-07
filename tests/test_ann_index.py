from rust_annie import AnnIndex, Distance
import numpy as np
import pytest

def test_len_and_dim():
    dim = 3
    index = AnnIndex(dim, Distance.EUCLIDEAN)

    # Should be empty at the start
    assert index.len() == 0
    assert index.dim() == dim

    # Add a vector and test again
    vectors = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    ids = np.array([42], dtype=np.int64)
    index.add(vectors, ids)

    assert index.len() == 1
    assert index.dim() == dim

def test_search_batch_invalid_dimension_should_fail():
    index = AnnIndex(4, Distance.EUCLIDEAN)  # expects dim = 4
    index.add(np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32), np.array([1], dtype=np.int64))

    bad_query = np.array([[1.0, 2.0]], dtype=np.float32)  # dim = 2, should trigger failure

    with pytest.raises(ValueError) as e:  # Assuming ValueError is raised for invalid dimensions
        index.search_batch(bad_query, k=1)

    assert "Expected dimension 4" in str(e.value)
