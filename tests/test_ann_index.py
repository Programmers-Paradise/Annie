from rust_annie import AnnIndex, Distance
import numpy as np

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
