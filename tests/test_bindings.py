import numpy as np
from rust_annie import AnnIndex, Distance

def test_add_and_search():
    data = np.random.rand(100, 16).astype(np.float32)
    ids = np.arange(100, dtype=np.int64)

    idx = AnnIndex(16, Distance.EUCLIDEAN)
    idx.add(data, ids)

    query = data[0]
    results = idx.search(query, 5)
    assert len(results[0]) == 5
    assert isinstance(results[0][0], np.integer)
    
