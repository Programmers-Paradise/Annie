import numpy as np
from rust_annie import PyHnswIndex

def test_hnsw_basic():
    dim = 64
    index = PyHnswIndex(dims=dim)
    
    # Generate sample data
    data = np.random.rand(1000, dim).astype(np.float32)
    ids = np.arange(1000, dtype=np.int64)
    
    # Add to index
    index.add(data, ids)
    
    # Generate a random query
    query = np.random.rand(dim).astype(np.float32)
    
    # Search
    retrieved_ids = index.search(query, k=10)

    # Convert to numpy arrays if not already
    retrieved_ids = np.array(retrieved_ids)
    
    # Assertions
    assert retrieved_ids.shape == (10,)
    assert issubclass(retrieved_ids.dtype.type, np.integer)
