import numpy as np
from rust_annie import PyHnswIndex
import pytest
import os

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
# Not implemented yet
# def test_hnsw_save_load(tmp_path):
#     dim = 64
#     index = PyHnswIndex(dims=dim)
    
#     # Generate sample data
#     data = np.random.rand(1000, dim).astype(np.float32)
#     ids = np.arange(1000, dtype=np.int64)
    
#     # Add to index
#     index.add(data, ids)
    
#     # Save index
#     save_path = os.path.join(tmp_path, "test_index.bin")
#     index.save(save_path)
    
#     # Load index
#     loaded_index = PyHnswIndex.load(save_path)
    
#     # Verify same dimensions
#     assert loaded_index.dims() == dim
    
#     # Generate query
#     query = np.random.rand(dim).astype(np.float32)
    
#     # Search in both indexes
#     original_ids = index.search(query, k=10)
#     loaded_ids = loaded_index.search(query, k=10)
    
#     # Verify same results
#     np.testing.assert_array_equal(original_ids, loaded_ids)

# def test_path_sanitization():
    # with pytest.raises(Exception):
    #     index = PyHnswIndex(dims=64)
    #     index.save("../invalid_path")
        
    # with pytest.raises(Exception):
    #     index = PyHnswIndex.load("/etc/passwd")