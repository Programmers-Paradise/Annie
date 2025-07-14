# AnnIndex - Brute-force Nearest Neighbor Search

The `AnnIndex` class provides efficient brute-force nearest neighbor search with support for multiple distance metrics and additional features like filtered search. It leverages parallel processing capabilities for batch operations.

## Constructor

### `AnnIndex(dim: int, metric: Distance)`
Creates a new brute-force index for unit-variant metrics (Euclidean, Cosine, Manhattan, Chebyshev).

- `dim` (int): Vector dimension. Must be greater than 0.
- `metric` (Distance): Distance metric to use for similarity computation. Options: `Distance.Euclidean()`, `Distance.Cosine()`, `Distance.Manhattan()`, `Distance.Chebyshev()`.
- Returns: `AnnIndex`: A new empty index instance.
- Raises: `RustAnnError`: If dimension is 0 or invalid.

Example:
```python
from annindex import AnnIndex, Distance
index = AnnIndex(128, Distance.Euclidean())
index = AnnIndex(256, Distance.Cosine())
```

### `new_minkowski(dim: int, p: float)`
Creates a Minkowski distance index.

- `dim` (int): Vector dimension. Must be greater than 0.
- `p` (float): Minkowski exponent. Must be greater than 0. When p=1, equivalent to Manhattan distance. When p=2, equivalent to Euclidean distance.
- Returns: `AnnIndex`: A new empty index instance configured for Minkowski-p distance.
- Raises: `RustAnnError`: If dimension is 0 or p <= 0.

Example:
```python
index = AnnIndex.new_minkowski(128, 1.5)
index = AnnIndex.new_minkowski(64, 3.0)
```

### `new_with_metric(dim: int, metric_name: str)`
Creates a new index using a custom distance metric by name.

- `dim` (int): Vector dimension. Must be greater than 0.
- `metric_name` (str): Name of the distance metric to use. Can be built-in ("euclidean", "cosine", "manhattan", "chebyshev") or a custom metric registered via `register_metric()`.
- Returns: `AnnIndex`: A new empty index instance.
- Raises: `RustAnnError`: If dimension is 0 or invalid.

Example:
```python
index = AnnIndex.new_with_metric(128, "euclidean")
index = AnnIndex.new_with_metric(64, "my_custom_metric")
```

## Methods

### `add(data: ndarray, ids: ndarray)`
Add a batch of vectors with their corresponding IDs to the index.

- `data` (numpy.ndarray): N x dim array of vectors to add to the index. Each row represents a vector.
- `ids` (numpy.ndarray): N-dimensional array of integer IDs corresponding to each vector in data.
- Raises: `RustAnnError`: If data and ids have different lengths, or if any vector has incorrect dimension.

Example:
```python
import numpy as np
data = np.random.rand(100, 128).astype(np.float32)
ids = np.arange(100, dtype=np.int64)
index.add(data, ids)
```

### `remove(ids: List[int])`
Remove entries from the index by their IDs.

- `ids` (List[int]): List of IDs to remove from the index. IDs that don't exist in the index are ignored.

Example:
```python
index.remove([1, 5, 10])  # Remove vectors with IDs 1, 5, and 10
index.remove([])  # No-op for empty list
```

### `search(query: ndarray, k: int) -> Tuple[ndarray, ndarray]`
Search for the k nearest neighbors of a query vector.

- `query` (numpy.ndarray): Query vector with dimension matching the index. Should be a 1D array of float32 values.
- `k` (int): Number of nearest neighbors to return. Must be positive.
- Returns: Tuple[numpy.ndarray, numpy.ndarray]: A tuple containing:
  - neighbor_ids: Array of k nearest neighbor IDs (int64)
  - distances: Array of k corresponding distances (float32)
- Raises: `RustAnnError`: If query dimension doesn't match index dimension or if the index is empty.

Example:
```python
import numpy as np
query = np.random.rand(128).astype(np.float32)
neighbor_ids, distances = index.search(query, 10)
print(f"Found {len(neighbor_ids)} neighbors")
```

### `search_batch(queries: ndarray, k: int) -> Tuple[ndarray, ndarray]`
Batch search for k nearest neighbors for multiple query vectors.

- `queries` (numpy.ndarray): N x dim array of query vectors. Each row is a query.
- `k` (int): Number of nearest neighbors to return for each query.
- Returns: Tuple[numpy.ndarray, numpy.ndarray]: A tuple containing:
  - neighbor_ids: N x k array of neighbor IDs for each query (int64)
  - distances: N x k array of distances for each query (float32)
- Raises: `RustAnnError`: If query dimensions don't match index dimension, or if parallel processing fails.

Example:
```python
import numpy as np
queries = np.random.rand(50, 128).astype(np.float32)
neighbor_ids, distances = index.search_batch(queries, 5)
print(f"Shape: {neighbor_ids.shape}")  # (50, 5)
```

### `search_filter_py(query: ndarray, k: int, filter_fn: Callable[[int], bool]) -> Tuple[ndarray, ndarray]`
Search with ID filtering.

- `query` (dim-dimensional query vector)
- `k` (Maximum neighbors to return)
- `filter_fn` (Function that returns True for allowed IDs)
- Returns: (filtered IDs, filtered distances)

### `len() -> int`
Get the number of vectors currently stored in the index.

- Returns: int: The number of vectors in the index.

Example:
```python
len(index)  # This calls __len__ internally
1000
index.len()  # Direct method call
1000
```

### `dim() -> int`
Get the dimension of vectors stored in the index.

- Returns: int: The vector dimension.

Example:
```python
index.dim()
128
```

### `save(path: str)`
Save the index to a binary file.

- `path` (str): Base path for the saved file. The '.bin' extension will be automatically appended. The path must be relative and must not contain traversal sequences (e.g., "..").
- Raises: `RustAnnError`: If the file cannot be written, serialization fails, or the path is invalid.

Example:
```python
index.save("my_index")  # Saves to "my_index.bin"
index.save("relative/path/to/index")  # Saves to "relative/path/to/index.bin"
```

### `static load(path: str) -> AnnIndex`
Load an index from a binary file.

- `path` (str): Base path of the saved file. The '.bin' extension will be automatically appended. The path must be relative and must not contain traversal sequences (e.g., "..").
- Returns: `AnnIndex`: The loaded index instance with all vectors and configuration.
- Raises: `RustAnnError`: If the file cannot be read, deserialization fails, or the path is invalid.

Example:
```python
index = AnnIndex.load("my_index")  # Loads from "my_index.bin"
index = AnnIndex.load("relative/path/to/index")  # Loads from "relative/path/to/index.bin"
```

### `__len__() -> int`
Get the number of vectors in the index (implements len()).

- Returns: int: The number of vectors in the index.

Example:
```python
len(index)
1000
```

### `__repr__() -> str`
Get a string representation of the index.

- Returns: str: A descriptive string showing index statistics.

Example:
```python
print(index)
AnnIndex(dim=128, metric=Euclidean, entries=1000)
```

### `enable_metrics(port: int = None)`
Enable metrics collection for the index.

- `port` (int, optional): Port number for HTTP metrics server. If provided, starts an HTTP server.

Example:
```python
index.enable_metrics(8000)  # Start server on port 8000
index.enable_metrics()      # Enable metrics without HTTP server
```

### `get_metrics() -> dict`
Get current metrics as a Python dictionary.

- Returns: dict: Dictionary containing current metrics.

Example:
```python
metrics = index.get_metrics()
# Returns:
# {
#     'query_count': 150,
#     'avg_query_latency_us': 45.2,
#     'index_size': 1000,
#     'dimensions': 128,
#     'distance_metric': 'euclidean',
#     'uptime_seconds': 300,
#     'recall_estimates': {}
# }
```

### `update_recall_estimate(k: int, recall: float)`
Update recall estimate for a specific k value.

- `k` (int): The k value for nearest neighbor search
- `recall` (float): Estimated recall value (0.0 to 1.0)

Example:
```python
index.update_recall_estimate(10, 0.95)  # 95% recall for k=10
```

## Example
```python
import numpy as np
from rust_annie import AnnIndex, Distance

# Create index
index = AnnIndex(128, Distance.Euclidean())

# Add data
data = np.random.rand(1000, 128).astype(np.float32)
ids = np.arange(1000, dtype=np.int64)
index.add(data, ids)

# Check index properties
print("Number of entries:", index.len())
print("Dimension of vectors:", index.dim())

# Search
query = np.random.rand(128).astype(np.float32)
neighbor_ids, distances = index.search(query, k=5)
```

## Features

- **Multiple Backends**:
  - **Brute-force** (exact) with SIMD acceleration
  - **HNSW** (approximate) for large-scale datasets
- **Multiple Distance Metrics**: Euclidean, Cosine, Manhattan, Chebyshev, and custom metrics
- **Batch Queries** for efficient processing
- **Thread-safe** indexes with concurrent access
- **Zero-copy** NumPy integration
- **On-disk Persistence** with serialization
- **Filtered Search** with custom Python callbacks
- **GPU Acceleration** for brute-force calculations
- **Multi-platform** support (Linux, Windows, macOS)
- **Automated CI** with performance tracking
- **Real-time Metrics**: Query latency and index statistics with Prometheus integration

## Installation

```bash
# Stable release from PyPI:
pip install rust-annie

# Install with GPU support (requires CUDA):
pip install rust-annie[gpu]

# Or install from source:
git clone https://github.com/Programmers-Paradise/Annie.git
cd Annie
pip install maturin
maturin develop --release
```

## Quick Start

### Brute-Force Index
```python
import numpy as np
from rust_annie import AnnIndex, Distance

# Create index
index = AnnIndex(128, Distance.Euclidean())

# Add data
data = np.random.rand(1000, 128).astype(np.float32)
ids = np.arange(1000, dtype=np.int64)
index.add(data, ids)

# Check index properties
print("Number of entries:", index.len())
print("Dimension of vectors:", index.dim())

# Search
query = np.random.rand(128).astype(np.float32)
neighbor_ids, distances = index.search(query, k=5)
```

### HNSW Index
```python 
from rust_annie import PyHnswIndex

index = PyHnswIndex(dims=128)
data = np.random.rand(10000, 128).astype(np.float32)
ids = np.arange(10000, dtype=np.int64)
index.add(data, ids)

# Search
query = np.random.rand(128).astype(np.float32)
neighbor_ids, _ = index.search(query, k=10)
```

## Examples

### Brute-Force Index
```python
from rust_annie import AnnIndex, Distance
import numpy as np

# Create index
idx = AnnIndex(4, Distance.Cosine())

# Add data
data = np.random.rand(50, 4).astype(np.float32)
ids = np.arange(50, dtype=np.int64)
idx.add(data, ids)

# Check index properties
print("Number of entries:", idx.len())
print("Dimension of vectors:", idx.dim())

# Search
labels, dists = idx.search(data[10], k=3)
print(labels, dists)
```

### Batch Query
```python
from rust_annie import AnnIndex, Distance
import numpy as np

# Create index
idx = AnnIndex(16, Distance.Euclidean())

# Add data
data = np.random.rand(1000, 16).astype(np.float32)
ids = np.arange(1000, dtype=np.int64)
idx.add(data, ids)

# Check index properties
print("Number of entries:", idx.len())
print("Dimension of vectors:", idx.dim())

# Batch search
queries = data[:32]
labels_batch, dists_batch = idx.search_batch(queries, k=10)
print(labels_batch.shape)  # (32, 10)
```

### Thread-Safe Index
```python
from rust_annie import ThreadSafeAnnIndex, Distance
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Create thread-safe index
idx = ThreadSafeAnnIndex(32, Distance.Euclidean())

# Add data
data = np.random.rand(500, 32).astype(np.float32)
ids = np.arange(500, dtype=np.int64)
idx.add(data, ids)

# Check index properties
print("Number of entries:", idx.len())
print("Dimension of vectors:", idx.dim())

# Concurrent searches
def task(q):
    return idx.search(q, k=5)

queries = data[:10]
with ThreadPoolExecutor(max_workers=8) as executor:
    results = executor.map(task, queries)
    for f in results:
        print(f)
```

### Filtered Search
```python
from rust_annie import AnnIndex, Distance
import numpy as np

# Create index
index = AnnIndex(3, Distance.Euclidean())
data = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0]
], dtype=np.float32)
ids = np.array([10, 20, 30], dtype=np.int64)
index.add(data, ids)

# Check index properties
print("Number of entries:", index.len())
print("Dimension of vectors:", index.dim())

# Filter function
def even_ids(id: int) -> bool:
    return id % 2 == 0

# Filtered search
query = np.array([1.0, 2.0, 3.0], dtype=np.float32)
filtered_ids, filtered_dists = index.search_filter_py(
    query, 
    k=3, 
    filter_fn=even_ids
)
print(filtered_ids)  # [10, 30] (20 is filtered out)
```

### Custom Distance Metrics
```python
from rust_annie import AnnIndex, register_metric, list_metrics
import numpy as np

# Register a custom L1.5 distance metric
def l1_5_distance(a, b):
    return np.sum(np.abs(np.array(a) - np.array(b)) ** 1.5) ** (1.0 / 1.5)

register_metric("l1_5", l1_5_distance)

# Create index with custom metric
index = AnnIndex.new_with_metric(2, "l1_5")

# Add data
data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
ids = np.array([0, 1], dtype=np.int64)
index.add(data, ids)

# Search
query = np.array([1.5, 2.5], dtype=np.float32)
labels, distances = index.search(query, k=1)
print(labels, distances)
```

### Metrics and Monitoring
```python
import numpy as np
from rust_annie import AnnIndex, Distance

# Create an index
index = AnnIndex(128, Distance.Euclidean())

# Enable metrics with HTTP server on port 8000
index.enable_metrics(8000)

# Add some data
data = np.random.random((1000, 128)).astype(np.float32)
ids = np.arange(1000, dtype=np.int64)
index.add(data, ids)

# Run queries (metrics collected automatically)
query = np.random.random(128).astype(np.float32)
labels, distances = index.search(query, k=10)

# Check current metrics
metrics = index.get_metrics()
print(f"Query count: {metrics['query_count']}")
print(f"Average latency: {metrics['avg_query_latency_us']} μs")

# Access metrics via HTTP
# Get Prometheus format metrics
# curl http://localhost:8000/metrics

# Health check
# curl http://localhost:8000/health
```

## Build and Query a Brute-Force AnnIndex in Python (Complete Example)

This section demonstrates a complete, beginner-friendly example of how to build and query a `brute-force AnnIndex` using Python.

## Benchmark Results

Measured on a 6-core CPU:

That’s a \~4× speedup vs. NumPy!

| Operation	           | Dataset Size  | Time (ms) | Speedup vs Python | 
| -------------------- | ------------- | --------- | ----------------- | 
| Single Query (Brute) | 10,000 × 64   | 0.7	     | 4×                | 
| Batch Query (64)	   | 10,000 × 64   | 0.23	     | 12×               | 
| HNSW Query	         | 100,000 × 128 | 0.05	     | 56×               |

##### [View Full Benchmark Dashboard →](https://programmers-paradise.github.io/Annie/)

You’ll find:

## API Reference

### Core Classes

| Class              | Description                                |
| ------------------ | ------------------------------------------ |
| AnnIndex	         | Brute-force exact search                   |
| PyHnswIndex	       | Approximate HNSW index                     |
| ThreadSafeAnnIndex | 	Thread-safe wrapper for AnnIndex          |
| Distance           | 	Distance metrics (Euclidean, Cosine, etc) |

## Key Methods

| Method                                | Description                                | 
| ------------------------------------- | ------------------------------------------ |
| add(data, ids)	                      | Add vectors to index                       | 
| remove(ids)                           | Remove vectors from index by IDs           |
| search(query, k)	                    | Single query search                        | 
| search_batch(queries, k)              | Batch query search                         | 
| search_filter_py(query, k, filter_fn) | Filtered search                            | 
| len()                                 | Get the number of entries in the index     |
| dim()                                 | Get the dimension of vectors in the index  |
| save(path)                            | Save index to disk                         | 
| load(path)                            | Load index from disk                       | 
| enable_metrics(port)                  | Enable metrics collection                  |
| get_metrics()                         | Retrieve current metrics                   |
| update_recall_estimate(k, recall)     | Update recall estimate for a specific k    |

## Development & CI

**CI** runs on GitHub Actions, building wheels on Linux, Windows, macOS, plus:

* `benchmark.py` & `batch_benchmark.py` & `compare_results.py`

```bash
# Run tests
cargo test
pytest tests/

# Run benchmarks
python scripts/benchmark.py
python scripts/batch_benchmark.py

# Generate documentation
mkdocs build
```

CI pipeline includes:
  - Cross-platform builds (Linux, Windows, macOS)
  - Unit tests and integration tests
  - Performance benchmarking
  - Documentation generation

### Benchmark Automation

Benchmarks are tracked over time using:

## GPU Acceleration

### Enable GPU in Rust

Enable CUDA support for brute-force calculations:
```bash
# Install with GPU support
pip install rust-annie[gpu]

# Or build from source with GPU features
maturin develop --release --features gpu
```

Supported operations:
  - Batch L2 distance calculations
  - High-dimensional similarity search

Requirements:
  - NVIDIA GPU with CUDA support
  - CUDA Toolkit installed

## Contributing

Contributions are welcome! Please:

See [CONTRIBUTING.md](./CONTRIBUTING.md) for details.

## License

This project is licensed under the **MIT License**. See [LICENSE](./LICENSE) for details.