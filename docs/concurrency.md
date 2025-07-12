# Using `ThreadSafeAnnIndex` and `PyHnswIndex` for Concurrent Access

Annie exposes a thread-safe version of its ANN index (`AnnIndex`) for use in Python. This is useful when you want to perform parallel search or update operations from Python threads. Additionally, the `PyHnswIndex` class provides a Python interface to the HNSW index, which now includes enhanced data handling capabilities.

## Key Features

- Safe concurrent read access (`search`, `search_batch`)
- Exclusive write access (`add`, `remove`)
- Backed by Rust `RwLock` and exposed via PyO3
- `PyHnswIndex` supports mapping internal IDs to user IDs and handling vector data efficiently
- Enhanced error handling for read lock acquisition

## Example

```python
from annie import ThreadSafeAnnIndex, Distance
import numpy as np
import threading

# Create index
index = ThreadSafeAnnIndex(128, Distance.Cosine)

# Add vectors
data = np.random.rand(1000, 128).astype('float32')
ids = np.arange(1000, dtype=np.int64)
index.add(data, ids)

# Run concurrent searches
def run_search():
    query = np.random.rand(128).astype('float32')
    ids, distances = index.search(query, 10)
    print(ids)

threads = [threading.Thread(target=run_search) for _ in range(4)]
[t.start() for t in threads]
[t.join() for t in threads]

# Using PyHnswIndex
from rust_annie import PyHnswIndex

# Create HNSW index
hnsw_index = PyHnswIndex(dims=128)

# Add vectors to HNSW index
hnsw_index.add(data, ids)

# Search in HNSW index
query = np.random.rand(128).astype('float32')
user_ids, distances = hnsw_index.search(query, 10)
print(user_ids)
```

# CI/CD Pipeline for PyPI Publishing

The CI/CD pipeline for PyPI publishing has been updated to include parallel jobs for building wheels and source distributions across multiple operating systems and Python versions. This involves concurrency considerations that should be documented for users who are integrating or maintaining the pipeline.

## Pipeline Overview

The pipeline is triggered on pushes and pull requests to the `main` branch, as well as manually via `workflow_dispatch`. It includes the following jobs:

- **Test**: Runs on `ubuntu-latest` and includes steps for checking out the code, setting up Rust, caching dependencies, running tests, and checking code formatting.
- **Build Wheels**: Runs in parallel across `ubuntu-latest`, `windows-latest`, and `macos-latest` for Python versions 3.8, 3.9, 3.10, and 3.11. This job builds the wheels using `maturin` and uploads them as artifacts.
- **Build Source Distribution**: Runs on `ubuntu-latest` and builds the source distribution using `maturin`, uploading it as an artifact.
- **Publish to TestPyPI**: Publishes the built artifacts to TestPyPI if triggered via `workflow_dispatch` with the appropriate input.
- **Publish to PyPI**: Publishes the built artifacts to PyPI if triggered via `workflow_dispatch` with the appropriate input.

## Concurrency Considerations

- **Parallel Builds**: The `build-wheels` job utilizes a matrix strategy to run builds concurrently across different operating systems and Python versions. This reduces the overall build time but requires careful management of dependencies and environment setup to ensure consistency across platforms.
- **Artifact Management**: Artifacts from parallel jobs are downloaded and flattened before publishing to ensure all necessary files are available in a single directory structure for the publish steps.
- **Conditional Publishing**: Publishing steps are conditionally executed based on manual triggers and input parameters, allowing for flexible deployment strategies.

By understanding these concurrency considerations, users can effectively manage and extend the CI/CD pipeline to suit their specific needs.

# AnnIndex - Brute-force Nearest Neighbor Search

The `AnnIndex` class provides efficient brute-force nearest neighbor search with support for multiple distance metrics.

## Constructor

### `AnnIndex(dim: int, metric: Distance)`
Creates a new brute-force index.

- `dim` (int): Vector dimension
- `metric` (Distance): Distance metric (`EUCLIDEAN`, `COSINE`, `MANHATTAN`, `CHEBYSHEV`)

### `new_minkowski(dim: int, p: float)`
Creates a Minkowski distance index.

- `dim` (int): Vector dimension
- `p` (float): Minkowski exponent (p > 0)

## Methods

### `add(data: ndarray, ids: ndarray)`
Add vectors to the index.

- `data`: N×dim array of float32 vectors
- `ids`: N-dimensional array of int64 IDs

### `search(query: ndarray, k: int) -> Tuple[ndarray, ndarray]`
Search for k nearest neighbors.

- `query`: dim-dimensional query vector
- `k`: Number of neighbors to return
- Returns: (neighbor IDs, distances)

### `search_batch(queries: ndarray, k: int) -> Tuple[ndarray, ndarray]`
Batch search for multiple queries.

- `queries`: M×dim array of queries
- `k`: Number of neighbors per query
- Returns: (M×k IDs, M×k distances)
- Note: Enhanced error handling for dimension mismatches

### `search_filter_py(query: ndarray, k: int, filter_fn: Callable[[int], bool]) -> Tuple[ndarray, ndarray]`
Search with ID filtering.

- `query`: dim-dimensional query vector
- `k`: Maximum neighbors to return
- `filter_fn`: Function that returns True for allowed IDs
- Returns: (filtered IDs, filtered distances)

### `save(path: str)`
Save index to disk.

### `static load(path: str) -> AnnIndex`
Load index from disk.

## Example

```python
import numpy as np
from rust_annie import AnnIndex, Distance

# Create index
index = AnnIndex(128, Distance.EUCLIDEAN)

# Add data
data = np.random.rand(1000, 128).astype(np.float32)
ids = np.arange(1000, dtype=np.int64)
index.add(data, ids)

# Search
query = np.random.rand(128).astype(np.float32)
neighbor_ids, distances = index.search(query, k=5)
```

# PyHnswIndex - Approximate Nearest Neighbors with HNSW

The `PyHnswIndex` class provides approximate nearest neighbor search using Hierarchical Navigable Small World (HNSW) graphs.

## Constructor

### `PyHnswIndex(dims: int)`
Creates a new HNSW index.
- `dims` (int): Vector dimension

## Methods

### `add(data: ndarray, ids: ndarray)`
Add vectors to the index.
- `data`: N×dims array of float32 vectors
- `ids`: N-dimensional array of int64 IDs

### `search(vector: ndarray, k: int) -> Tuple[ndarray, ndarray]`
Search for k approximate nearest neighbors.
- `vector`: dims-dimensional query vector
- `k`: Number of neighbors to return
- Returns: (neighbor IDs, distances)

### `save(path: str)`
Save index to disk.

### `static load(path: str) -> PyHnswIndex`
Load index from disk (currently not implemented)

## Example

```python
import numpy as np
from rust_annie import PyHnswIndex

# Create index
index = PyHnswIndex(dims=128)

# Add data
data = np.random.rand(10000, 128).astype(np.float32)
ids = np.arange(10000, dtype=np.int64)
index.add(data, ids)

# Search
query = np.random.rand(128).astype(np.float32)
neighbor_ids, _ = index.search(query, k=10)
```

# ThreadSafeAnnIndex - Thread-safe Nearest Neighbor Index

The `ThreadSafeAnnIndex` class provides a thread-safe wrapper around `AnnIndex` for concurrent access.

## Constructor

### `ThreadSafeAnnIndex(dim: int, metric: Distance)`
Creates a new thread-safe index.

- `dim` (int): Vector dimension
- `metric` (Distance): Distance metric

### `from_arc(inner: Arc<RwLock<AnnIndex>>) -> ThreadSafeAnnIndex`
Internal constructor for testing: wraps an existing `Arc<RwLock<AnnIndex>>`.

- `inner` (Arc<RwLock<AnnIndex>>): The internal index wrapped in a thread-safe lock.

## Methods

### `add(data: ndarray, ids: ndarray)`
Thread-safe vector addition.

### `remove(ids: List[int])`
Thread-safe removal by IDs.

### `search(query: ndarray, k: int) -> Tuple[ndarray, ndarray]`
Thread-safe single query search.

### `search_batch(queries: ndarray, k: int) -> Tuple[ndarray, ndarray]`
Thread-safe batch search with enhanced error handling for lock acquisition.

### `save(path: str)`
Thread-safe save.

### `static load(path: str) -> ThreadSafeAnnIndex`
Thread-safe load.

## Example

```python
import numpy as np
from rust_annie import ThreadSafeAnnIndex, Distance
from concurrent.futures import ThreadPoolExecutor

# Create index
index = ThreadSafeAnnIndex(128, Distance.COSINE)

# Add data from multiple threads
with ThreadPoolExecutor() as executor:
    for i in range(4):
        data = np.random.rand(250, 128).astype(np.float32)
        ids = np.arange(i*250, (i+1)*250, dtype=np.int64)
        executor.submit(index.add, data, ids)

# Concurrent searches
with ThreadPoolExecutor() as executor:
    futures = []
    for _ in range(10):
        query = np.random.rand(128).astype(np.float32)
        futures.append(executor.submit(index.search, query, k=5))
    
    for future in futures:
        ids, dists = future.result()
```

# Annie Examples

## Basic Usage

```python
import numpy as np
from rust_annie import AnnIndex, Distance

# Create index
index = AnnIndex(128, Distance.EUCLIDEAN)

# Generate and add data
data = np.random.rand(1000, 128).astype(np.float32)
ids = np.arange(1000, dtype=np.int64)
index.add(data, ids)

# Single query
query = np.random.rand(128).astype(np.float32)
neighbor_ids, distances = index.search(query, k=5)

# Batch queries
queries = np.random.rand(10, 128).astype(np.float32)
batch_ids, batch_dists = index.search_batch(queries, k=3)
```

## Filtered Search

```python
# Create index with sample data
index = AnnIndex(3, Distance.EUCLIDEAN)
data = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0]
], dtype=np.float32)
ids = np.array([10, 20, 30], dtype=np.int64)
index.add(data, ids)

# Define filter function
def even_ids(id: int) -> bool:
    return id % 2 == 0

# Filtered search
query = np.array([1.0, 2.0, 3.0], dtype=np.float32)
filtered_ids, filtered_dists = index.search_filter_py(query, k=3, filter_fn=even_ids)
# Only IDs 10 and 30 will be returned (20 is odd)
```

## HNSW Index

```python
from rust_annie import PyHnswIndex

# Create HNSW index
index = PyHnswIndex(dims=128)

# Add large dataset
data = np.random.rand(100000, 128).astype(np.float32)
ids = np.arange(100000, dtype=np.int64)
index.add(data, ids)

# Fast approximate search
query = np.random.rand(128).astype(np.float32)
neighbor_ids, _ = index.search(query, k=10)
```

## Saving and Loading

```python
# Create and save index
index = AnnIndex(64, Distance.COSINE)
data = np.random.rand(500, 64).astype(np.float32)
ids = np.arange(500, dtype=np.int64)
index.add(data, ids)
index.save("my_index")

# Load index
loaded_index = AnnIndex.load("my_index")
```

## Thread-safe Operations

```python
from rust_annie import ThreadSafeAnnIndex, Distance
from concurrent.futures import ThreadPoolExecutor

index = ThreadSafeAnnIndex(256, Distance.MANHATTAN)

# Concurrent writes
with ThreadPoolExecutor() as executor:
    for i in range(10):
        data = np.random.rand(100, 256).astype(np.float32)
        ids = np.arange(i*100, (i+1)*100, dtype=np.int64)
        executor.submit(index.add, data, ids)

# Concurrent reads
with ThreadPoolExecutor() as executor:
    futures = []
    for _ in range(100):
        query = np.random.rand(256).astype(np.float32)
        futures.append(executor.submit(index.search, query, k=3))
    
    results = [f.result() for f in futures]
```

## Minkowski Distance

```python
# Create index with custom distance
index = AnnIndex.new_minkowski(dim=64, p=2.5)
data = np.random.rand(200, 64).astype(np.float32)
ids = np.arange(200, dtype=np.int64)
index.add(data, ids)

# Search with Minkowski distance
query = np.random.rand(64).astype(np.float32)
ids, dists = index.search(query, k=5)
```

# Filtering

## Why Filtering?

Filters allow you to narrow down search results dynamically based on:
- Metadata (e.g., tags, IDs, labels)
- Numeric thresholds (e.g., only items above/below a value)
- Custom user-defined logic

This improves both precision and flexibility of search.

### Example: Python API

```python
from rust_annie import AnnIndex
import numpy as np

# 1. Create an index with vector dimension 128
index = AnnIndex(dimension=128)

# 2. Add data with metadata
vector0 = np.random.rand(128).astype(np.float32)
vector1 = np.random.rand(128).astype(np.float32)

index.add_item(0, vector0, metadata={"category": "A"})
index.add_item(1, vector1, metadata={"category": "B"})

# 3. Define a filter function (e.g., only include items where category == "A")
def category_filter(metadata):
    return metadata.get("category") == "A"

# 4. Perform search with the filter applied
query_vector = np.random.rand(128).astype(np.float32)
results = index.search(query_vector, k=5, filter=category_filter)

print("Filtered search results:", results)
```

## Supported Filters

This library supports applying filters to narrow down ANN search results dynamically.

| Filter type         | Example                              |
|-------------------  |--------------------------------------|
| **Equals**          | `Filter.equals("category", "A")`     |
| **Greater than**    | `Filter.gt("score", 0.8)`            |
| **Less than**       | `Filter.lt("price", 100)`            |
| **Custom predicate**| `Filter.custom(lambda metadata: ...)`|

Filters work on the metadata you provide when adding items to the index.

## Sorting Behavior

The BruteForceIndex now uses `total_cmp` for sorting, which provides NaN-resistant sorting behavior. This change ensures that any NaN values in the data are handled consistently, preventing potential issues with partial comparisons.

## Benchmarking Indices

The library now includes a benchmarking function to evaluate the performance of different index types, specifically `PyHnswIndex` and `AnnIndex`. This function measures the average, maximum, and minimum query times, providing insights into the efficiency of each index type.

### Example: Benchmarking Script

```python
import numpy as np
import time
from rust_annie import PyHnswIndex, AnnIndex

def benchmark(index_cls, name, dim=128, n=10_000, q=100, k=10):
    print(f"\nBenchmarking {name} with {n} vectors (dim={dim})...")

    # Data
    data = np.random.rand(n, dim).astype(np.float32)
    ids = np.arange(n, dtype=np.int64)
    queries = np.random.rand(q, dim).astype(np.float32)

    # Index setup
    index = index_cls(dims=dim)
    index.add(data, ids)

    # Warm-up + Timing
    times = []
    for i in range(q):
        start = time.perf_counter()
        _ = index.search(queries[i], k=k)
        times.append((time.perf_counter() - start) * 1000)

    print(f"  Avg query time: {np.mean(times):.3f} ms")
    print(f"  Max query time: {np.max(times):.3f} ms")
    print(f"  Min query time: {np.min(times):.3f} ms")

if __name__ == "__main__":
    benchmark(PyHnswIndex, "HNSW")
    benchmark(AnnIndex, "Brute-Force")
```

## Integration & Extensibility

- Filters are exposed from Rust to Python via **PyO3** bindings.
- New filters can be added by extending `src/filters.rs` in the Rust code.
- Filters integrate cleanly with the existing ANN index search logic, so adding or combining filters doesn't require changes in the core search API.

### See also

- Example usage: [`scripts/filter_example.py`](../scripts/filter_example.py)
- Unit tests covering filter behavior: [`tests/test_filters.py`](../tests/test_filters.py)
- Benchmarking script: [`scripts/compare.py`](../scripts/compare.py)

# Annie Documentation

Blazingly fast Approximate Nearest Neighbors in Rust

## Installation

```bash
pip install rust_annie
```

## Basic Usage

```python
import numpy as np
from rust_annie import AnnIndex, Distance

# Create index
index = AnnIndex(128, Distance.EUCLIDEAN)

# Add data
data = np.random.rand(1000, 128).astype(np.float32)
ids = np.arange(1000, dtype=np.int64)
index.add(data, ids)

# Search
query = np.random.rand(128).astype(np.float32)
neighbor_ids, distances = index.search(query, k=5)
```

## Key Features

- Multiple distance metrics
- CPU/GPU acceleration
- Thread-safe indexes
- Filtered search
- HNSW support