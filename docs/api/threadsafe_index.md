# ThreadSafeAnnIndex - Thread-safe Nearest Neighbor Index

The `ThreadSafeAnnIndex` class provides a thread-safe wrapper around `AnnIndex` for concurrent access.

## Constructor

### `ThreadSafeAnnIndex(dim: int, metric: Distance)`
Creates a new thread-safe index.

- `dim` (int): Vector dimension
- `metric` (Distance): Distance metric

### `from_arc(inner: Arc<RwLock<AnnIndex>>) -> ThreadSafeAnnIndex`
Internal constructor for testing: wraps an existing `Arc<RwLock<AnnIndex>>`.

- `inner` (Arc<RwLock<AnnIndex>>): The inner index wrapped in a thread-safe manner.

## Methods

### `add(data: ndarray, ids: ndarray)`
Thread-safe vector addition.

### `remove(ids: List[int])`
Thread-safe removal by IDs.

### `search(query: ndarray, k: int) -> Tuple[ndarray, ndarray]`
Thread-safe single query search.

### `search_batch(queries: ndarray, k: int) -> Tuple[ndarray, ndarray]`
Thread-safe batch search. This method now includes enhanced error handling for read locks. If a read lock cannot be acquired, a `RustAnnError` is raised with the message "Failed to acquire read lock for search_batch".

### `search_filter_py(query: ndarray, k: int, filter_fn: Callable[[int], bool]) -> Tuple[ndarray, ndarray]`
Thread-safe filtered search using a custom Python callback function.

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

# Filtered search
def even_ids(id: int) -> bool:
    return id % 2 == 0

query = np.random.rand(128).astype(np.float32)
filtered_ids, filtered_dists = index.search_filter_py(query, k=5, filter_fn=even_ids)
print(filtered_ids)
```

## Core Classes

| Class              | Description                                |
| ------------------ | ------------------------------------------ |
| AnnIndex	         | Brute-force exact search                   |
| PyHnswIndex	     | Approximate HNSW index                     |
| ThreadSafeAnnIndex | Thread-safe wrapper for AnnIndex           |
| Distance           | Distance metrics (Euclidean, Cosine, etc)  |

## Key Methods

| Method                                | Description                                | 
| ------------------------------------- | ------------------------------------------ |
| add(data, ids)	                    | Add vectors to index                       | 
| search(query, k)	                    | Single query search                        | 
| search_batch(queries, k)              | Batch query search                         | 
| search_filter_py(query, k, filter_fn) | Filtered search                            | 
| save(path)                            | Save index to disk                         | 
| load(path)                            | Load index from disk                       | 

## Development & CI

**CI** runs on GitHub Actions, building wheels on Linux, Windows, macOS, plus:

- `benchmark.py` & `batch_benchmark.py` & `compare_results.py`

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

- Continuous integration with performance tracking
- Automated scripts for consistent benchmarking

### GPU Acceleration

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

- Fork the repository
- Create a feature branch
- Submit a pull request

See [CONTRIBUTING.md](./CONTRIBUTING.md) for details.

## License

This project is licensed under the **MIT License**. See [LICENSE](./LICENSE) for details.