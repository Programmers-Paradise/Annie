```markdown
# PyHnswIndex - Approximate Nearest Neighbors with HNSW

The `PyHnswIndex` class provides approximate nearest neighbor search using Hierarchical Navigable Small World (HNSW) graphs.

## Constructor

### `PyHnswIndex(dims: int)`
Creates a new HNSW index initialized with the Euclidean distance metric.

- `dims` (int): Vector dimension

## Methods

### `add(data: ndarray, ids: ndarray)`
Add vectors to the index.

- `data`: N×dims array of float32 vectors
- `ids`: N-dimensional array of int64 IDs

### `search(vector: ndarray, k: int) -> ndarray`
Search for k approximate nearest neighbors.

- `vector`: dims-dimensional query vector
- `k`: Number of neighbors to return
- Returns: Neighbor IDs

### `save(path: str)`
Save index to disk.

### `static load(path: str) -> PyHnswIndex`
Load index from disk (currently not implemented)

## New Features

### `user_ids` Field
The `HnswIndex` struct now includes a `user_ids` field, which stores user-defined IDs for the vectors. This allows for more flexible identification and retrieval of vectors within the index.

### `py_annindex` Macro
The `py_annindex` macro is used to automatically generate Python bindings for the HNSW index. It simplifies the creation of Python classes from Rust structs, ensuring that the HNSW index can be easily used within Python environments.

### Pluggable Distance Metric Registry
The library now supports a pluggable distance metric registry, allowing users to register custom distance metrics for use in the index. This feature enhances flexibility in defining how distances between vectors are calculated.

## Example
```python
import numpy as np
from rust_annie import PyHnswIndex, register_metric, list_metrics

# Create index
index = PyHnswIndex(dims=128)

# Add data
data = np.random.rand(10000, 128).astype(np.float32)
ids = np.arange(10000, dtype=np.int64)
index.add(data, ids)

# Search
query = np.random.rand(128).astype(np.float32)
neighbor_ids = index.search(query, k=10)

# Register a custom distance metric
def custom_metric(a, b):
    return np.sum(np.abs(a - b) ** 1.5) ** (1.0 / 1.5)

register_metric("custom_metric", custom_metric)

# List available metrics
print(list_metrics())
```

![Annie](https://github.com/Programmers-Paradise/.github/blob/main/ChatGPT%20Image%20May%2015,%202025,%2003_58_16%20PM.png?raw=true)

[![PyPI](https://img.shields.io/pypi/v/rust-annie.svg)](https://pypi.org/project/rust-annie)  
[![CI](https://img.shields.io/badge/Workflow-CI-white.svg)](https://github.com/Programmers-Paradise/Annie/blob/main/.github/workflows/CI.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Benchmark](https://img.shields.io/badge/benchmark-online-blue.svg)](https://programmers-paradise.github.io/Annie/)
[![GPU Support](https://img.shields.io/badge/GPU-CUDA-green.svg)](https://github.com/Programmers-Paradise/Annie/pull/53)
[![Documentation](https://img.shields.io/badge/docs-online-purple.svg)](https://programmers-paradise.github.io/Annie/)

A lightning-fast, Rust-powered Approximate Nearest Neighbor library for Python with multiple backends, thread-safety, and GPU acceleration.

## Table of Contents

1. [Features](#features)  
2. [Installation](#installation)  
3. [Quick Start](#quick-start)  
4. [Examples](#examples)  
   - [Brute-Force Index](#brute-force-index)  
   - [HNSW Index](#hnsw-index)  
   - [Thread-Safe Index](#thread-safe-index)  
5. [Benchmark Results](#benchmark-results)  
6. [API Reference](#api-reference)  
7. [Development & CI](#development--ci)  
8. [GPU Acceleration](#gpu-acceleration)
9. [Documentation](#documentation)
10. [Contributing](#contributing)  
11. [License](#license)

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
index = AnnIndex(128, Distance.EUCLIDEAN)

# Add data
data = np.random.rand(1000, 128).astype(np.float32)
ids = np.arange(1000, dtype=np.int64)
index.add(data, ids)

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
neighbor_ids = index.search(query, k=10)
```

## Examples

### Brute-Force Index
```python
from rust_annie import AnnIndex, Distance
import numpy as np

# Create index
idx = AnnIndex(4, Distance.COSINE)

# Add data
data = np.random.rand(50, 4).astype(np.float32)
ids = np.arange(50, dtype=np.int64)
idx.add(data, ids)

# Search
labels, dists = idx.search(data[10], k=3)
print(labels, dists)
```

### Batch Query
```python
from rust_annie import AnnIndex, Distance
import numpy as np

# Create index
idx = AnnIndex(16, Distance.EUCLIDEAN)

# Add data
data = np.random.rand(1000, 16).astype(np.float32)
ids = np.arange(1000, dtype=np.int64)
idx.add(data, ids)

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
idx = ThreadSafeAnnIndex(32, Distance.EUCLIDEAN)

# Add data
data = np.random.rand(500, 32).astype(np.float32)
ids = np.arange(500, dtype=np.int64)
idx.add(data, ids)

# Concurrent searches
def task(q):
    return idx.search(q, k=5)

with ThreadPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(task, data[i]) for i in range(10)]
    for f in futures:
        print(f.result())
```

### Filtered Search
```python
from rust_annie import AnnIndex, Distance
import numpy as np

# Create index
index = AnnIndex(3, Distance.EUCLIDEAN)
data = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0]
], dtype=np.float32)
ids = np.array([10, 20, 30], dtype=np.int64)
index.add(data, ids)

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

## Build and Query a Brute-Force AnnIndex in Python (Complete Example)

This section demonstrates a complete, beginner-friendly example of how to build and query a `brute-force AnnIndex` using Python.

Measured on a 6-core CPU:

That’s a \~4× speedup vs. NumPy!

| Operation	           | Dataset Size  | Time (ms) | Speedup vs Python | 
| -------------------- | ------------- | --------- | ----------------- | 
| Single Query (Brute) | 10,000 × 64   | 0.7	   | 4×                | 
| Batch Query (64)	   | 10,000 × 64   | 0.23	   | 12×               | 
| HNSW Query	       | 100,000 × 128 | 0.05	   | 56×               |

##### [View Full Benchmark Dashboard →](https://programmers-paradise.github.io/Annie/)

You’ll find:

## API Reference

### AnnIndex

- **Constructor**: `AnnIndex(dims: int, distance: Distance)`
- **Methods**: `add`, `search`, `search_batch`, `save`, `load`
- **Distance Metrics**: Enum: `Distance.EUCLIDEAN`, `Distance.COSINE`, `Distance.MANHATTAN`, `Distance.CHEBYSHEV`, and custom metrics

### ThreadSafeAnnIndex

Same API as `AnnIndex`, safe for concurrent use.

### Core Classes

| Class              | Description                                |
| ------------------ | ------------------------------------------ |
| AnnIndex	         | Brute-force exact search                   |
| PyHnswIndex	     | Approximate HNSW index                     |
| ThreadSafeAnnIndex | 	Thread-safe wrapper for AnnIndex          |
| Distance           | 	Distance metrics (Euclidean, Cosine, etc) |

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

### CUDA Support

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

ROCm (AMD GPU) support is not yet available.

### Enable GPU in Rust

Enable CUDA support for brute-force calculations:
```bash
# Install with GPU support
pip install rust-annie[gpu]

# Or build from source with GPU features
maturin develop --release --features gpu
```

## Contributing

Contributions are welcome! Please:

- Fork the repository
- Create a feature branch
- Submit a pull request

See [CONTRIBUTING.md](./CONTRIBUTING.md) for details.

## License

This project is licensed under the **MIT License**. See [LICENSE](./LICENSE) for details.
```