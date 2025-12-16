# Rust-annie

![Annie](https://github.com/Programmers-Paradise/.github/blob/main/ChatGPT%20Image%20May%2015,%202025,%2003_58_16%20PM.png?raw=true)

[![PyPI](https://img.shields.io/pypi/v/rust-annie.svg)](https://pypi.org/project/rust-annie)
[![CodeQL](https://img.shields.io/github/actions/workflow/status/arnavk23/Annie/CodeQL.yml?branch=main&label=CodeQL)](https://github.com/arnavk23/Annie/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Benchmark](https://img.shields.io/badge/benchmark-online-blue.svg)](https://arnavk23.github.io/Annie/)
[![GPU Support](https://img.shields.io/badge/GPU-CUDA-green.svg)](./#gpu-acceleration)
[![PyPI Downloads](https://static.pepy.tech/badge/rust-annie)](https://pepy.tech/projects/rust-annie)

A lightning-fast, Rust-powered Approximate Nearest Neighbor library for Python with multiple backends, thread-safety, and GPU acceleration.

## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Examples](#examples)
5. [Benchmark Results](#benchmark-results)
6. [API Reference](#api-reference)
7. [Architecture](#architecture)
8. [Performance](#performance)
9. [GPU Acceleration](#gpu-acceleration)
10. [Contributing](#contributing)
11. [License](#license)

## Features

- **Multiple Backends**:
  - **Brute-force** (exact) with SIMD acceleration for guaranteed accuracy
  - **HNSW** (approximate) for large-scale datasets with near-constant memory
- **Multiple Distance Metrics**: Euclidean, Cosine, Manhattan, Chebyshev
- **Batch Queries** for efficient processing of multiple vectors
- **Thread-safe** indexes with concurrent access patterns
- **Zero-copy** NumPy integration for minimal memory overhead
- **On-disk Persistence** with serialization/deserialization
- **Filtered Search** with custom Python callbacks and metadata
- **GPU Acceleration** for brute-force calculations on NVIDIA GPUs
- **Multi-platform** support (Linux, Windows, macOS)
- **Automated CI/CD** with benchmarking and performance tracking

## Installation

### From PyPI (Recommended)

```bash
# Stable release
pip install rust-annie

# With GPU support (requires CUDA Toolkit)
pip install rust-annie[gpu]
```

### From Source

```bash
git clone https://github.com/arnavk23/Annie.git
cd Annie
pip install maturin
maturin develop --release
```

## Quick Start

### Brute-Force Index (Exact Search)

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
print(f"Top 5 neighbors: {neighbor_ids}")
print(f"Distances: {distances}")
```

### HNSW Index (Approximate, Scalable)

```python
from rust_annie import PyHnswIndex
import numpy as np

# Create index
index = PyHnswIndex(dims=128)

# Add data
data = np.random.rand(10000, 128).astype(np.float32)
ids = np.arange(10000, dtype=np.int64)
index.add(data, ids)

# Search
query = np.random.rand(128).astype(np.float32)
neighbor_ids, distances = index.search(query, k=10)
print(f"Approximate neighbors: {neighbor_ids}")
```

## Examples

### Batch Queries

```python
from rust_annie import AnnIndex, Distance
import numpy as np

index = AnnIndex(16, Distance.EUCLIDEAN)
data = np.random.rand(1000, 16).astype(np.float32)
ids = np.arange(1000, dtype=np.int64)
index.add(data, ids)

# Batch search (32 queries at once)
queries = data[:32]
labels_batch, dists_batch = index.search_batch(queries, k=10)
print(labels_batch.shape)  # (32, 10)
```

### Thread-Safe Index

```python
from rust_annie import ThreadSafeAnnIndex, Distance
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Create thread-safe index
index = ThreadSafeAnnIndex(32, Distance.EUCLIDEAN)
data = np.random.rand(500, 32).astype(np.float32)
ids = np.arange(500, dtype=np.int64)
index.add(data, ids)

# Concurrent searches
def search_task(q):
    return index.search(q, k=5)

with ThreadPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(search_task, data[i]) for i in range(8)]
    results = [f.result() for f in futures]
```

### Filtered Search

```python
from rust_annie import AnnIndex, Distance
import numpy as np

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
filtered_ids, filtered_dists = index.search_filter_py(query, k=3, filter_fn=even_ids)
print(filtered_ids)  # [10, 30] (20 is filtered out)
```

### Persistence (Save/Load)

```python
from rust_annie import AnnIndex, Distance
import numpy as np

# Create and populate index
index = AnnIndex(64, Distance.COSINE)
data = np.random.rand(5000, 64).astype(np.float32)
ids = np.arange(5000, dtype=np.int64)
index.add(data, ids)

# Save to disk
index.save("my_index.bin")

# Load later
loaded_index = AnnIndex.load("my_index.bin")
query = np.random.rand(64).astype(np.float32)
neighbors, distances = loaded_index.search(query, k=5)
```

## Benchmark Results

### Single Query Performance

| Operation | Dataset | Time | Speedup |
|-----------|---------|------|---------|
| Single Query | 10k × 64 | 0.7 ms | 4× vs NumPy |
| Batch Query (64) | 10k × 64 | 0.23 ms per query | 12× vs NumPy |
| HNSW Query | 100k × 128 | 0.05 ms | 56× vs NumPy |

See the [Live Benchmark Dashboard](https://arnavk23.github.io/Annie/) for continuous performance tracking across versions.

## API Reference

### AnnIndex

Brute-force exact nearest neighbor search.

```python
AnnIndex(dim: int, metric: Distance)
```

**Methods:**
- `add(data: np.ndarray[N×D], ids: np.ndarray[N]) -> None` - Add vectors to index
- `search(query: np.ndarray[D], k: int) -> (ids, distances)` - Single query search
- `search_batch(queries: np.ndarray[N×D], k: int) -> (ids, distances)` - Batch search
- `search_filter_py(query: np.ndarray[D], k: int, filter_fn: Callable) -> (ids, distances)` - Filtered search
- `remove(ids: Sequence[int]) -> None` - Remove vectors by ID
- `save(path: str) -> None` - Serialize to disk
- `load(path: str) -> AnnIndex` - Load from disk (static method)

### PyHnswIndex

Hierarchical Navigable Small World (HNSW) approximate search.

```python
PyHnswIndex(dims: int, ef_construction: int = 200, M: int = 5)
```

**Methods:**
- `add(data: np.ndarray[N×D], ids: np.ndarray[N]) -> None` - Add vectors
- `search(query: np.ndarray[D], k: int, ef: int = 200) -> (ids, distances)` - Search
- `save(path: str) -> None` - Serialize
- `load(path: str) -> PyHnswIndex` - Load (static method)

### ThreadSafeAnnIndex

Thread-safe wrapper for concurrent access.

```python
ThreadSafeAnnIndex(dim: int, metric: Distance)
```

Same API as `AnnIndex`, safe for multi-threaded use.

### Distance

Enum for distance metrics:
- `Distance.EUCLIDEAN` - L2 distance
- `Distance.COSINE` - Cosine similarity
- `Distance.MANHATTAN` - L1 distance
- `Distance.CHEBYSHEV` - L∞ distance

## Architecture

### Design Philosophy

- **Performance First**: Rust + SIMD for speed, PyO3 stable ABI for compatibility
- **Safety**: Thread-safe by design, validated input handling
- **Flexibility**: Multiple backends (brute-force, HNSW) for different use cases
- **Integration**: Zero-copy NumPy bindings, minimal Python overhead

### Project Structure

```
.
├── src/
│   ├── lib.rs                 # Main library bindings
│   ├── index.rs              # AnnIndex implementation
│   ├── hnsw_wrapper.rs       # HNSW wrapper
│   └── gpu/                  # GPU acceleration (CUDA)
├── .github/
│   ├── workflows/            # CI/CD pipelines
│   └── scripts/              # Automation scripts
├── scripts/                  # Benchmarking & testing
└── tests/                    # Integration tests
```

## Performance

### Single-Query Overhead

For small queries, Python function call overhead dominates. Use `.search_batch()` for multiple vectors.

### Batch Query Efficiency

Process 64+ queries together for near-optimal throughput:

```python
# ✓ Efficient: amortizes overhead
ids, dists = index.search_batch(queries_1000, k=5)

# ✗ Inefficient: repeats overhead
for q in queries_1000:
    ids, dists = index.search(q, k=5)
```

### Memory Usage

- **Brute-force**: O(N·D) where N=vectors, D=dimensions
- **HNSW**: ~O(N·D + N·M) where M=connectivity parameter (~5-15)

## GPU Acceleration

### Requirements

- NVIDIA GPU with CUDA compute capability 5.0+
- CUDA Toolkit 11.0+ installed
- cuBLAS libraries available

### Build with GPU Support

```bash
# From source with GPU
maturin develop --release --features gpu

# Or install pre-built wheels with GPU
pip install rust-annie[gpu]
```

### GPU Usage

Automatically used for:
- Batch L2 distance calculations
- High-dimensional searches (D > 256)

```python
from rust_annie import AnnIndex, Distance

# GPU acceleration is automatic for large batches
index = AnnIndex(512, Distance.EUCLIDEAN)
data = np.random.rand(100000, 512).astype(np.float32)
index.add(data, np.arange(len(data), dtype=np.int64))

# This will use GPU if available and beneficial
neighbors, distances = index.search_batch(queries, k=10)
```

## Development

### Local Setup

```bash
git clone https://github.com/arnavk23/Annie.git
cd Annie

# Install development dependencies
pip install maturin
cargo install cargo-watch

# Build and test
maturin develop
pytest tests/
```

### Running Tests

```bash
# Rust tests
cargo test --all

# Python tests
pytest tests/ -v

# Benchmarks
python scripts/benchmark.py --dataset medium
python scripts/dashboard.py
```

### CI/CD Pipeline

GitHub Actions automatically runs:
- Cross-platform builds (Linux, Windows, macOS)
- Comprehensive test suite
- Performance benchmarks
- CodeQL security scanning
- Type checking with mypy

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Add tests and documentation
4. Submit a pull request

See [CONTRIBUTING.md](./CONTRIBUTING.md) for detailed guidelines.

## License

This project is licensed under the **MIT License**. See [LICENSE](./LICENSE) for details.

---

**Built with ❤️ in Rust for Python**
