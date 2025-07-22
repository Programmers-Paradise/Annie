# Annie Documentation

Blazingly fast Approximate Nearest Neighbors in Rust

## Installation
```bash
pip install rust-annie

# Install with GPU support (requires CUDA):
pip install rust-annie[gpu]

# Or install from source:
git clone https://github.com/Programmers-Paradise/Annie.git
cd Annie
pip install maturin
maturin develop --release
```

## Basic Usage
```python
import numpy as np
from rust_annie import AnnIndex, Distance, Filter

# Create index
index = AnnIndex(128, Distance.EUCLIDEAN)

# Add data
data = np.random.rand(1000, 128).astype(np.float32)
ids = np.arange(1000, dtype=np.int64)
index.add(data, ids)

# Search with optional filter
query = np.random.rand(128).astype(np.float32)
neighbor_ids, distances = index.search(query, k=5)
```

## Key Features

- **Multiple Backends**:
  - **Brute-force** (exact) with SIMD acceleration
  - **HNSW** (approximate) for large-scale datasets
  - **GPU** (CUDA) for high-performance computations
- **Multiple Distance Metrics**: Euclidean, Cosine, Manhattan, Chebyshev
- **Batch Queries** for efficient processing
- **Batch Addition** of vectors for improved performance
- **Thread-safe** indexes with concurrent access
- **Zero-copy** NumPy integration
- **On-disk Persistence** with serialization
- **Filtered Search** with custom Python callbacks and built-in filters
- **GPU Acceleration** for brute-force calculations
- **Multi-platform** support (Linux, Windows, macOS)
- **Automated CI** with performance tracking
- **Fuzz Testing** for robustness and security
- **Versioning** for concurrent access and consistency
- **Handling of Deleted Entries** with auto-compaction
- **Enhanced Index Information Retrieval** with `get_info` method
- **Index Integrity Validation** with `validate` method

## Quick Start

### Brute-Force Index
```python
import numpy as np
from rust_annie import AnnIndex, Distance, Filter

# Create index
index = AnnIndex(128, Distance.EUCLIDEAN)

# Add data
data = np.random.rand(1000, 128).astype(np.float32)
ids = np.arange(1000, dtype=np.int64)
index.add(data, ids)

# Search with optional filter
query = np.random.rand(128).astype(np.float32)
neighbor_ids, distances = index.search(query, k=5)
```

### HNSW Index
```python 
from rust_annie import PyHnswIndex, PyHnswConfig
import numpy as np

# Create a configuration
config = PyHnswConfig(m=24, ef_construction=100, ef_search=128, max_elements=10000)
config.validate()

# Create the index using config
index = PyHnswIndex(dims=128, config=config)

# Add data
data = np.random.rand(10000, 128).astype(np.float32)
ids = np.arange(10000, dtype=np.int64)
index.add(data, ids)

# Search
query = np.random.rand(128).astype(np.float32)
neighbor_ids = index.search(query, k=10)
```

### GPU Index
```python
from rust_annie import Index, Distance
import numpy as np

# Create index with backend = "gpu"
index = Index("gpu", dim=128, metric=Distance.EUCLIDEAN)

# Add items one by one
index.add_item(np.random.rand(128).astype(np.float32))

# Search
query = np.random.rand(128).astype(np.float32)
ids, dists = index.search(query, k=5)
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

### Batch Addition
```python
from rust_annie import AnnIndex, Distance
import numpy as np

# Create index
idx = AnnIndex(16, Distance.EUCLIDEAN)

# Add data in batch
data = np.random.rand(1000, 16).astype(np.float32)
ids = np.arange(1000, dtype=np.int64)
idx.add(data, ids)

# Add more data in batch with progress reporting
def progress_callback(current, total):
    print(f"Progress: {current}/{total}")

idx.add_batch_with_progress(data, ids, progress_callback)
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

queries = data[:10]
with ThreadPoolExecutor(max_workers=8) as executor:
    results = executor.map(task, queries)
    for f in results:
        print(f)
```

### Filtered Search
```python
from rust_annie import AnnIndex, Distance, Filter
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

# Using built-in filters
id_range_filter = Filter.id_range(10, 20)
filtered_ids, filtered_dists = index.search(query, k=3, filter=id_range_filter)
print(filtered_ids)  # [10, 20]
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

| Class              | Description                                 |
| ------------------ | --------------------------------------------|
| AnnIndex	         |  Brute-force exact search                   |
| PyHnswIndex	       |  Approximate HNSW index                     |
| ThreadSafeAnnIndex | 	Thread-safe wrapper for AnnIndex           |
| Distance           | 	Distance metrics (Euclidean, Cosine, etc)  |
| Index              |Unified wrapper over AnnIndex and PyHnswIndex|
| PyHnswConfig       |Configurable struct for HNSW                 |
| Filter             |  Built-in filtering capabilities            |

### Utility Module

| Module | Description                                         |
| ------ | --------------------------------------------------- |
| utils  | Provides utility functions for distance computation |

### Key Methods

| Method                                | Description                                | 
| ----------------------------------------------------- | --------------------------------------------- |
| add(data, ids)	                                      | Add vectors to index                          | 
| add_batch_with_progress(data, ids, progress_callback) | Add vectors in batch with progress reporting  |
| remove(ids)                                           | Remove vectors by IDs                         |
| update(id, vector)                                    | Update vector by ID                           |
| compact()                                             | Compact index by removing deleted entries     |
| search(query, k, filter=None)                         | Single query search with optional filter      | 
| search_batch(queries, k, filter=None)                 | Batch query search with optional filter       | 
| search_filter_py(query, k, filter_fn)                 | Filtered search with Python callback          | 
| save(path)                                            | Save index to disk                            | 
| load(path)                                            | Load index from disk                          | 
| update_boolean_filter(name, bits)                     | Update a boolean filter by name               | 
| get_boolean_filter(name)                              | Retrieve a boolean filter by name             | 
| version()                                             | Get current version of the index              |
| get_info()                                            | Retrieve detailed information about the index |
| validate()                                            | Validate the integrity of the index           |

### Utility Functions

| Function                                                                          | Description                                             |
| --------------------------------------------------------------------------------- | ------------------------------------------------------- |
| compute_distances_with_ids(entries, query, query_sq_norm, metric, minkowski_p, k) | Computes distances and returns sorted IDs and distances |

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
  - Fuzz testing for robustness
  - Auto-release on version change

### Fuzz Testing

Fuzz testing is integrated to ensure robustness and security of the codebase. It runs automatically on GitHub Actions for each push and pull request.

```yaml
name: Fuzz Testing
on: [push, pull_request]

jobs:
  fuzz:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v3
        
      - name: Install Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
          components: rust-src
          
      - name: Install cargo-fuzz
        run: cargo install cargo-fuzz
        
      - name: Run fuzz tests
        run: |
          cd fuzz
          cargo fuzz run distances -- -max_total_time=60
        env:
          RUSTFLAGS: -C debug-assertions=no
          RUST_BACKTRACE: full
```

### Benchmark Automation

Benchmarks are tracked over time using:

- **Daily Benchmarking**: Automated daily benchmarks to track performance over time.
- **Dataset Variability**: Benchmarks now include small, medium, and large datasets.
- **Memory Usage Tracking**: Memory usage is tracked during index build and search operations.
- **Additional Libraries**: Benchmarks now compare against scikit-learn, FAISS, and Annoy.

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

### GPU Performance Optimization Guide

#### Memory Management
- Use `GpuMemoryPool` for buffer reuse
- Monitor usage with `memory_usage()`
- Pre-allocate buffers during initialization

#### Multi-GPU Setup
```rust
// Distribute across 4 GPUs
for device_id in 0..4 {
    set_active_device(device_id)?;
    // Add portion of dataset
}
```

#### Precision Selection
```rust
gpu_backend.set_precision(Precision::Fp16);  // 2x memory savings
```

#### Kernel Selection
We provide optimized kernels for:
- `l2_distance_fp32.ptx`
- `l2_distance_fp16.ptx`
- `l2_distance_int8.ptx`

#### Benchmark Results

Command: `cargo bench --features cuda`

Typical results on V100:
- FP32: 15ms @ 1M vectors
- FP16: 9ms @ 1M vectors
- INT8: 6ms @ 1M vectors (with quantization)

## Contributing

Contributions are welcome! Please:

- Fork the repository
- Create a feature branch
- Submit a pull request

See [CONTRIBUTING.md](./CONTRIBUTING.md) for details.

## License

This project is licensed under the **MIT License**. See [LICENSE](./LICENSE) for details.
