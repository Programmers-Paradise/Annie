# rust-annie

[![PyPI](https://img.shields.io/pypi/v/rust-annie.svg)](https://pypi.org/project/rust-annie)
[![Python versions](https://img.shields.io/pypi/pyversions/rust-annie.svg)](https://pypi.org/project/rust-annie)
[![CI](https://github.com/Programmers-Paradise/Annie/actions/workflows/ci.yml/badge.svg)](https://github.com/Programmers-Paradise/Annie/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Downloads](https://static.pepy.tech/badge/rust-annie)](https://pepy.tech/projects/rust-annie)

**rust-annie** is a Python library for exact and approximate *k*-nearest-neighbour
search, implemented in Rust. It provides:

- **Exact search** (`AnnIndex`) with nine built-in metrics — Euclidean, cosine,
  Manhattan, Chebyshev, Minkowski-*p*, angular, Hamming, Jaccard, Canberra — and
  **user-defined metrics written in Python** (e.g. Mahalanobis), parallelised across
  CPU cores with the GIL released.
- **Approximate search** (`PyHnswIndex`) using HNSW graphs for Euclidean, cosine and
  Manhattan distance, with a per-query accuracy/speed trade-off (`ef`).
- **Filtered search** by ID range, ID set, bitmask, boolean combinations of those, or any
  Python predicate.
- **Mutable indexes**: add, remove and update vectors by your own integer IDs.
- **Persistence** to a versioned binary format, and a **thread-safe** index for
  multi-threaded Python programs.

## Installation

```bash
pip install rust-annie
```

Pre-built wheels are available for Linux (x86-64, aarch64, musl), macOS (Intel and Apple
silicon) and Windows (x86-64) for Python ≥ 3.10. NumPy is the only runtime dependency.
To build from source you need a Rust toolchain (≥ 1.87):

```bash
git clone https://github.com/Programmers-Paradise/Annie.git
cd Annie
pip install maturin
maturin develop --release
```

## Quick start

```python
import numpy as np
from rust_annie import AnnIndex, Distance, PyHnswIndex

rng = np.random.default_rng(0)
data = rng.standard_normal((10_000, 128)).astype(np.float32)
ids = np.arange(10_000)

# Exact search
index = AnnIndex(128, Distance.COSINE)
index.add(data, ids)
neighbor_ids, distances = index.search(data[0], k=5)
assert neighbor_ids[0] == 0

# Approximate search: faster on large collections
hnsw = PyHnswIndex(128, metric=Distance.COSINE)
hnsw.add(data, ids)
neighbor_ids, distances = hnsw.search(data[0], k=5, ef=100)

# Many queries at once -> arrays of shape (n_queries, k)
batch_ids, batch_distances = index.search_batch(data[:32], k=10)
assert batch_ids.shape == (32, 10)
```

### Custom distance functions

Any Python function of two NumPy vectors can be used as the metric for exact search:

```python
import numpy as np
from rust_annie import AnnIndex, register_metric

cov_inv = np.linalg.inv(np.array([[2.0, 0.3], [0.3, 0.5]]))


def mahalanobis(a, b):
    d = a - b
    return float(np.sqrt(d @ cov_inv @ d))


register_metric("mahalanobis", mahalanobis)
index = AnnIndex.new_with_metric(2, "mahalanobis")
index.add(np.random.rand(1000, 2), np.arange(1000))
ids, dists = index.search([0.5, 0.5], k=3)
```

### Filtering, removal and persistence

```python
import numpy as np
from rust_annie import AnnIndex, Filter

index = AnnIndex(8)
index.add(np.random.rand(100, 8), np.arange(100))

ids, _ = index.search(np.zeros(8), k=5, filter=Filter.id_range(50, 99))
ids, _ = index.search(np.zeros(8), k=5, filter=lambda i: i % 2 == 0)

index.remove([3, 4, 5])
index.update(6, np.ones(8))

index.save("my_index.annie")
restored = AnnIndex.load("my_index.annie")
assert len(restored) == 97
```

## Documentation

- [User guide](https://github.com/Programmers-Paradise/Annie/blob/main/docs/user-guide.md): choosing an index, metrics, filters, threads,
  tuning HNSW, monitoring
- [API reference](https://github.com/Programmers-Paradise/Annie/blob/main/docs/api.md)
- [Benchmarks](https://github.com/Programmers-Paradise/Annie/blob/main/docs/benchmarks.md): methodology and how to reproduce
- [Examples](https://github.com/Programmers-Paradise/Annie/tree/main/examples/): runnable scripts
- [Changelog](https://github.com/Programmers-Paradise/Annie/blob/main/CHANGELOG.md)

## Performance

Exact search runs in Rust over contiguous memory with vectorised distance kernels,
in parallel across cores and without holding the GIL. In our measurements it is 2–9×
faster than NumPy and faster than FAISS's flat index for single queries; for large
batches of queries FAISS (which uses BLAS matrix products) is up to 10× faster.
`PyHnswIndex` builds on the [`hnsw_rs`](https://crates.io/crates/hnsw_rs) crate; it
reaches the same recall as hnswlib and FAISS but is 3–7× slower per query, so prefer
those libraries when raw HNSW throughput is what matters. See
[docs/benchmarks.md](https://github.com/Programmers-Paradise/Annie/blob/main/docs/benchmarks.md)
for the numbers and how to reproduce them with
[`benchmarks/compare.py`](https://github.com/Programmers-Paradise/Annie/blob/main/benchmarks/compare.py).

## Contributing

Bug reports, feature requests and pull requests are welcome. Please read
[CONTRIBUTING.md](https://github.com/Programmers-Paradise/Annie/blob/main/CONTRIBUTING.md) and our [Code of Conduct](https://github.com/Programmers-Paradise/Annie/blob/main/CODE_OF_CONDUCT.md).
For questions, open a [GitHub issue](https://github.com/Programmers-Paradise/Annie/issues).

## Citing

If you use rust-annie in published research, please cite it using the metadata in
[CITATION.cff](https://github.com/Programmers-Paradise/Annie/blob/main/CITATION.cff) (GitHub's "Cite this repository" button generates BibTeX).

## License

MIT. See [LICENSE](https://github.com/Programmers-Paradise/Annie/blob/main/LICENSE).
